from collections.abc import Mapping
from itertools import chain
from typing import Final

from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.protocol_routing._mapping import infer_protocols
from litellm.protocol_routing._types import (
    ROUTE_TYPE_TO_PROTOCOL,
    ProtocolMismatchError,
    ProtocolRoutingMode,
    SupportedProtocol,
    get_protocol_routing_mode,
)

_MAPPING: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))


def _as_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING.validate_python(value)
    except ValidationError:
        return None


def _extract_provider(deployment: Mapping[str, object]) -> str | None:
    """Extract provider from deployment, falling back to model-prefix parsing.

    Most deployments leave litellm_params.custom_llm_provider unset and rely
    on the "provider/model" prefix in litellm_params.model. Strict-mode
    filtering needs the same resolution rule the rest of the router uses.
    """
    litellm_params: Final = _as_mapping(deployment.get("litellm_params"))
    if litellm_params is None:
        return None
    provider: Final = litellm_params.get("custom_llm_provider")
    if isinstance(provider, str) and provider:
        return provider
    model: Final = litellm_params.get("model")
    if not isinstance(model, str) or not model:
        return None
    import litellm

    try:
        _, resolved_provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:  # noqa: BLE001  # prefix parsing raises whatever the unknown-provider path throws, and no provider means no protocol claim
        return None
    return resolved_provider


def filter_deployments_by_protocol(
    healthy_deployments: list[dict[str, object]],  # mutable-ok: the router owns and re-reads this list
    route_type: str,
    model: str,
    mode: ProtocolRoutingMode | None = None,
) -> list[dict[str, object]]:  # mutable-ok: the router consumes the filtered list it asked for
    """Filter deployments by protocol compatibility in strict mode.

    In bridged mode (default), this is a no-op that returns all deployments.
    In strict mode, only returns deployments whose provider supports the
    requested protocol.

    Args:
        healthy_deployments: List of deployment dicts from router
        route_type: The route type (e.g., "anthropic_messages", "acompletion")
        model: The model name requested by the user
        mode: Optional override for protocol routing mode

    Returns:
        Filtered list of deployments (or all deployments in bridged mode)

    Raises:
        ProtocolMismatchError: In strict mode when no deployments support the protocol
    """
    if not healthy_deployments:
        return healthy_deployments

    requested_protocol: Final = ROUTE_TYPE_TO_PROTOCOL.get(route_type)
    if requested_protocol is None or requested_protocol == SupportedProtocol.LLM_PASSTHROUGH:
        return healthy_deployments

    effective_mode: Final = mode if mode is not None else get_protocol_routing_mode()
    if effective_mode == "bridged":
        return healthy_deployments

    inferred: Final = tuple(
        (
            deployment,
            infer_protocols(_extract_provider(deployment), _as_mapping(deployment.get("model_info"))),
        )
        for deployment in healthy_deployments
    )
    compatible: Final = tuple(deployment for deployment, protocols in inferred if requested_protocol in protocols)
    if not compatible:
        available: Final = sorted(
            {protocol.value for protocol in chain.from_iterable(protocols for _, protocols in inferred)}
        )
        raise ProtocolMismatchError(
            model=model,
            requested_protocol=requested_protocol.value,
            available_protocols=available,
        )

    return list(compatible)


def check_strict_protocol_for_provider(
    provider: str | None,
    requested_protocol: SupportedProtocol,
    model: str,
    mode: ProtocolRoutingMode | None = None,
) -> None:
    """Check if a provider supports the requested protocol in strict mode.

    This is called at the protocol conversion decision point (e.g., when
    deciding whether to use the Anthropic adapter for a non-Anthropic provider).

    Args:
        provider: The LLM provider name
        requested_protocol: The protocol being requested
        model: The model name (for error messages)
        mode: Optional override for protocol routing mode

    Raises:
        ProtocolMismatchError: In strict mode when provider doesn't support the protocol
    """
    effective_mode: Final = mode if mode is not None else get_protocol_routing_mode()
    if effective_mode == "bridged":
        return

    provider_protocols: Final = infer_protocols(provider)
    if requested_protocol not in provider_protocols:
        raise ProtocolMismatchError(
            model=model,
            requested_protocol=requested_protocol.value,
            available_protocols=sorted({protocol.value for protocol in provider_protocols}),
        )
