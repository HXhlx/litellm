from collections.abc import Mapping
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, NamedTuple, TypeAlias

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.responses.transformation import (
    STANDARD_MODE_WITHOUT_CONTEXT_REASONING_FIELD_VALUES,
    ReasoningFieldValues,
)
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.xai.common_utils import (
    XAIModelInfo,
    omit_tool_choice_without_tools,
    xai_as_mapping,
    xai_oauth_token_file,
    xai_reported_cost_in_usd,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as _LiteLLMLoggingObj,
    )

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_STR_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def _usage_restated_from_xai_ticks(usage: ResponseAPIUsage | None) -> ResponseAPIUsage | None:
    reported_cost: Final = xai_reported_cost_in_usd(getattr(usage, "cost_in_usd_ticks", None))
    if usage is None or reported_cost is None:
        return None
    return usage.model_copy(update=MappingProxyType({"cost": reported_cost}))


_NAMESPACE_TOOL_TYPE: Final = "namespace"
_NAMESPACE_SEPARATOR: Final = "__"

# qualified name -> (namespace, tool name as the client declared it)
InlinedToolNames: TypeAlias = Mapping[str, tuple[str, str]]

_TOOL_SEQUENCE: Final = TypeAdapter(tuple[object, ...], config=ConfigDict(hide_input_in_errors=True))


class _InlinedTools(NamedTuple):
    tools: tuple[object, ...]
    names: InlinedToolNames


def _qualified(namespace: str, name: str) -> str:
    return f"{namespace}{_NAMESPACE_SEPARATOR}{name}"


def _is_tool_group(value: object) -> bool:
    return isinstance(value, list)


def _as_tool_sequence(tools: object) -> tuple[object, ...]:
    try:
        return _TOOL_SEQUENCE.validate_python(tools)
    except ValidationError:
        return (tools,)


def _rename_member(member: object, namespace: str) -> tuple[object, tuple[str, tuple[str, str]] | None]:
    payload: Final = xai_as_mapping(member)
    if payload is None:
        return member, None
    name: Final = payload.get("name")
    if not isinstance(name, str):
        return member, None
    qualified: Final = _qualified(namespace, name)
    return {**payload, "name": qualified}, (qualified, (namespace, name))


def _expand_namespace_tool(tool: object) -> tuple[tuple[object, ...], tuple[tuple[str, tuple[str, str]], ...]]:
    """One inbound tool -> (tools to send, qualified-name back-references).

    A `namespace` group contributes its members under `<namespace>__<tool>`, the
    qualified name Codex itself uses for MCP tools when they are not namespaced,
    which keeps names unique across groups. Anything else passes through.
    """
    payload: Final = xai_as_mapping(tool)
    if payload is None or payload.get("type") != _NAMESPACE_TOOL_TYPE:
        return (tool,), ()

    namespace: Final = payload.get("name")
    members: Final = payload.get("tools")
    if not isinstance(namespace, str) or not namespace or not _is_tool_group(members):
        return (tool,), ()

    renamed: Final = tuple(_rename_member(member, namespace) for member in _as_tool_sequence(members))
    return tuple(out for out, _ in renamed), tuple(ref for _, ref in renamed if ref is not None)


def _inline_namespace_tools(tools: object) -> _InlinedTools:
    """Replace `namespace` tool groups with their member tools.

    xAI rejects `namespace` as a tool type, and Codex wraps every MCP server's
    tools plus its sub-agent tools in one, so a single group fails the whole
    request. The returned mapping lets the response side hand Codex back the
    `(namespace, name)` pair it declared the tool under.
    """
    expanded: Final = tuple(_expand_namespace_tool(tool) for tool in _as_tool_sequence(tools))
    return _InlinedTools(
        tools=tuple(chain.from_iterable(members for members, _ in expanded)),
        names=MappingProxyType(dict(chain.from_iterable(pairs for _, pairs in expanded))),
    )


class XAIResponsesAPIConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for XAI's Responses API.

    Inherits from OpenAIResponsesAPIConfig since XAI's Responses API is largely
    compatible with OpenAI's, with a few differences:
    - Has no `namespace` tool type, so Codex tool groups are inlined
    - Requires code_interpreter tools to have 'container' field removed
    - Recommends store=false when sending images

    Reference: https://docs.x.ai/docs/api-reference#create-new-response
    """

    def __init__(self) -> None:
        super().__init__()
        # Set per request in map_openai_params; the config instance is built per
        # call by ProviderConfigManager, so this never spans two requests.
        self._inlined_tool_names: InlinedToolNames = MappingProxyType({})

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.XAI

    def _transform_web_search_tool(self, tool: Mapping[str, object]) -> Mapping[str, object]:
        """
        Transform web_search tool to XAI format.

        XAI supports web_search with specific filters:
        - allowed_domains (max 5)
        - excluded_domains (max 5)
        - enable_image_understanding

        XAI does NOT support search_context_size (OpenAI-specific).

        Domains may come nested under 'filters' (the OpenAI/XAI documented shape) or flat on the tool.
        """
        xai_tool: Final[dict[str, object]] = {"type": "web_search"}

        if "search_context_size" in tool:
            verbose_logger.info(
                "XAI does not support 'search_context_size' parameter. Removing it from web_search tool."
            )

        nested_filters: Final = tool.get("filters")
        domains: Final = (
            _STR_MAPPING_ADAPTER.validate_python(nested_filters) if isinstance(nested_filters, Mapping) else tool
        )
        filters: Final = {key: domains[key] for key in ("allowed_domains", "excluded_domains") if key in domains}

        if filters:
            xai_tool["filters"] = filters

        if "enable_image_understanding" in tool:
            xai_tool["enable_image_understanding"] = tool["enable_image_understanding"]

        return xai_tool

    def _transform_x_search_tool(self, tool: Mapping[str, object]) -> Mapping[str, object]:
        """
        Transform x_search tool to XAI format.

        XAI supports x_search with specific parameters:
        - allowed_x_handles (max 10)
        - excluded_x_handles (max 10)
        - from_date (ISO8601: YYYY-MM-DD)
        - to_date (ISO8601: YYYY-MM-DD)
        - enable_image_understanding
        - enable_video_understanding
        """
        xai_tool: Final[dict[str, object]] = {"type": "x_search"}

        # Handle allowed_x_handles
        if "allowed_x_handles" in tool:
            allowed_handles: Final = tool["allowed_x_handles"]
            xai_tool["allowed_x_handles"] = allowed_handles

        # Handle excluded_x_handles
        if "excluded_x_handles" in tool:
            excluded_handles: Final = tool["excluded_x_handles"]
            xai_tool["excluded_x_handles"] = excluded_handles

        # Handle date range
        if "from_date" in tool:
            xai_tool["from_date"] = tool["from_date"]

        if "to_date" in tool:
            xai_tool["to_date"] = tool["to_date"]

        # Handle media understanding flags
        if "enable_image_understanding" in tool:
            xai_tool["enable_image_understanding"] = tool["enable_image_understanding"]

        if "enable_video_understanding" in tool:
            xai_tool["enable_video_understanding"] = tool["enable_video_understanding"]

        return xai_tool

    def _transform_tool(self, tool: object) -> object:
        payload: Final = xai_as_mapping(tool)
        if payload is None:
            return tool

        tool_type: Final = payload.get("type")
        if tool_type == "code_interpreter":
            # XAI supports code_interpreter but doesn't use the container field
            verbose_logger.debug("XAI: Transforming code_interpreter tool, removing container field")
            return {"type": "code_interpreter"}
        if tool_type == "web_search":
            verbose_logger.debug("XAI: Transforming web_search tool to XAI format")
            return self._transform_web_search_tool(payload)
        if tool_type == "x_search":
            verbose_logger.debug("XAI: Transforming x_search tool to XAI format")
            return self._transform_x_search_tool(payload)
        return tool

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map parameters for XAI Responses API.

        Handles XAI-specific transformations:
        1. Inlines `namespace` tool groups into qualified function tools
        2. Transforms code_interpreter tools to remove 'container' field
        3. Transforms web_search tools to XAI format (removes search_context_size, adds filters)
        4. Transforms x_search tools to XAI format
        """
        params: Final = dict(response_api_optional_params)

        if "metadata" in params:
            verbose_logger.debug("XAI Responses API does not support 'metadata' parameter. Dropping it.")
            params.pop("metadata")

        # Transform tools: inline `namespace` groups first, then per-type transforms
        if params.get("tools"):
            inlined: Final = _inline_namespace_tools(params["tools"])
            self._inlined_tool_names = inlined.names
            params["tools"] = [self._transform_tool(tool) for tool in inlined.tools]

        return omit_tool_choice_without_tools(params)

    def _restore_dict_item(self, item: Mapping[str, object]) -> Mapping[str, object]:
        name: Final = item.get("name")
        target: Final = self._inlined_tool_names.get(name) if isinstance(name, str) else None
        if target is None:
            return item

        namespace, restored = target
        return {**item, "name": restored, "namespace": namespace}

    def _restore_any_item(self, item: object) -> object:
        """Give a tool call back the `(namespace, name)` pair the client declared.

        Codex registers a namespaced tool under that pair and its router rejects
        the qualified name we had to send xAI, so the split has to be undone on
        the way back. Items are pydantic models on the non-streaming path and
        plain dicts inside streaming chunks.
        """
        payload: Final = xai_as_mapping(item)
        if payload is not None:
            return self._restore_dict_item(payload)

        name: Final = getattr(item, "name", None)  # pyright: ignore[reportAny]  # output items are pydantic models with no shared name field
        target: Final = self._inlined_tool_names.get(name) if isinstance(name, str) else None
        if target is None:
            return item

        namespace, restored = target
        item.name = restored  # rebind-ok: the output item is the object being restored  # pyright: ignore[reportAttributeAccessIssue]  # the pydantic union has no shared declaration
        item.namespace = namespace  # rebind-ok: ditto, xAI hands the pair back on the way out  # pyright: ignore[reportAttributeAccessIssue]  # ditto
        return item

    def _restore_chunk_tool_calls(
        self, chunk: Mapping[str, object]
    ) -> dict[str, object]:  # mutable-ok: streaming chunk dict handed back to the caller
        if not self._inlined_tool_names:
            return dict(chunk)

        patched: Final = dict(chunk)
        if patched.get("item") is not None:
            patched["item"] = self._restore_any_item(patched["item"])

        response: Final = xai_as_mapping(patched.get("response"))
        output: Final = response.get("output") if response is not None else None
        if response is not None and _is_tool_group(output):
            patched["response"] = {
                **response,
                "output": [self._restore_any_item(entry) for entry in _as_tool_sequence(output)],
            }
        return patched

    def _oauth_token(self, model: str, litellm_params: GenericLiteLLMParams) -> str | None:
        """OAuth bearer token for deployments configured with use_xai_oauth."""
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        if not should_use_xai_oauth(litellm_params.model_dump()):
            return None
        try:
            return XAIOAuthAuthenticator(auth_file=litellm_params.xai_oauth_token_file).get_access_token()
        except XAIOAuthError as exc:
            raise AuthenticationError(
                model=model,
                llm_provider=self.custom_llm_provider.value,
                message=str(exc),
            ) from exc

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        """
        Validate environment and set up headers for XAI API.

        Uses the shared xAI key resolver with Responses API legacy precedence.
        """
        params: Final = litellm_params or GenericLiteLLMParams()
        api_key: Final = XAIModelInfo.get_api_key(params.api_key, legacy_generic_before_env=True) or self._oauth_token(
            model=model,
            litellm_params=params,
        )

        if not api_key:
            raise ValueError(
                "XAI API key is required. Set api_key, litellm.xai_key, "
                "litellm.api_key, XAI_API_KEY, or use_xai_oauth=True."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for XAI Responses API endpoint.

        Returns:
            str: The full URL for the XAI /responses endpoint
        """
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        api_key: Final = XAIModelInfo.get_api_key(litellm_params.get("api_key"), legacy_generic_before_env=True)
        if should_use_xai_oauth(litellm_params) and not api_key:
            token_file: Final = xai_oauth_token_file(litellm_params)  # pyright: ignore[reportUnknownArgumentType]  # litellm_params arrives untyped from the provider config
            api_base = XAIOAuthAuthenticator(auth_file=token_file).get_api_base()
        else:
            api_base = api_base or litellm.api_base or get_secret_str("XAI_API_BASE") or XAI_API_BASE

        return join_api_base(api_base, "responses")

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        response: Final = super().transform_response_api_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        restated_usage: Final = _usage_restated_from_xai_ticks(response.usage)
        if restated_usage is not None:
            response.usage = restated_usage
        if self._inlined_tool_names:
            restored: Final = [self._restore_any_item(item) for item in (response.output or ())]  # pyright: ignore[reportUnknownArgumentType]  # output items mix pydantic models with a bare dict member
            response.output = restored  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]  # output is a union of list types; restored items keep their runtime type
        return response

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        event: Final = super().transform_streaming_response(
            model=model,
            parsed_chunk=self._restore_chunk_tool_calls(parsed_chunk),
            logging_obj=logging_obj,
        )
        if not isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent)):
            return event
        restated_usage: Final = _usage_restated_from_xai_ticks(event.response.usage)
        if restated_usage is not None:
            event.response.usage = restated_usage
        return event

    def supports_native_websocket(self) -> bool:
        """XAI does not support native WebSocket for Responses API"""
        return False

    def reasoning_field_values(self) -> ReasoningFieldValues:
        return STANDARD_MODE_WITHOUT_CONTEXT_REASONING_FIELD_VALUES

    def ignores_unsupported_reasoning_fields(self) -> bool:
        return True
