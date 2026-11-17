"""
Protocol-level routing for LiteLLM.

Provides strict mode to enforce protocol compatibility without automatic conversion.
"""

from litellm.protocol_routing._filter import (
    check_strict_protocol_for_provider,
    filter_deployments_by_protocol,
)
from litellm.protocol_routing._mapping import (
    PROVIDER_DEFAULT_PROTOCOLS,
    infer_protocols,
)
from litellm.protocol_routing._types import (
    ProtocolMismatchError,
    ProtocolRoutingMode,
    SupportedProtocol,
    get_protocol_routing_mode,
    set_protocol_routing_mode,
)

__all__ = [
    "PROVIDER_DEFAULT_PROTOCOLS",
    "ProtocolMismatchError",
    "ProtocolRoutingMode",
    "SupportedProtocol",
    "check_strict_protocol_for_provider",
    "filter_deployments_by_protocol",
    "get_protocol_routing_mode",
    "infer_protocols",
    "set_protocol_routing_mode",
]
