from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter

from litellm.protocol_routing._types import SupportedProtocol

PROVIDER_DEFAULT_PROTOCOLS: Final[Mapping[str, Sequence[SupportedProtocol]]] = {
    "openai": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.OPENAI_RESPONSES],
    "azure": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.OPENAI_RESPONSES],
    "azure_ai": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.OPENAI_RESPONSES],
    "anthropic": [SupportedProtocol.ANTHROPIC_MESSAGES],
    "bedrock": [SupportedProtocol.ANTHROPIC_MESSAGES, SupportedProtocol.OPENAI_CHAT],
    "vertex_ai": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.GOOGLE_GENERATE_CONTENT],
    "gemini": [SupportedProtocol.GOOGLE_GENERATE_CONTENT, SupportedProtocol.OPENAI_CHAT],
    "deepseek": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.ANTHROPIC_MESSAGES],
    "minimax": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.ANTHROPIC_MESSAGES],
    "cohere": [SupportedProtocol.OPENAI_CHAT],
    "mistral": [SupportedProtocol.OPENAI_CHAT],
    "ollama": [SupportedProtocol.OPENAI_CHAT],
    "vllm": [SupportedProtocol.OPENAI_CHAT],
    "together_ai": [SupportedProtocol.OPENAI_CHAT],
    "groq": [SupportedProtocol.OPENAI_CHAT],
    "xai": [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.OPENAI_RESPONSES],
    "nvidia_nim": [SupportedProtocol.OPENAI_CHAT],
    "modelscope": [SupportedProtocol.OPENAI_CHAT],
    "volcengine": [SupportedProtocol.OPENAI_CHAT],
}

_SUPPORTED_PROTOCOLS: Final = TypeAdapter(list[SupportedProtocol])


def infer_protocols(
    provider: str | None,
    model_info: Mapping[str, object] | None = None,
) -> list[SupportedProtocol]:  # mutable-ok: public helper hands back a list callers own and may sort
    """Infer supported protocols for a deployment.

    Priority:
    1. Explicit model_info.supported_protocols (if provided)
    2. Provider default from PROVIDER_DEFAULT_PROTOCOLS
    3. Fallback to OPENAI_CHAT

    Args:
        provider: The LLM provider name (e.g., "openai", "anthropic")
        model_info: Optional model metadata dict, may contain "supported_protocols"

    Returns:
        List of SupportedProtocol values (always a new list to prevent mutation)
    """
    if model_info is not None:
        explicit: Final = model_info.get("supported_protocols")
        if explicit is not None:
            return _SUPPORTED_PROTOCOLS.validate_python(explicit)

    if provider is None:
        return [SupportedProtocol.OPENAI_CHAT]

    defaults: Final = PROVIDER_DEFAULT_PROTOCOLS.get(provider)
    if defaults is not None:
        return list(defaults)

    return [SupportedProtocol.OPENAI_CHAT]
