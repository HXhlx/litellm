from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.xai.common_utils import XAIModelInfo, omit_tool_choice_without_tools
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.llms.xai import XAIWebSearchTool, XAIXSearchTool
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_NAMESPACE_TOOL_TYPE: Final = "namespace"
_NAMESPACE_SEPARATOR: Final = "__"

# qualified name -> (namespace, tool name as the client declared it)
InlinedToolNames = Mapping[str, tuple[str, str]]


def _qualified(namespace: str, name: str) -> str:
    return f"{namespace}{_NAMESPACE_SEPARATOR}{name}"


def _is_named_member(member: Any) -> bool:
    return isinstance(member, dict) and isinstance(member.get("name"), str)


def _rename_member(member: Any, namespace: str) -> tuple[Any, tuple[str, tuple[str, str]] | None]:
    if not _is_named_member(member):
        return member, None
    qualified: Final = _qualified(namespace, member["name"])
    return {**member, "name": qualified}, (qualified, (namespace, member["name"]))


def _expand_namespace_tool(tool: Any) -> tuple[list[Any], list[tuple[str, tuple[str, str]]]]:
    """One inbound tool -> (tools to send, qualified-name back-references).

    A `namespace` group contributes its members under `<namespace>__<tool>`, the
    qualified name Codex itself uses for MCP tools when they are not namespaced,
    which keeps names unique across groups. Anything else passes through.
    """
    if not isinstance(tool, dict) or tool.get("type") != _NAMESPACE_TOOL_TYPE:
        return [tool], []

    namespace: Final = tool.get("name")
    members: Final = tool.get("tools")
    if not isinstance(namespace, str) or not namespace or not isinstance(members, list):
        return [tool], []

    renamed: Final = [_rename_member(member, namespace) for member in members]
    return (
        [out for out, _ in renamed],
        [ref for _, ref in renamed if ref is not None],
    )


def _inline_namespace_tools(tools: list[Any]) -> tuple[list[Any], InlinedToolNames]:
    """Replace `namespace` tool groups with their member tools.

    xAI rejects `namespace` as a tool type, and Codex wraps every MCP server's
    tools plus its sub-agent tools in one, so a single group fails the whole
    request. The returned mapping lets the response side hand Codex back the
    `(namespace, name)` pair it declared the tool under.
    """
    expanded: Final = [_expand_namespace_tool(tool) for tool in tools]
    return (
        [member for members, _ in expanded for member in members],
        MappingProxyType({name: pair for _, pairs in expanded for name, pair in pairs}),
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

    def _transform_web_search_tool(self, tool: dict[str, Any]) -> XAIWebSearchTool | dict[str, Any]:
        """
        Transform web_search tool to XAI format.

        XAI supports web_search with specific filters:
        - allowed_domains (max 5)
        - excluded_domains (max 5)
        - enable_image_understanding

        XAI does NOT support search_context_size (OpenAI-specific).
        """
        xai_tool: Final[dict[str, Any]] = {"type": "web_search"}

        # Remove search_context_size if present (not supported by XAI)
        if "search_context_size" in tool:
            verbose_logger.info(
                "XAI does not support 'search_context_size' parameter. Removing it from web_search tool."
            )

        # Handle filters (XAI-specific structure)
        filters: Final = {}
        if "allowed_domains" in tool:
            allowed_domains: Final = tool["allowed_domains"]
            filters["allowed_domains"] = allowed_domains

        if "excluded_domains" in tool:
            excluded_domains: Final = tool["excluded_domains"]
            filters["excluded_domains"] = excluded_domains

        # Add filters if any were specified
        if filters:
            xai_tool["filters"] = filters

        # Handle enable_image_understanding (top-level in XAI format)
        if "enable_image_understanding" in tool:
            xai_tool["enable_image_understanding"] = tool["enable_image_understanding"]

        return xai_tool

    def _transform_x_search_tool(self, tool: dict[str, Any]) -> XAIXSearchTool | dict[str, Any]:
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
        xai_tool: Final[dict[str, Any]] = {"type": "x_search"}

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

    def _transform_tool(self, tool: Any) -> Any:
        if not isinstance(tool, dict):
            return tool

        tool_type: Final = tool.get("type")
        if tool_type == "code_interpreter":
            # XAI supports code_interpreter but doesn't use the container field
            return {"type": "code_interpreter"}
        if tool_type == "web_search":
            return self._transform_web_search_tool(tool)
        if tool_type == "x_search":
            return self._transform_x_search_tool(tool)
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

        raw_tools: Final = params.get("tools")
        if raw_tools:
            tools_list: Final = raw_tools if isinstance(raw_tools, list) else [raw_tools]
            inlined, inlined_names = _inline_namespace_tools(tools_list)
            self._inlined_tool_names = inlined_names
            params["tools"] = [self._transform_tool(tool) for tool in inlined]

        return omit_tool_choice_without_tools(params)

    def _restore_output_item(self, item: Any) -> Any:
        """Give a tool call back the `(namespace, name)` pair the client declared.

        Codex registers a namespaced tool under that pair and its router rejects
        the qualified name we had to send xAI, so the split has to be undone on
        the way back. Items are pydantic models on the non-streaming path and
        plain dicts inside streaming chunks.
        """
        target: Final = self._inlined_tool_names.get(
            item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
        )
        if target is None:
            return item

        namespace, name = target
        if isinstance(item, dict):
            return {**item, "name": name, "namespace": namespace}
        item.name = name
        item.namespace = namespace
        return item

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
        if self._inlined_tool_names:
            response.output = [self._restore_output_item(item) for item in (response.output or [])]
        return response

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        return super().transform_streaming_response(
            model=model,
            parsed_chunk=self._restore_chunk_tool_calls(parsed_chunk),
            logging_obj=logging_obj,
        )

    def _restore_chunk_tool_calls(self, chunk: dict) -> dict:
        if not self._inlined_tool_names or not isinstance(chunk, dict):
            return chunk

        patched: Final = dict(chunk)
        if patched.get("item") is not None:
            patched["item"] = self._restore_output_item(patched["item"])

        response = patched.get("response")
        if isinstance(response, dict) and isinstance(response.get("output"), list):
            patched["response"] = {
                **response,
                "output": [self._restore_output_item(item) for item in response["output"]],
            }
        return patched

    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        """
        Validate environment and set up headers for XAI API.

        Uses the shared xAI key resolver with Responses API legacy precedence.
        """
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = XAIModelInfo.get_api_key(litellm_params.api_key, legacy_generic_before_env=True)

        if not api_key:
            from litellm.llms.xai.oauth import (
                XAIOAuthAuthenticator,
                XAIOAuthError,
                should_use_xai_oauth,
            )

            if should_use_xai_oauth(litellm_params.model_dump()):
                _token_file = litellm_params.xai_oauth_token_file
                try:
                    api_key = XAIOAuthAuthenticator(auth_file=_token_file).get_access_token()
                except XAIOAuthError as exc:
                    raise AuthenticationError(
                        model=model,
                        llm_provider=self.custom_llm_provider.value,
                        message=str(exc),
                    ) from exc

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
            _token_file = litellm_params.get("xai_oauth_token_file")
            api_base = XAIOAuthAuthenticator(auth_file=_token_file).get_api_base()
        else:
            api_base = api_base or litellm.api_base or get_secret_str("XAI_API_BASE") or XAI_API_BASE

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/responses"

    def supports_native_websocket(self) -> bool:
        """XAI does not support native WebSocket for Responses API"""
        return False
