"""
Tests for XAI Responses API transformation

Tests the XAIResponsesAPIConfig class that handles XAI-specific
transformations for the Responses API.

Source: litellm/llms/xai/responses/transformation.py
"""

from unittest.mock import MagicMock


import pytest

import litellm
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.responses.utils import ResponseAPILoggingUtils, ResponsesAPIRequestUtils
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import LlmProviders, Usage
from litellm.utils import ProviderConfigManager


class TestXAIResponsesAPITransformation:
    """Test XAI Responses API configuration and transformations"""

    def test_xai_provider_config_registration(self):
        """Test that XAI provider returns XAIResponsesAPIConfig"""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="xai/grok-4-fast",
            provider=LlmProviders.XAI,
        )

        assert config is not None, "Config should not be None for XAI provider"
        assert isinstance(config, XAIResponsesAPIConfig), f"Expected XAIResponsesAPIConfig, got {type(config)}"
        assert config.custom_llm_provider == LlmProviders.XAI, "custom_llm_provider should be XAI"

    def test_code_interpreter_container_field_removed(self):
        """Test that container field is removed from code_interpreter tools"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(tools=[{"type": "code_interpreter", "container": {"type": "auto"}}])

        result = config.map_openai_params(response_api_optional_params=params, model="grok-4-fast", drop_params=False)

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "code_interpreter"
        assert "container" not in result["tools"][0], "Container field should be removed"

    def test_instructions_parameter_preserved(self):
        """Codex carries its system prompt in `instructions`; xAI accepts the field."""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(instructions="You are a helpful assistant.", temperature=0.7)

        result = config.map_openai_params(response_api_optional_params=params, model="grok-4-fast", drop_params=False)

        assert result["instructions"] == "You are a helpful assistant.", "Instructions must reach xAI"
        assert result.get("temperature") == 0.7, "Other params should be preserved"

    def test_supported_params_includes_instructions(self):
        """Test that get_supported_openai_params includes instructions"""
        config = XAIResponsesAPIConfig()
        supported = config.get_supported_openai_params("grok-4-fast")

        assert "instructions" in supported, "instructions should be supported"
        assert "tools" in supported, "tools should be supported"
        assert "temperature" in supported, "temperature should be supported"
        assert "model" in supported, "model should be supported"

    def test_instructions_survives_the_optional_param_gate(self):
        """With drop_params off, an undeclared `instructions` would 400 before egress."""
        config = XAIResponsesAPIConfig()
        params = ResponsesAPIOptionalRequestParams(instructions="You are Codex.")

        mapped = ResponsesAPIRequestUtils.get_optional_params_responses_api(
            model="grok-4.7",
            responses_api_provider_config=config,
            response_api_optional_params=params,
            drop_params=False,
        )

        assert mapped["instructions"] == "You are Codex."

    def test_namespace_tool_group_inlined_as_qualified_functions(self):
        """Codex wraps each MCP server's tools in a `namespace` group, a tool type
        xAI rejects, so one group fails the whole request. Members must go out as
        functions named `<namespace>__<tool>`."""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {"type": "function", "name": "exec_command", "parameters": {}},
                {
                    "type": "namespace",
                    "name": "mcp__gbrain",
                    "description": "gbrain tools",
                    "tools": [
                        {"type": "function", "name": "search", "parameters": {}},
                        {"type": "function", "name": "fetch", "parameters": {}},
                    ],
                },
                {"type": "namespace", "name": "multi_agent_v1", "tools": [{"type": "function", "name": "spawn_agent"}]},
            ]
        )

        result = config.map_openai_params(response_api_optional_params=params, model="grok-4.7", drop_params=True)

        assert [tool["type"] for tool in result["tools"]] == ["function"] * 4, "no namespace type may survive"
        assert [tool["name"] for tool in result["tools"]] == [
            "exec_command",
            "mcp__gbrain__search",
            "mcp__gbrain__fetch",
            "multi_agent_v1__spawn_agent",
        ]

    def test_tool_choice_dropped_when_compaction_sends_no_tools(self):
        """Codex compaction sends tool_choice=auto with tools omitted or empty. xAI 400s."""
        config = XAIResponsesAPIConfig()

        missing = config.map_openai_params(
            response_api_optional_params=ResponsesAPIOptionalRequestParams(
                tool_choice="auto",
                parallel_tool_calls=True,
                instructions="compact",
            ),
            model="grok-4.7",
            drop_params=True,
        )
        empty = config.map_openai_params(
            response_api_optional_params=ResponsesAPIOptionalRequestParams(
                tool_choice="auto",
                parallel_tool_calls=True,
                tools=[],
            ),
            model="grok-4.7",
            drop_params=True,
        )
        kept = config.map_openai_params(
            response_api_optional_params=ResponsesAPIOptionalRequestParams(
                tool_choice="auto",
                parallel_tool_calls=True,
                tools=[{"type": "function", "name": "exec_command", "parameters": {}}],
            ),
            model="grok-4.7",
            drop_params=True,
        )

        assert "tool_choice" not in missing
        assert "parallel_tool_calls" not in missing
        assert "tools" not in missing
        assert missing["instructions"] == "compact"
        assert "tool_choice" not in empty
        assert "tools" not in empty
        assert kept["tool_choice"] == "auto"
        assert kept["parallel_tool_calls"] is True
        assert kept["tools"][0]["name"] == "exec_command"

    def test_xai_responses_endpoint_url(self):
        """Test that get_complete_url returns correct XAI endpoint"""
        config = XAIResponsesAPIConfig()

        # Test with default XAI API base
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://api.x.ai/v1/responses", f"Expected XAI responses endpoint, got {url}"

        # Test with custom api_base
        custom_url = config.get_complete_url(api_base="https://custom.x.ai/v1", litellm_params={})
        assert custom_url == "https://custom.x.ai/v1/responses", f"Expected custom endpoint, got {custom_url}"

        # Test with trailing slash
        url_with_slash = config.get_complete_url(api_base="https://api.x.ai/v1/", litellm_params={})
        assert url_with_slash == "https://api.x.ai/v1/responses", "Should handle trailing slash"

    def test_web_search_tool_transformation(self):
        """Test that web_search tools are transformed to XAI format"""
        config = XAIResponsesAPIConfig()

        # Test with allowed_domains
        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "web_search",
                    "allowed_domains": ["wikipedia.org", "x.ai"],
                    "enable_image_understanding": True,
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "web_search"
        assert "filters" in tool
        assert tool["filters"]["allowed_domains"] == ["wikipedia.org", "x.ai"]
        assert tool["enable_image_understanding"] is True

    def test_web_search_search_context_size_removed(self):
        """Test that search_context_size is removed from web_search tools"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "high",  # Not supported by XAI
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "web_search"
        assert "search_context_size" not in tool

    def test_web_search_excluded_domains(self):
        """Test web_search with excluded_domains"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[{"type": "web_search", "excluded_domains": ["example.com", "test.com"]}]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        tool = result["tools"][0]
        assert "filters" in tool
        assert tool["filters"]["excluded_domains"] == ["example.com", "test.com"]

    def test_web_search_domains_limit(self):
        """Test that allowed_domains and excluded_domains are limited to 5"""
        config = XAIResponsesAPIConfig()

        # Test with more than 5 allowed_domains
        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "web_search",
                    "allowed_domains": [
                        "d1.com",
                        "d2.com",
                        "d3.com",
                        "d4.com",
                        "d5.com",
                        "d6.com",
                        "d7.com",
                    ],
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        tool = result["tools"][0]
        assert len(tool["filters"]["allowed_domains"]) == 7

    def test_x_search_tool_transformation(self):
        """Test that x_search tools are transformed correctly"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "x_search",
                    "allowed_x_handles": ["elonmusk", "xai"],
                    "from_date": "2025-01-01",
                    "to_date": "2025-01-28",
                    "enable_image_understanding": True,
                    "enable_video_understanding": True,
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "x_search"
        assert tool["allowed_x_handles"] == ["elonmusk", "xai"]
        assert tool["from_date"] == "2025-01-01"
        assert tool["to_date"] == "2025-01-28"
        assert tool["enable_image_understanding"] is True
        assert tool["enable_video_understanding"] is True

    def test_x_search_excluded_handles(self):
        """Test x_search with excluded_x_handles"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "x_search",
                    "excluded_x_handles": ["spam_account", "bot_account"],
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        tool = result["tools"][0]
        assert tool["excluded_x_handles"] == ["spam_account", "bot_account"]

    def test_mixed_tools(self):
        """Test transformation with multiple tool types"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {"type": "code_interpreter", "container": {"type": "auto"}},
                {"type": "web_search", "allowed_domains": ["wikipedia.org"]},
                {"type": "x_search", "allowed_x_handles": ["elonmusk"]},
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        assert len(result["tools"]) == 4

        # Verify code_interpreter
        assert result["tools"][0]["type"] == "code_interpreter"
        assert "container" not in result["tools"][0]

        # Verify web_search
        assert result["tools"][1]["type"] == "web_search"
        assert "filters" in result["tools"][1]

        # Verify x_search
        assert result["tools"][2]["type"] == "x_search"
        assert result["tools"][2]["allowed_x_handles"] == ["elonmusk"]

        # Verify function tool is unchanged
        assert result["tools"][3]["type"] == "function"
        assert result["tools"][3]["name"] == "get_weather"


class TestXAIResponsesWebSearchBilling:
    """Web search billing must not change the client-visible Responses usage schema."""

    _TOOL_DETAILS = {
        "web_search_calls": 2,
        "x_search_calls": 0,
        "code_interpreter_calls": 0,
        "file_search_calls": 0,
        "mcp_calls": 0,
        "document_search_calls": 0,
    }

    def _raw_response_json(self, include_web_search: bool) -> dict:
        web_search_output = (
            [{
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "grok"},
            }] if include_web_search else []
        )
        tool_usage = {"server_side_tool_usage_details": self._TOOL_DETAILS} if include_web_search else {}
        return {
            "id": "resp_1",
            "object": "response",
            "created_at": 1754900000,
            "model": "grok-4",
            "status": "completed",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "output": web_search_output
            + [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "grok says hi", "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
                **tool_usage,
            },
        }

    def _transform(self, include_web_search: bool) -> ResponsesAPIResponse:
        raw_response = MagicMock()
        raw_response.json.return_value = self._raw_response_json(include_web_search)
        raw_response.text = "raw"
        raw_response.headers = {}
        return XAIResponsesAPIConfig().transform_response_api_response(
            model="grok-4", raw_response=raw_response, logging_obj=MagicMock()
        )

    def test_response_usage_keeps_responses_api_schema(self):
        response = self._transform(include_web_search=True)

        assert isinstance(response.usage, ResponseAPIUsage)
        assert response.usage.input_tokens == 100
        assert response.usage.output_tokens == 20
        assert response.usage.model_extra["server_side_tool_usage_details"] == self._TOOL_DETAILS

    def test_bridged_usage_keeps_tool_details_for_billing(self):
        response = self._transform(include_web_search=True)

        bridged = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(response.usage)

        assert isinstance(bridged, Usage)
        assert bridged.prompt_tokens == 100
        assert bridged.completion_tokens == 20
        assert getattr(bridged, "server_side_tool_usage_details") == self._TOOL_DETAILS

    def test_completion_cost_bills_web_search_calls(self):
        with_search = litellm.completion_cost(
            completion_response=self._transform(include_web_search=True),
            model="xai/grok-4",
            custom_llm_provider="xai",
        )
        without_search = litellm.completion_cost(
            completion_response=self._transform(include_web_search=False),
            model="xai/grok-4",
            custom_llm_provider="xai",
        )

        assert with_search - without_search == pytest.approx(2 * 5.0 / 1000.0)

    def test_streaming_terminal_event_keeps_schema_and_details(self):
        parsed_chunk = {
            "type": "response.completed",
            "sequence_number": 7,
            "response": self._raw_response_json(include_web_search=True),
        }

        event = XAIResponsesAPIConfig().transform_streaming_response(
            model="grok-4", parsed_chunk=parsed_chunk, logging_obj=MagicMock()
        )

        assert isinstance(event, ResponseCompletedEvent)
        assert isinstance(event.response.usage, ResponseAPIUsage)
        assert event.response.usage.input_tokens == 100

        bridged = ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(event.response.usage)
        assert getattr(bridged, "server_side_tool_usage_details") == self._TOOL_DETAILS


class TestXAINamespaceToolRoundTrip:
    """xAI has no `namespace` tool type, so Codex tool groups are inlined on the
    way out and the (namespace, name) pair is restored on the way back."""

    _NAMESPACE = "mcp__sequential_thinking"
    _TOOL = "sequentialthinking"
    _QUALIFIED = "mcp__sequential_thinking__sequentialthinking"

    def _config_after_request(self, tools: list) -> XAIResponsesAPIConfig:
        config = XAIResponsesAPIConfig()
        config.map_openai_params(
            response_api_optional_params=ResponsesAPIOptionalRequestParams(tools=tools),
            model="grok-4.7",
            drop_params=True,
        )
        return config

    def _namespaced_config(self) -> XAIResponsesAPIConfig:
        return self._config_after_request(
            [
                {
                    "type": "namespace",
                    "name": self._NAMESPACE,
                    "tools": [{"type": "function", "name": self._TOOL}],
                }
            ]
        )

    def _raw_response(self, tool_name: str) -> MagicMock:
        raw = MagicMock()
        raw.json.return_value = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1791128515,
            "model": "grok-4.7",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": tool_name,
                    "arguments": "{}",
                    "namespace": None,
                    "status": "completed",
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }
        raw.text = "{}"
        raw.headers = {}
        raw.status_code = 200
        return raw

    def test_non_streaming_function_call_restores_namespace(self):
        config = self._namespaced_config()

        response = config.transform_response_api_response(
            model="grok-4.7",
            raw_response=self._raw_response(self._QUALIFIED),
            logging_obj=MagicMock(),
        )

        item = response.output[0]
        assert item.name == self._TOOL, "Codex registers the tool under its bare name"
        assert item.namespace == self._NAMESPACE, "without the namespace its router rejects the call"

    def test_streaming_output_item_restores_namespace(self):
        config = self._namespaced_config()

        event = config.transform_streaming_response(
            model="grok-4.7",
            parsed_chunk={
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": self._QUALIFIED,
                    "arguments": "",
                    "namespace": None,
                },
            },
            logging_obj=MagicMock(),
        )

        item = event.model_dump()["item"]
        assert item["name"] == self._TOOL
        assert item["namespace"] == self._NAMESPACE

    def test_streaming_terminal_event_restores_namespace(self):
        config = self._namespaced_config()

        event = config.transform_streaming_response(
            model="grok-4.7",
            parsed_chunk={
                "type": "response.completed",
                "sequence_number": 7,
                "response": self._raw_response(self._QUALIFIED).json.return_value,
            },
            logging_obj=MagicMock(),
        )

        item = event.response.output[0]
        assert item.name == self._TOOL
        assert item.namespace == self._NAMESPACE

    def test_client_supplied_qualified_name_is_not_split(self):
        """A client that already sends flat MCP names gets them back untouched."""
        config = self._config_after_request([{"type": "function", "name": self._QUALIFIED}])

        response = config.transform_response_api_response(
            model="grok-4.7",
            raw_response=self._raw_response(self._QUALIFIED),
            logging_obj=MagicMock(),
        )

        item = response.output[0]
        assert item.name == self._QUALIFIED
        assert item.namespace is None

    def test_unrelated_response_is_untouched_without_namespace_tools(self):
        config = XAIResponsesAPIConfig()

        response = config.transform_response_api_response(
            model="grok-4.7",
            raw_response=self._raw_response("exec_command"),
            logging_obj=MagicMock(),
        )

        assert response.output[0].name == "exec_command"
