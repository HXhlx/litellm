from collections.abc import Iterable, Mapping
from typing import Final

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ProviderSpecificModelInfo

USD_TICKS_PER_DOLLAR: Final = 10_000_000_000
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_VALUES: Final = TypeAdapter(Iterable[object], config=ConfigDict(hide_input_in_errors=True))
_MODEL_ID: Final = TypeAdapter(str, config=ConfigDict(strict=True, hide_input_in_errors=True))


def xai_reported_cost_in_usd(cost_in_usd_ticks: object) -> float | None:
    """xAI bills in ticks of a dollar: https://docs.x.ai/developers/cost-tracking"""
    if not isinstance(cost_in_usd_ticks, int) or isinstance(cost_in_usd_ticks, bool):
        return None
    if cost_in_usd_ticks < 0:
        return None
    return cost_in_usd_ticks / USD_TICKS_PER_DOLLAR


_TOOL_FIELDS_WITHOUT_TOOLS: Final = frozenset({"tool_choice", "parallel_tool_calls", "tools"})


def omit_tool_choice_without_tools(
    params: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: provider request params
    """Drop tool fields when the request carries no tools, since xAI rejects a tool_choice without tools."""
    if params.get("tools"):
        return dict(params)
    return {key: value for key, value in params.items() if key not in _TOOL_FIELDS_WITHOUT_TOOLS}


def xai_as_mapping(value: object) -> Mapping[str, object] | None:
    """Read a provider payload as a string-keyed mapping, or None when it is not one."""
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def xai_optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def xai_oauth_token_file(litellm_params: Mapping[str, object] | None) -> str | None:
    return xai_optional_str((litellm_params or {}).get("xai_oauth_token_file"))


def xai_uses_oauth(litellm_params: Mapping[str, object] | None, api_key: str | None) -> bool:
    from litellm.llms.xai.oauth import should_use_xai_oauth

    return should_use_xai_oauth(litellm_params) and not XAIModelInfo.get_api_key(api_key)


def xai_api_root(api_base: str | None, api_key: str | None, litellm_params: Mapping[str, object] | None) -> str:
    """OAuth deployments resolve their own base, API-key deployments use the configured one."""
    from litellm.llms.xai.oauth import XAIOAuthAuthenticator

    if xai_uses_oauth(litellm_params, api_key):
        return XAIOAuthAuthenticator(auth_file=xai_oauth_token_file(litellm_params)).get_api_base()
    return api_base or get_secret_str("XAI_API_BASE") or get_secret_str("XAI_OAUTH_API_BASE") or XAI_API_BASE


def xai_authorization(
    model: str,
    api_key: str | None,
    litellm_params: Mapping[str, object] | None,
    purpose: str,
) -> str:
    """Bearer value for xAI, from the OAuth token when configured and the API key otherwise."""
    from litellm.llms.xai.oauth import XAIOAuthAuthenticator, XAIOAuthError

    dynamic_api_key: Final = XAIModelInfo.get_api_key(api_key)
    if not xai_uses_oauth(litellm_params, api_key):
        if not dynamic_api_key:
            raise AuthenticationError(
                model=model,
                llm_provider="xai",
                message=(
                    f"Missing xAI credentials for {purpose}. Pass api_key / XAI_API_KEY, or set use_xai_oauth=True."
                ),
            )
        return f"Bearer {dynamic_api_key}"

    try:
        return f"Bearer {XAIOAuthAuthenticator(auth_file=xai_oauth_token_file(litellm_params)).get_access_token()}"
    except XAIOAuthError as exc:
        raise AuthenticationError(model=model, llm_provider="xai", message=str(exc)) from exc


def xai_json_headers(
    headers: Mapping[str, str], authorization: str
) -> dict[str, str]:  # mutable-ok: HTTP header dict the request layer consumes
    """Base headers plus the bearer token, defaulting the JSON content type."""
    content_type: Final = (
        {} if "content-type" in headers or "Content-Type" in headers else {"Content-Type": "application/json"}
    )
    return {**headers, "Authorization": authorization, **content_type}


class XAIModelInfo(BaseLLMModelInfo):
    def get_provider_info(
        self,
        model: str,
    ) -> ProviderSpecificModelInfo | None:
        """
        Default values all models of this provider support.
        """
        return {
            "supports_web_search": True,
        }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"

        # Ensure Content-Type is set to application/json
        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"

        return headers

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("XAI_API_BASE") or "https://api.x.ai"

    @staticmethod
    def get_api_key(
        api_key: str | None = None,
        legacy_generic_before_env: bool = False,
    ) -> str | None:
        """
        Resolve xAI API keys while preserving endpoint-specific legacy order.

        Chat uses xai_key before XAI_API_KEY without adding a generic
        litellm.api_key fallback. Responses and realtime historically
        preferred litellm.api_key over XAI_API_KEY, so those paths opt into
        the legacy order with legacy_generic_before_env=True. In both modes,
        the provider-specific litellm.xai_key takes precedence over fallbacks.
        """
        if legacy_generic_before_env:
            return api_key or litellm.xai_key or litellm.api_key or get_secret_str("XAI_API_KEY")

        return api_key or litellm.xai_key or get_secret_str("XAI_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model.replace("xai/", "")

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        api_base = self.get_api_base(api_base)
        api_key = self.get_api_key(api_key)
        if api_base is None or api_key is None:
            raise ValueError(
                "XAI API base or key is not set. Set XAI_API_BASE and provide an xAI API key via api_key, litellm.xai_key, or XAI_API_KEY."
            )
        response: Final = litellm.module_level_client.get(
            url=join_api_base(api_base, "v1/models"),
            headers={"Authorization": f"Bearer {api_key}"},
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise Exception(
                f"Failed to fetch models from XAI. Status code: {response.status_code}, Response: {response.text}"
            )

        models: Final = _JSON_VALUES.validate_python(_JSON_OBJECT.validate_python(response.json())["data"])

        litellm_model_names: Final = []
        for model in models:
            stripped_model_name = _MODEL_ID.validate_python(_JSON_OBJECT.validate_python(model)["id"])
            litellm_model_name = "xai/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names
