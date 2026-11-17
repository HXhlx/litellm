import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.xai.common_utils import (
    XAIModelInfo,
    xai_api_root,
    xai_authorization,
    xai_json_headers,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.tokenizer import Encoding

_SIZE_TO_ASPECT_RATIO: Final[Mapping[str, str]] = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1920x1080": "16:9",
    "1080x1920": "9:16",
}

_XAI_NATIVE_PARAMS: Final = frozenset({"aspect_ratio", "n"})
_DROPPED_PARAMS: Final = frozenset({"size", "response_format", "user"})
_INT: Final = TypeAdapter(int)


class _ImageDatum(BaseModel, frozen=True):
    url: str | None = None
    b64_json: str | None = None
    b64: str | None = None


class _ImagePayload(BaseModel, frozen=True):
    data: tuple[_ImageDatum, ...] = ()


_PAYLOAD: Final = TypeAdapter(_ImagePayload)


class XAIImageGenerationConfig(BaseImageGenerationConfig):
    """xAI Imagine image generation (SuperGrok OAuth or API key)."""

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: BaseImageGenerationConfig contract returns list
        return ["n", "response_format", "size", "user"]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseImageGenerationConfig contract
        supported: Final = frozenset(self.get_supported_openai_params(model))
        accepted: Final = {
            key: value
            for key, value in non_default_params.items()
            if key not in optional_params and (key in supported or key in _XAI_NATIVE_PARAMS)
        }
        unsupported: Final = tuple(
            key
            for key in non_default_params
            if key not in optional_params and key not in supported and key not in _XAI_NATIVE_PARAMS
        )
        if unsupported and not drop_params:
            raise ValueError(
                f"Parameter {unsupported[0]} is not supported for model {model}. "
                f"Supported parameters are {sorted(supported | _XAI_NATIVE_PARAMS)}. "
                "Set drop_params=True to drop unsupported parameters."
            )

        merged: Final = {**optional_params, **accepted}
        size: Final = merged.get("size")
        ratio: Final = merged.get("aspect_ratio") or (_SIZE_TO_ASPECT_RATIO.get(str(size), "1:1") if size else None)
        count: Final = merged.get("n")
        with_ratio: Final = (
            {**merged, "aspect_ratio": ratio} if ratio is not None and "aspect_ratio" not in merged else merged
        )
        coerced: Final = {**with_ratio, "n": _INT.validate_python(count)} if count is not None else with_ratio
        return {key: value for key, value in coerced.items() if key not in _DROPPED_PARAMS}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        base: Final = xai_api_root(api_base, api_key, litellm_params).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/images/generations"
        return f"{base}/v1/images/generations"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: BaseImageGenerationConfig contract
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseImageGenerationConfig contract
        return xai_json_headers(
            headers,
            xai_authorization(model, api_key, litellm_params, "image generation"),
        )

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: BaseImageGenerationConfig contract
        request: Final = {
            "model": XAIModelInfo.get_base_model(model) or model,
            "prompt": prompt,
        }
        ratio: Final = optional_params.get("aspect_ratio")
        count: Final = optional_params.get("n")
        return {
            **request,
            **({"aspect_ratio": ratio} if ratio is not None else {}),
            **({"n": _INT.validate_python(count)} if count is not None else {}),
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "Logging",
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            payload: Final = _PAYLOAD.validate_python(raw_response.json())
        except (json.JSONDecodeError, ValidationError):
            raise self.get_error_class(  # pyright: ignore[reportUnknownMemberType]  # base error helper takes an untyped headers dict
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            ) from None

        logging_obj.post_call(  # pyright: ignore[reportUnknownMemberType]  # Logging.post_call parameters are untyped
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=payload,
        )

        images: Final = tuple(
            ImageObject(url=datum.url, b64_json=datum.b64_json or datum.b64) for datum in payload.data
        )
        if not images:
            raise self.get_error_class(  # pyright: ignore[reportUnknownMemberType]  # base error helper takes an untyped headers dict
                error_message=f"xAI image generation returned no image data: {payload}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response.data = list(images)  # rebind-ok: the base contract fills the caller's response object
        return model_response
