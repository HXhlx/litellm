import base64
from collections.abc import Mapping
from io import BufferedIOBase
from typing import TYPE_CHECKING, Final, TypeAlias

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.xai.common_utils import (
    XAIModelInfo,
    xai_api_root,
    xai_authorization,
    xai_json_headers,
    xai_optional_str,
)
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

_ImageInput: TypeAlias = FileTypes | Mapping[str, object]

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

_XAI_NATIVE_PARAMS: Final = frozenset({"aspect_ratio", "n", "resolution"})
_PASSTHROUGH_PARAMS: Final = frozenset({"aspect_ratio", "resolution"})
_DROPPED_PARAMS: Final = frozenset({"size", "response_format", "user"})
_INT: Final = TypeAdapter(int)


class _ImageDatum(BaseModel, frozen=True):
    url: str | None = None
    b64_json: str | None = None
    b64: str | None = None


class _ImagePayload(BaseModel, frozen=True):
    data: tuple[_ImageDatum, ...] = ()


_PAYLOAD: Final = TypeAdapter(_ImagePayload)


def _as_image_list(image: _ImageInput | list[_ImageInput]) -> list[_ImageInput]:  # mutable-ok: JSON payload list
    if isinstance(image, list):
        return list(image)
    return [image]


def _rewind_and_read(stream: BufferedIOBase) -> bytes:
    position: Final = stream.tell()
    stream.seek(0)
    data: Final = stream.read()
    stream.seek(position)
    return data


def _read_all_bytes(image: FileTypes) -> bytes:
    if isinstance(image, bytes):
        return image
    if isinstance(image, bytearray):
        return bytes(image)
    if isinstance(image, BufferedIOBase):
        return _rewind_and_read(image)
    raise ValueError(f"Unsupported image input type for xAI image edit: {type(image)}")


def _data_url(image: FileTypes) -> str:
    mime: Final = ImageEditRequestUtils.get_image_content_type(image)
    encoded: Final = base64.b64encode(_read_all_bytes(image)).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _to_image_url(image: _ImageInput) -> Mapping[str, str]:
    if isinstance(image, str):
        return {"url": image}
    if isinstance(image, Mapping):
        target: Final = image.get("url") or image.get("file_id")
        if not target:
            raise ValueError("xAI image edit reference needs a 'url' or a 'file_id'.")
        if "file_id" in image and "url" not in image:
            return {"file_id": str(target)}
        return {"url": str(target)}
    return {"url": _data_url(image)}


class XAIImageEditConfig(BaseImageEditConfig):
    """xAI Imagine image edit (SuperGrok OAuth or API key). JSON POST /v1/images/edits."""

    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: BaseImageEditConfig contract returns list
        return ["n", "response_format", "size", "user"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseImageEditConfig contract
        supported: Final = frozenset(self.get_supported_openai_params(model))
        accepted: Final = {
            key: value
            for key, value in image_edit_optional_params.items()
            if key in supported or key in _XAI_NATIVE_PARAMS
        }
        unsupported: Final = tuple(
            key for key in image_edit_optional_params if key not in supported and key not in _XAI_NATIVE_PARAMS
        )
        if unsupported and not drop_params:
            raise ValueError(
                f"Parameter {unsupported[0]} is not supported for model {model}. "
                f"Supported parameters are {sorted(supported | _XAI_NATIVE_PARAMS)}. "
                "Set drop_params=True to drop unsupported parameters."
            )

        size: Final = accepted.get("size")
        ratio: Final = accepted.get("aspect_ratio") or (_SIZE_TO_ASPECT_RATIO.get(str(size), "1:1") if size else None)
        count: Final = accepted.get("n")
        with_ratio: Final = (
            {**accepted, "aspect_ratio": ratio} if ratio is not None and "aspect_ratio" not in accepted else accepted
        )
        coerced: Final = {**with_ratio, "n": _INT.validate_python(count)} if count is not None else with_ratio
        return {key: value for key, value in coerced.items() if key not in _DROPPED_PARAMS}

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        base: Final = xai_api_root(
            api_base,
            xai_optional_str(litellm_params.get("api_key")),
            litellm_params,
        ).rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/images/edits"
        return f"{base}/v1/images/edits"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseImageEditConfig contract
        return xai_json_headers(
            headers,
            xai_authorization(model, api_key, litellm_params, "image edit"),
        )

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: _ImageInput | list[_ImageInput] | None,  # mutable-ok: callers pass a literal list of reference images
        image_edit_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles]:  # mutable-ok: BaseImageEditConfig contract
        if image is None:
            raise ValueError("xAI image edit requires at least one reference image.")

        image_payloads: Final = [_to_image_url(item) for item in _as_image_list(image)]
        if not image_payloads:
            raise ValueError("xAI image edit requires at least one reference image.")

        extras: Final = {
            key: value
            for key, value in image_edit_optional_request_params.items()
            if key in _PASSTHROUGH_PARAMS and value is not None
        }
        count: Final = image_edit_optional_request_params.get("n")
        request: Final = {
            "model": XAIModelInfo.get_base_model(model) or model,
            **({"prompt": prompt} if prompt is not None else {}),
            **({"image": image_payloads[0]} if len(image_payloads) == 1 else {"images": image_payloads}),
            **extras,
            **({"n": _INT.validate_python(count)} if count is not None else {}),
        }
        return request, []

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> ImageResponse:
        try:
            payload: Final = _PAYLOAD.validate_python(raw_response.json())
        except (ValueError, ValidationError):
            raise self.get_error_class(  # pyright: ignore[reportUnknownMemberType]  # base error helper takes an untyped headers dict
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            ) from None

        images: Final = tuple(
            ImageObject(url=datum.url, b64_json=datum.b64_json or datum.b64) for datum in payload.data
        )
        if not images:
            raise self.get_error_class(  # pyright: ignore[reportUnknownMemberType]  # base error helper takes an untyped headers dict
                error_message=f"xAI image edit returned no image data: {payload}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response: Final = ImageResponse()
        model_response.data = list(images)
        return model_response
