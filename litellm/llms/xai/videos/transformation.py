import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # house cached-client factory has no public alias and its stub leaves params untyped
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # factory stub leaves params untyped
)
from litellm.llms.xai.common_utils import (
    XAIModelInfo,
    xai_api_root,
    xai_authorization,
    xai_json_headers,
    xai_optional_str,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

_SIZE_TO_ASPECT_RATIO: Final[Mapping[str, str]] = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1920x1080": "16:9",
    "1080x1920": "9:16",
}

_CREATE_PASSTHROUGH_KEYS: Final = frozenset({"image", "images", "duration", "resolution_name", "aspect_ratio", "size"})
_DROPPED_KEYS: Final = frozenset({"user", "extra_headers", "model"})
_DEFAULT_DURATION_SECONDS: Final = 6

_STATUS_ALIASES: Final[Mapping[str, str]] = {
    "done": "completed",
    "completed": "completed",
    "succeeded": "completed",
    "failed": "failed",
    "expired": "failed",
    "pending": "processing",
    "processing": "processing",
    "in_progress": "processing",
}

_INT: Final = TypeAdapter(int)


class _VideoMeta(BaseModel, frozen=True):
    url: str | None = None
    duration: int | None = None


class _VideoCreatePayload(BaseModel, frozen=True):
    request_id: str | None = None
    id: str | None = None
    usage: Mapping[str, object] | None = None


class _VideoStatusPayload(BaseModel, frozen=True):
    status: str | None = None
    video: _VideoMeta | None = None
    request_id: str | None = None
    id: str | None = None
    created_at: int | None = None
    model: str | None = None
    progress: int | None = None
    usage: Mapping[str, object] | None = None


class _VideoContentPayload(BaseModel, frozen=True):
    status: str | None = None
    video: _VideoMeta | None = None


_CREATE_PAYLOAD: Final = TypeAdapter(_VideoCreatePayload)
_STATUS_PAYLOAD: Final = TypeAdapter(_VideoStatusPayload)
_CONTENT_PAYLOAD: Final = TypeAdapter(_VideoContentPayload)


def _keys_if(present: bool, *keys: str) -> frozenset[str]:
    return frozenset(keys) if present else frozenset()


def _duration(value: object) -> int:
    try:
        return _INT.validate_python(value)
    except ValidationError:
        return _DEFAULT_DURATION_SECONDS


def _as_params(litellm_params: GenericLiteLLMParams | Mapping[str, object] | None) -> Mapping[str, object]:
    if isinstance(litellm_params, GenericLiteLLMParams):
        return litellm_params.model_dump()
    return litellm_params or {}


def _sync_client() -> HTTPHandler:
    return _get_httpx_client()  # pyright: ignore[reportUnknownVariableType]  # factory stub leaves params untyped


def _async_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(  # pyright: ignore[reportUnknownVariableType]  # factory stub leaves params untyped
        llm_provider=litellm.LlmProviders.XAI,
    )


class XAIVideoConfig(BaseVideoConfig):
    """
    xAI Imagine video generation.

    Create: POST /v1/videos/generations  -> {request_id}
    Status: GET  /v1/videos/{request_id} -> {status, video.url, ...}
    """

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig contract returns list
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseVideoConfig contract
        params: Final[Mapping[str, object]] = video_create_optional_params
        seconds_replace: Final = "seconds" in params and "duration" not in params
        size_replaces: Final = "size" in params and "aspect_ratio" not in params
        reference_replaces: Final = "input_reference" in params and "image" not in params
        size: Final = params.get("size")

        dropped: Final = (
            _DROPPED_KEYS
            | _keys_if(seconds_replace, "seconds")
            | _keys_if(size_replaces, "size")
            | _keys_if(reference_replaces, "input_reference")
        )
        return {
            **{key: value for key, value in params.items() if key not in dropped},
            **({"duration": _duration(params.get("seconds"))} if seconds_replace else {}),
            **({"aspect_ratio": _SIZE_TO_ASPECT_RATIO.get(str(size), "16:9")} if size_replaces and size else {}),
            **({"image": params.get("input_reference")} if reference_replaces else {}),
        }

    def _resolve_api_base(
        self,
        api_base: str | None,
        api_key: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        resolved: Final = xai_api_root(
            api_base or xai_optional_str(litellm_params.get("api_base")),
            api_key,
            litellm_params,
        )
        return resolved.rstrip("/")

    def _v1_root(self, api_base: str) -> str:
        base: Final = api_base.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract
        params: Final = _as_params(litellm_params)
        effective_key: Final = api_key or (litellm_params.api_key if litellm_params is not None else None)
        return xai_json_headers(
            headers,
            xai_authorization(model or "xai-video", effective_key, params, "video generation"),
        )

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        resolved: Final = self._resolve_api_base(
            api_base=api_base,
            api_key=xai_optional_str(litellm_params.get("api_key")),
            litellm_params=litellm_params,
        )
        # Empty model used by status/content handlers that only need the root.
        if not model:
            return self._v1_root(resolved)
        return f"{self._v1_root(resolved)}/videos/generations"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract
        passthrough: Final = {
            key: value
            for key, value in video_create_optional_request_params.items()
            if key in _CREATE_PASSTHROUGH_KEYS and value is not None
        }
        request: Final = {
            "model": XAIModelInfo.get_base_model(model) or model,
            **({"prompt": prompt} if prompt else {}),
            **passthrough,
            "duration": passthrough.get("duration", _DEFAULT_DURATION_SECONDS),
        }
        return request, [], api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        payload: Final = _CREATE_PAYLOAD.validate_python(raw_response.json())
        request_id: Final = payload.request_id or payload.id
        if not request_id:
            raise ValueError(f"xAI video generation response missing request_id: {payload}")

        return VideoObject(
            id=(
                encode_video_id_with_provider(request_id, custom_llm_provider, model)
                if custom_llm_provider
                else request_id
            ),
            object="video",
            status="processing",
            created_at=int(time.time()),
            model=XAIModelInfo.get_base_model(model) or model,
            progress=0,
            usage=dict(payload.usage) if payload.usage is not None else {},
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        original_id: Final = extract_original_video_id(video_id)
        # Status is GET /v1/videos/{request_id} (not under /videos/generations)
        return f"{self._v1_root(api_base)}/videos/{original_id}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        client: "HTTPHandler | None" = None,
    ) -> VideoObject:
        payload: Final = _STATUS_PAYLOAD.validate_python(raw_response.json())
        status_raw: Final = (payload.status or "processing").lower()
        status: Final = _STATUS_ALIASES.get(status_raw, status_raw)
        video_url: Final = payload.video.url if payload.video is not None else None
        # Prefer the provider id in the body; fall back to the CDN file name.
        request_id: Final = (
            payload.request_id
            or payload.id
            or (video_url.split("/")[-1].replace(".mp4", "") if video_url else "unknown")
        )

        video_obj: Final = VideoObject(
            id=(
                encode_video_id_with_provider(str(request_id), custom_llm_provider, payload.model)
                if custom_llm_provider and str(request_id) != "unknown"
                else str(request_id)
            ),
            object="video",
            status=status,
            created_at=payload.created_at or int(time.time()),
            completed_at=int(time.time()) if status == "completed" else None,
            model=payload.model,
            progress=payload.progress,
            seconds=str(payload.video.duration)
            if payload.video is not None and payload.video.duration is not None
            else None,
            usage=dict(payload.usage) if payload.usage is not None else {},
        )
        video_obj.hidden_params["video_url"] = video_url
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        original_id: Final = extract_original_video_id(video_id)
        return f"{self._v1_root(api_base)}/videos/{original_id}", {}

    def _video_cdn_url(self, raw_response: httpx.Response) -> str | None:
        content_type: Final = (raw_response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type and raw_response.content[:1] != b"{":
            return None
        payload: Final = _CONTENT_PAYLOAD.validate_python(raw_response.json())
        url: Final = payload.video.url if payload.video is not None else None
        if url:
            return url
        raise ValueError(f"xAI video not ready for download (status={payload.status})")

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        client: "HTTPHandler | None" = None,
    ) -> bytes:
        url: Final = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        httpx_client: Final = client or _sync_client()
        video_response: Final = httpx_client.get(url)  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get stub leaves params untyped
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        client: "AsyncHTTPHandler | None" = None,
    ) -> bytes:
        url: Final = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        async_httpx_client: Final = client or _async_client()
        video_response: Final = await async_httpx_client.get(url)  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get stub leaves params untyped
        video_response.raise_for_status()
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("Video remix is not supported by xAI Imagine API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by xAI Imagine API")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")
