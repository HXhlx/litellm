from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import ConfigDict, TypeAdapter, ValidationError

_SEND_VIA_EXTRA_BODY: Final = "extra_body"
_SEND_VIA_PROVIDER_MAPPED: Final = "provider_mapped"

_ENABLED_STRINGS: Final = frozenset({"enabled", "true", "1", "auto"})
_ENABLED_TYPE_VALUES: Final = frozenset({"enabled", "auto", "true"})

_EFFORT_FALLBACKS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "xhigh": ("max", "high"),
        "max": ("xhigh", "high"),
        "medium": ("high", "low"),
        "minimal": ("low", "none"),
        "none": ("low",),
    }
)

_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_OBJECTS: Final = TypeAdapter(tuple[object, ...], config=ConfigDict(hide_input_in_errors=True))


@dataclass(frozen=True, slots=True)
class ThinkingParamsState:
    thinking: object | None
    reasoning_effort: object | None
    extra_body: Mapping[str, object]


def _as_mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _OBJECT.validate_python(value)
    except ValidationError:
        return None


def extra_body_mapping(value: object) -> Mapping[str, object]:
    """Normalize a caller's extra_body into a read-only mapping, treating anything else as empty."""
    payload: Final = _as_mapping(value)
    return payload if payload is not None else MappingProxyType({})


def _as_str_tuple(value: object) -> tuple[str, ...]:
    try:
        items: Final = _OBJECTS.validate_python(value)
    except ValidationError:
        return ()
    return tuple(item for item in items if isinstance(item, str))


def _thinking_enabled(thinking: object) -> bool:
    if isinstance(thinking, bool):
        return thinking
    if isinstance(thinking, str):
        return thinking.lower() in _ENABLED_STRINGS
    payload: Final = _as_mapping(thinking)
    if payload is None:
        return False
    typ: Final = payload.get("type")
    if isinstance(typ, str):
        return typ.lower() in _ENABLED_TYPE_VALUES
    enabled: Final = payload.get("enabled")
    return enabled if isinstance(enabled, bool) else False


def _thinking_candidate(thinking: object) -> str | None:
    match thinking:
        case str():
            return thinking
        case bool():
            return "enabled" if thinking else "disabled"
        case _:
            payload: Final = _as_mapping(thinking)
            if payload is None:
                return None
            raw: Final = payload.get("type")
            return raw if isinstance(raw, str) else None


def _thinking_type_value(thinking: object, allowed: Sequence[str]) -> str | None:
    candidate: Final = _thinking_candidate(thinking)
    if candidate is None:
        return None
    if not allowed or candidate in allowed:
        return candidate
    if candidate == "auto" and "enabled" in allowed:
        return "enabled"
    return None


def _thinking_payload(thinking: object, typ: str) -> Mapping[str, object]:
    payload: Final = _as_mapping(thinking)
    budget: Final = payload.get("budget_tokens") if payload is not None else None
    if isinstance(budget, int):
        return MappingProxyType({"type": typ, "budget_tokens": budget})
    return MappingProxyType({"type": typ})


def _clamp_effort(value: object, allowed: Sequence[str]) -> str | None:
    if not isinstance(value, str):
        return None
    if not allowed:
        return None
    if value in allowed:
        return value
    for fallback in _EFFORT_FALLBACKS.get(value, ()):
        if fallback in allowed:
            return fallback
    return None


def _map_thinking_to_extra_body(
    *,
    thinking_param: str | None,
    thinking: object,
    thinking_values: Sequence[str],
) -> Mapping[str, object]:
    match thinking_param:
        case "thinking.type":
            typ: Final = _thinking_type_value(thinking, thinking_values)
            if typ is None:
                return MappingProxyType({})
            return MappingProxyType({"thinking": dict(_thinking_payload(thinking, typ))})
        case "thinking":
            payload: Final = _as_mapping(thinking)
            if payload is not None:
                return MappingProxyType({"thinking": dict(payload)})
            typ_only: Final = _thinking_type_value(thinking, thinking_values or ("enabled", "disabled"))
            if typ_only is None:
                return MappingProxyType({})
            return MappingProxyType({"thinking": {"type": typ_only}})
        case "enable_thinking":
            return MappingProxyType({"enable_thinking": _thinking_enabled(thinking)})
        case "chat_template_kwargs":
            return MappingProxyType({"chat_template_kwargs": {"enable_thinking": _thinking_enabled(thinking)}})
        case _:
            return MappingProxyType({})


def translate_thinking_params(
    *,
    model_info: Mapping[str, object] | None,
    state: ThinkingParamsState,
) -> ThinkingParamsState:
    if model_info is None:
        return state

    send_via: Final = model_info.get("thinking_send_via")
    if send_via not in {_SEND_VIA_EXTRA_BODY, _SEND_VIA_PROVIDER_MAPPED}:
        return state

    supports_reasoning: Final = model_info.get("supports_reasoning") is True
    thinking_param_raw: Final = model_info.get("thinking_param")
    thinking_param: Final = thinking_param_raw if isinstance(thinking_param_raw, str) else None
    thinking_values: Final = _as_str_tuple(model_info.get("thinking_values"))
    effort_values: Final = _as_str_tuple(model_info.get("reasoning_effort_values"))

    if not supports_reasoning and send_via != _SEND_VIA_PROVIDER_MAPPED:
        return state

    thinking: Final = state.thinking
    effort: Final = state.reasoning_effort
    if thinking is None and effort is None:
        return state

    keep_thinking: Final = send_via == _SEND_VIA_PROVIDER_MAPPED

    thinking_patch: Final[Mapping[str, object]] = (
        _map_thinking_to_extra_body(
            thinking_param=thinking_param,
            thinking=thinking,
            thinking_values=thinking_values,
        )
        if thinking is not None and send_via == _SEND_VIA_EXTRA_BODY
        else MappingProxyType({})
    )

    clamped: Final = _clamp_effort(effort, effort_values) if effort is not None else None
    effort_patch: Final = (
        {"reasoning_effort": clamped}
        if clamped is not None
        else (
            {"reasoning_effort": effort}
            if effort is not None and not effort_values and send_via == _SEND_VIA_EXTRA_BODY and isinstance(effort, str)
            else {}
        )
    )

    patch: Final = {**thinking_patch, **effort_patch}
    if not patch:
        return state

    thinking_mapped: Final = any(key in patch for key in ("thinking", "enable_thinking", "chat_template_kwargs"))
    merged_extra: Final = MappingProxyType({**dict(state.extra_body), **patch})
    return ThinkingParamsState(
        thinking=thinking if (keep_thinking or not thinking_mapped) else None,
        reasoning_effort=None if "reasoning_effort" in patch else effort,
        extra_body=merged_extra,
    )


def apply_thinking_param_translation(
    *,
    model_info: Mapping[str, object] | None,
    thinking: object | None,
    reasoning_effort: object | None,
    existing_extra_body: Mapping[str, object] | None,
) -> ThinkingParamsState:
    base_extra: Final[Mapping[str, object]] = (
        dict(existing_extra_body) if existing_extra_body is not None else MappingProxyType({})
    )
    return translate_thinking_params(
        model_info=model_info,
        state=ThinkingParamsState(
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            extra_body=base_extra,
        ),
    )
