"""Generation layer for Phase 2 practice — `/practice/generate` and `/practice/evaluate` stay isolated."""

import logging
from typing import Any

import anthropic

from yomitoku_api.config import Settings
from yomitoku_api.exceptions import GenerationFailedError, MissingApiKeyError
from yomitoku_api.schemas import RawOutput
from yomitoku_api.services.prompts import PromptBundle

logger = logging.getLogger(__name__)


def _assistant_text_from_message(message: Any) -> str:
    parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def generate_practice_items(settings: Settings, bundle: PromptBundle) -> RawOutput:
    """Claude emits `PracticeGenerateEnvelope` JSON — never merges with analyse payload."""

    if not settings.anthropic_api_key.strip():
        raise MissingApiKeyError()

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=4096,
        system=bundle.system,
        messages=[{"role": "user", "content": [{"type": "text", "text": bundle.user}]}],
    )
    text = _assistant_text_from_message(message)
    if not text:
        raise GenerationFailedError()

    logger.info(
        "anthropic.generation.practice_generate",
        extra={
            "model": settings.anthropic_model,
            "prompt_versions": bundle.prompt_versions,
            "stop_reason": message.stop_reason,
        },
    )
    return RawOutput(
        raw_text=text,
        model_id=settings.anthropic_model,
        prompt_versions=dict(bundle.prompt_versions),
    )


def evaluate_practice_attempt(settings: Settings, bundle: PromptBundle) -> RawOutput:
    """Claude emits `PracticeEvaluateEnvelope` wrapping `PracticeResult` — separate endpoint from `/generate`."""

    if not settings.anthropic_api_key.strip():
        raise MissingApiKeyError()

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2048,
        system=bundle.system,
        messages=[{"role": "user", "content": [{"type": "text", "text": bundle.user}]}],
    )
    text = _assistant_text_from_message(message)
    if not text:
        raise GenerationFailedError()

    logger.info(
        "anthropic.generation.practice_evaluate",
        extra={
            "model": settings.anthropic_model,
            "prompt_versions": bundle.prompt_versions,
            "stop_reason": message.stop_reason,
        },
    )
    return RawOutput(
        raw_text=text,
        model_id=settings.anthropic_model,
        prompt_versions=dict(bundle.prompt_versions),
    )
