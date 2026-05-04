"""Generation layer — targeted reading scan + passage-grounded Q&A (Phase 1.7)."""

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


def generate_targeted_scan(settings: Settings, bundle: PromptBundle) -> RawOutput:
    """Claude emits `ScanEnvelope` JSON."""

    if not settings.anthropic_api_key.strip():
        raise MissingApiKeyError()

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=8192,
        system=bundle.system,
        messages=[{"role": "user", "content": [{"type": "text", "text": bundle.user}]}],
    )
    text = _assistant_text_from_message(message)
    if not text:
        raise GenerationFailedError()

    logger.info(
        "anthropic.generation.targeted_scan",
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


def generate_ask_answer(settings: Settings, bundle: PromptBundle) -> RawOutput:
    """Claude returns free-text prose for `POST /ask`."""

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
        "anthropic.generation.scan_ask",
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
