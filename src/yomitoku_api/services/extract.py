"""Generation layer for vision text extraction — returns `RawOutput` only."""

import logging
from typing import Any, cast

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


def generate_text_from_image(
    settings: Settings,
    bundle: PromptBundle,
    *,
    image_base64: str,
    media_type: str,
) -> RawOutput:
    """Calls Claude vision with a separate extraction prompt — never mixed with breakdown."""

    if not settings.anthropic_api_key.strip():
        raise MissingApiKeyError()

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    user_blocks = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_base64,
            },
        },
        {
            "type": "text",
            "text": bundle.user,
        },
    ]

    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=2048,
        system=bundle.system,
        # SDK TypedDicts lag behind valid multimodal dict literals — trust runtime layout.
        messages=cast(Any, [{"role": "user", "content": user_blocks}]),
    )
    text = _assistant_text_from_message(message)
    if not text:
        raise GenerationFailedError()

    logger.info(
        "anthropic.generation.extract",
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
