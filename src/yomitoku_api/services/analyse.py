"""Grammar breakdown generation — Claude `RawOutput` plus chunked analyse orchestration."""

import logging
import re
from typing import Any

import anthropic

from yomitoku_api.config import Settings
from yomitoku_api.exceptions import GenerationFailedError, MissingApiKeyError
from yomitoku_api.schemas import RawOutput, SentenceBreakdown, ValidationIssue, ValidationResult
from yomitoku_api.services import prompts as prompt_service
from yomitoku_api.services import validate as validate_service
from yomitoku_api.services.prompts import PromptBundle

logger = logging.getLogger(__name__)

_CHUNK_SENT_BOUNDARY_RE = re.compile(r"(?<=[。！？])")

_MAX_SENTENCES_PER_CHUNK_DEFAULT = 3


def chunk_japanese_text_for_analysis(
    japanese_text: str,
    *,
    max_sentences_per_chunk: int = _MAX_SENTENCES_PER_CHUNK_DEFAULT,
) -> list[str]:
    """Split prose on sentence-final punctuation then group up to N sentences per API chunk."""

    text = japanese_text.strip()
    if not text:
        return []
    fragments = [
        fragment.strip()
        for fragment in _CHUNK_SENT_BOUNDARY_RE.split(text)
        if fragment.strip()
    ]
    if not fragments:
        return [text]

    chunks: list[str] = []
    bucket: list[str] = []

    def flush_bucket() -> None:
        if bucket:
            joined = "".join(bucket).strip()
            if joined:
                chunks.append(joined)
            bucket.clear()

    for fragment in fragments:
        bucket.append(fragment)
        if len(bucket) >= max_sentences_per_chunk:
            flush_bucket()

    flush_bucket()

    out = [c.strip() for c in chunks if c.strip()]
    return out if out else [text]


def _assistant_text_from_message(message: Any) -> str:
    parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def generate_sentence_breakdowns(settings: Settings, bundle: PromptBundle) -> RawOutput:
    """Calls Claude text completion for breakdown JSON — strictly separate from extraction."""

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
        "anthropic.generation.breakdown",
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


def run_chunked_sentence_breakdown_analysis(
    settings: Settings,
    japanese_text: str,
    *,
    student_context: str,
) -> ValidationResult:
    """Split input into sentence groups, validate each reply, merge valid breakdowns.

    Invalid chunks are logged and omitted; callers only receive 422 when every chunk fails.
    """

    chunks = chunk_japanese_text_for_analysis(japanese_text)
    merged: list[SentenceBreakdown] = []
    all_issues: list[ValidationIssue] = []
    failed_chunks = 0

    logger.info(
        "analyse.chunked_plan",
        extra={"segment_count": len(chunks)},
    )

    for idx, chunk in enumerate(chunks):
        bundle = prompt_service.build_breakdown_analysis_bundle(
            settings,
            chunk,
            student_context=student_context,
        )
        raw = generate_sentence_breakdowns(settings, bundle)
        validation = validate_service.validate_breakdown_generation(raw)

        if not validation.is_valid or validation.breakdowns is None:
            failed_chunks += 1
            chunked_issues = [
                ValidationIssue(
                    code=i.code,
                    message=f"chunk[{idx}]: {i.message}",
                )
                for i in validation.issues
            ]
            all_issues.extend(chunked_issues)
            logger.warning(
                "analyse.chunk_validation_failed",
                extra={
                    "chunk_index": idx,
                    "issue_count": len(chunked_issues),
                    "codes": [i.code for i in chunked_issues],
                },
            )
            continue

        merged.extend(validation.breakdowns)

    if chunks and not merged:
        logger.info(
            "analyse.chunked_validation_failure_all_chunks",
            extra={
                "issue_count": len(all_issues),
                "chunk_count": len(chunks),
            },
        )
        return ValidationResult(
            is_valid=False,
            issues=all_issues,
            breakdowns=None,
        )

    if merged and failed_chunks:
        logger.info(
            "analyse.chunked_partial_success",
            extra={
                "total_chunks": len(chunks),
                "failed_chunks": failed_chunks,
                "breakdown_sentence_count": len(merged),
            },
        )

    return ValidationResult(is_valid=True, issues=[], breakdowns=merged)
