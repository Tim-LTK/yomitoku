"""Phase 2.1 practice engine — programmatic tiers + Claude Tier 2 + batch submit grading."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import anthropic
from pydantic import ValidationError

from yomitoku_api.config import Settings
from yomitoku_api.exceptions import GenerationFailedError, MissingApiKeyError
from yomitoku_api.schemas import (
    GapInterval,
    GrammarRole,
    KnowledgeGap,
    PracticeItem,
    PracticeResult,
    QuestionType,
    RawOutput,
    SessionResult,
    SessionSubmission,
)
from yomitoku_api.services import prompts as prompt_service
from yomitoku_api.services import srs_compute as srs_compute_gen
from yomitoku_api.services import validate as validate_service
from yomitoku_api.services.prompts import PromptBundle

logger = logging.getLogger(__name__)


_CONJUGATE_TARGET_LABEL: dict[GrammarRole | str, str] = {
    "verb_base": "辞書形（基本形）",
    "verb_te_form": "て形",
    "verb_ending": "文に合った語尾／活用形",
}


def _assistant_text(message: Any) -> str:
    parts: list[str] = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts).strip()


def _generate_raw(settings: Settings, bundle: PromptBundle) -> RawOutput:
    """Shared Anthropic Messages call."""

    if not settings.anthropic_api_key.strip():
        raise MissingApiKeyError()

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    message = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=8192,
        system=bundle.system,
        messages=[{"role": "user", "content": [{"type": "text", "text": bundle.user}]}],
    )
    text = _assistant_text(message)
    if not text:
        raise GenerationFailedError()

    logger.info(
        "anthropic.practice.phase21",
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


def determine_question_type(gap: KnowledgeGap) -> QuestionType:
    """Maps gap → question modality; SRS gate unlocks Tier 2 application tier."""

    hist = gap.practiceResults or []
    if len(hist) >= 2 and hist[-1].qualityScore >= 4 and hist[-2].qualityScore >= 4:
        return "application_mc"

    role = gap.element.role
    if role in ("verb_base", "verb_te_form", "verb_ending"):
        return "conjugate"
    if role == "grammar_pattern":
        return "fill_blank"
    if role in ("noun", "adjective_i", "adjective_na"):
        return "translate"

    return "fill_blank"


def compose_session(gaps: list[KnowledgeGap], max_items: int = 15) -> list[KnowledgeGap]:
    """Schedule gaps by SRS due date with per-role staggering."""

    def sort_key(gap: KnowledgeGap) -> tuple[int, str]:
        nr = gap.nextReviewAt
        if nr is None or not str(nr).strip():
            return (0, "")
        return (1, str(nr).strip())

    ranked = sorted(gaps, key=sort_key)
    out: list[KnowledgeGap] = []
    counts: dict[str, int] = {}

    for g in ranked:
        role_key = str(g.element.role)
        if counts.get(role_key, 0) >= 3:
            continue
        out.append(g)
        counts[role_key] = counts.get(role_key, 0) + 1
        if len(out) >= max_items:
            break

    return out


def _unique_item_ids(items: list[PracticeItem]) -> list[PracticeItem]:
    seen: set[str] = set()
    patched: list[PracticeItem] = []
    for it in items:
        iid = it.itemId
        if iid in seen:
            iid = f"{iid}-{uuid.uuid4().hex[:8]}"
        seen.add(iid)
        patched.append(it if iid == it.itemId else it.model_copy(update={"itemId": iid}))
    return patched


def build_tier0_question(gap: KnowledgeGap, question_type: QuestionType) -> PracticeItem:
    """Deterministic blanks / conjugation items — no Claude."""

    if question_type not in ("fill_blank", "conjugate"):
        raise ValueError("tier0 expects fill_blank / conjugate only")

    item_id = str(uuid.uuid4())
    surface = gap.element.text.strip()

    if question_type == "fill_blank":
        src = gap.sourceSentence.strip()
        if surface and surface in src:
            masked = src.replace(surface, "___", 1)
        else:
            masked = src
            logger.warning(
                "practice.fill_blank_no_span",
                extra={"gapId": gap.id, "surface": surface},
            )

        prompt = f"Fill in the blank: {masked}"
        return PracticeItem(
            itemId=item_id,
            gapId=gap.id,
            questionType="fill_blank",
            prompt=prompt,
            hint=None,
            options=None,
            canonicalAnswer=surface or None,
        )

    role = gap.element.role
    base_form = gap.element.text.strip() or gap.element.reading.strip()
    tgt_label = _CONJUGATE_TARGET_LABEL.get(role, "指定の活用形")

    conjugate_prompt = f"Conjugate {base_form} → {tgt_label}"
    return PracticeItem(
        itemId=item_id,
        gapId=gap.id,
        questionType="conjugate",
        prompt=conjugate_prompt,
        hint=None,
        options=None,
        canonicalAnswer=surface or None,
    )


def build_tier1_question(gap: KnowledgeGap) -> PracticeItem:
    """Plain translation prompt anchored on source sentence."""

    item_id = str(uuid.uuid4())
    return PracticeItem(
        itemId=item_id,
        gapId=gap.id,
        questionType="translate",
        prompt=f"Translate into English: {gap.sourceSentence.strip()}",
        hint=None,
        options=None,
        canonicalAnswer=None,
    )


def _slug_fragment(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower().strip()) if text.strip() else "item"
    return cleaned.strip("-") or "item"


def _parse_tier2_payload(payload: dict[str, Any] | Any, expected: int) -> list[PracticeItem | None]:
    if not isinstance(payload, dict):
        return [None] * expected

    items_any = payload.get("items")
    if not isinstance(items_any, list):
        return [None] * expected

    out: list[PracticeItem | None] = []
    for idx in range(expected):
        blob = items_any[idx] if idx < len(items_any) else None
        if not isinstance(blob, dict):
            out.append(None)
            continue
        try:
            out.append(PracticeItem.model_validate(blob))
        except ValidationError:
            out.append(None)
    return out


def _tier2_fallback_fill_blank(gap: KnowledgeGap, question_type: QuestionType) -> PracticeItem:
    slug = _slug_fragment(gap.element.text)
    slug_id = f"{slug}-{question_type}"
    surface = gap.element.text.strip()
    if surface and surface in gap.sourceSentence:
        masked = gap.sourceSentence.replace(surface, "___", 1)
    else:
        masked = gap.sourceSentence
    return PracticeItem(
        itemId=slug_id,
        gapId=gap.id,
        questionType="fill_blank",
        prompt=f"Fill in the blank: {masked}",
        hint=None,
        options=None,
        canonicalAnswer=surface or None,
    )


def _validate_tier2_item_shape(item: PracticeItem, desired: QuestionType) -> bool:
    if desired != item.questionType:
        return False
    if desired == "application_mc":
        return len(item.options or []) == 4
    if desired == "nuance_choice":
        return len(item.options or []) == 2
    return False


def generate_tier2_questions(
    settings: Settings,
    specs: list[tuple[KnowledgeGap, QuestionType]],
    *,
    student_context: str,
) -> list[PracticeItem]:
    """Single Claude invocation for Tier 2 drill synthesis."""

    if not specs:
        return []

    spec_payload = [
        {"gapId": gap.id, "questionType": qt, "gap": gap.model_dump(mode="json", by_alias=True)}
        for gap, qt in specs
    ]
    gap_specs_json = json.dumps(spec_payload, ensure_ascii=False, indent=2)
    bundle = prompt_service.build_practice_generate_tier2_bundle(
        settings,
        gap_specs_json=gap_specs_json,
        student_context=student_context,
    )
    raw = _generate_raw(settings, bundle)
    text = validate_service.strip_code_fences(raw.raw_text)

    payload: dict[str, Any]
    try:
        loaded = json.loads(text)
        payload = loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        payload = {}

    parsed = _parse_tier2_payload(payload, len(specs))
    fallback: list[PracticeItem] = []

    for idx, (gap, qt) in enumerate(specs):
        target = parsed[idx] if idx < len(parsed) else None
        if target is None or not _validate_tier2_item_shape(target, qt):
            logger.warning(
                "practice.tier2_fallback",
                extra={"gapId": gap.id, "desired": qt},
            )
            fallback.append(_tier2_fallback_fill_blank(gap, qt))
        else:
            fallback.append(target)

    return fallback


def compose_practice_session_items(
    settings: Settings,
    gaps: list[KnowledgeGap],
    *,
    student_context: str,
) -> list[PracticeItem]:
    """Main compose pipeline powering `/practice/generate`."""

    selected = compose_session(gaps)
    tier2_specs: list[tuple[KnowledgeGap, QuestionType]] = []
    ordered_specs: list[tuple[KnowledgeGap, QuestionType]] = []

    for gap in selected:
        qt = determine_question_type(gap)
        ordered_specs.append((gap, qt))
        if qt == "application_mc":
            tier2_specs.append((gap, qt))

    generated_tier2: list[PracticeItem] = []
    if tier2_specs:
        generated_tier2 = generate_tier2_questions(
            settings, tier2_specs, student_context=student_context
        )

    tier_iter = iter(generated_tier2)

    final: list[PracticeItem] = []
    for gap, qt in ordered_specs:
        if qt == "translate":
            final.append(build_tier1_question(gap))
            continue

        if qt == "application_mc":
            final.append(next(tier_iter))
            continue

        final.append(build_tier0_question(gap, qt))

    patched = _unique_item_ids(final)
    dup_issues = validate_service.validate_practice_item_ids_unique(patched)
    if dup_issues:
        raise ValueError("practice_generate_duplicate_item_ids")

    return patched


def _programmatic_floor(item: PracticeItem, user_answer: str) -> int | None:
    if item.questionType not in ("fill_blank", "conjugate"):
        return None

    canon = (item.canonical_answer or "").strip()
    if canon == "":
        return None

    ua = user_answer.strip()
    return 5 if ua == canon else 1


def _default_interval_fallback(gap: KnowledgeGap) -> tuple[int, str]:
    base_days = gap.intervalDays or 3
    days = max(1, min(366, base_days))
    iso_dt = datetime.now(UTC) + timedelta(days=days)
    iso = iso_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return days, iso


def compute_gap_interval_after_session(
    settings: Settings,
    gap: KnowledgeGap,
    session_results_for_gap: list[PracticeResult],
    *,
    student_context: str,
) -> GapInterval:
    """Reuse Claude SRS envelope — identical path to `/srs/compute` without HTTP hops."""

    history = [*list(gap.practiceResults or []), *session_results_for_gap]
    if not history:
        days, iso = _default_interval_fallback(gap)
        return GapInterval(
            gapId=gap.id,
            intervalDays=days,
            nextReviewAt=iso,
        )

    bundle = prompt_service.build_srs_compute_bundle(
        settings,
        gap=gap,
        results=history,
        student_context=student_context,
    )
    raw = srs_compute_gen.generate_srs_schedule(settings, bundle)
    validation = validate_service.validate_srs_compute(raw)

    if not validation.is_valid or validation.srs_compute is None:
        logger.warning("practice.srs_interval_fallback", extra={"gapId": gap.id})
        days, iso = _default_interval_fallback(gap)
        return GapInterval(
            gapId=gap.id,
            intervalDays=days,
            nextReviewAt=iso,
        )

    sc = validation.srs_compute
    return GapInterval(
        gapId=gap.id,
        intervalDays=int(sc.suggestedIntervalDays),
        nextReviewAt=sc.nextReviewAt,
    )


def save_practice_result_stub(gap_id: str, practice_item_id: str, result: PracticeResult) -> None:
    """Persist hook — SRS rows live client-side/Prompt 3; log only."""

    logger.debug(
        "practice.result_stub",
        extra={"gapId": gap_id, "practiceItemId": practice_item_id, "score": result.qualityScore},
    )


def finalize_session_results(
    settings: Settings,
    submission: SessionSubmission,
    *,
    student_context: str,
) -> SessionResult:
    """POST /practice/submit orchestration."""

    pits = submission.practice_items
    answers = submission.items
    if len(pits) != len(answers):
        raise ValueError("practice_submit_length_mismatch")

    for pit, sess in zip(pits, answers, strict=True):
        if sess.practice_item_id != pit.itemId:
            raise ValueError("practice_submit_id_mismatch")

    gap_by_id = {g.id: g for g in submission.gaps}
    for pit in pits:
        if pit.gapId not in gap_by_id:
            raise ValueError("practice_submit_unknown_gap")

    prog_scores: dict[int, int | None] = {}
    rows_for_ai: list[dict[str, Any]] = []
    for idx, (pit, ans) in enumerate(zip(pits, answers, strict=True)):
        ua = ans.user_answer
        preset = _programmatic_floor(pit, ua)
        prog_scores[idx] = preset

        gap = gap_by_id[pit.gapId]
        row = {
            "practiceItem": pit.model_dump(mode="json", by_alias=True),
            "gapId": pit.gapId,
            "userAnswer": ua,
            "priorQualityScore": preset,
            "tierNote": pit.questionType,
            "sentenceContext": gap.sourceSentence,
            "canonicalAnswer": pit.canonical_answer,
        }
        rows_for_ai.append(row)

    bundle = prompt_service.build_practice_submit_bundle(
        settings,
        submission_rows_json=json.dumps(rows_for_ai, ensure_ascii=False, indent=2),
        student_context=student_context,
    )
    raw = _generate_raw(settings, bundle)
    validation_ai = validate_service.validate_session_submit_generation(
        raw,
        expected_count=len(pits),
    )
    envelope = validation_ai.practice_submit_envelope
    if not validation_ai.is_valid or envelope is None:
        raise ValueError(
            "; ".join(i.message for i in validation_ai.issues)
            if validation_ai.issues
            else "practice_submit_generation_invalid",
        )

    merged_results: list[PracticeResult] = []
    for idx, ai_row in enumerate(envelope.results):
        deterministic = prog_scores.get(idx)
        score = deterministic if deterministic is not None else ai_row.qualityScore
        merged_results.append(
            PracticeResult(
                qualityScore=score,
                feedback=ai_row.feedback,
                errorTags=ai_row.errorTags,
            )
        )

    aggregated: dict[str, list[PracticeResult]] = {gid: [] for gid in gap_by_id}
    for pit, mr in zip(pits, merged_results, strict=True):
        aggregated[pit.gapId].append(mr)

    intervals: list[GapInterval] = []
    for gap in submission.gaps:
        interval_out = compute_gap_interval_after_session(
            settings,
            gap,
            aggregated.get(gap.id, []),
            student_context=student_context,
        )
        intervals.append(interval_out)

    for idx, mr in enumerate(merged_results):
        save_practice_result_stub(pits[idx].gapId, pits[idx].itemId, mr)

    return SessionResult(
        results=merged_results,
        tutorNotes=envelope.tutor_notes.strip(),
        intervals=intervals,
    )
