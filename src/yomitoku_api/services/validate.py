"""Validation layer — discrete checks aggregated into `ValidationResult` objects."""

import json
import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from yomitoku_api.schemas import (
    AnalyseEnvelope,
    BreakdownElement,
    ExplainEnvelope,
    OnboardingAssessEnvelope,
    PracticeEvaluateEnvelope,
    PracticeGenerateEnvelope,
    PracticeItem,
    RawOutput,
    SentenceBreakdown,
    SrsComputeResponse,
    StudentProfile,
    ValidationIssue,
    ValidationResult,
)

logger = logging.getLogger(__name__)


def strip_code_fences(raw: str) -> str:
    """Remove Markdown code fences Claude sometimes wraps JSON in."""

    text = raw.strip()
    if text.startswith("```"):
        segment = text[3:]
        newline = segment.find("\n")
        if newline != -1:
            segment = segment[newline + 1 :]
        text = segment
    trimmed = text.rstrip()
    if trimmed.endswith("```"):
        trimmed = trimmed[: trimmed.rfind("```")].rstrip()
    return trimmed.strip()


def issue_json_decode(exc: json.JSONDecodeError) -> ValidationIssue:
    """Named packaging for schema parse failures at the JSON layer."""

    return ValidationIssue(code="json_parse_failed", message=f"Malformed JSON ({exc}).")


def issue_pydantic_validation(exc: ValidationError) -> list[ValidationIssue]:
    """Flattens Pydantic field errors without dropping locations."""

    out: list[ValidationIssue] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        msg = err.get("msg", "invalid")
        out.append(ValidationIssue(code="schema_mismatch", message=f"{loc}: {msg}"))
    return out


def validate_breakdown_collections_nonempty(
    breakdowns: list[SentenceBreakdown],
) -> list[ValidationIssue]:
    """Each sentence row must expose at least one segmented element."""

    issues: list[ValidationIssue] = []
    if not breakdowns:
        issues.append(
            ValidationIssue(
                code="breakdown_list_empty",
                message="Provide at least one sentence breakdown.",
            )
        )
    return issues


def validate_original_sentence_non_empty(
    breakdowns: list[SentenceBreakdown],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for i, bd in enumerate(breakdowns):
        if not bd.original.strip():
            issues.append(
                ValidationIssue(
                    code="sentence_original_empty",
                    message=f"breakdowns[{i}].original must be non-empty.",
                )
            )
    return issues


def validate_elements_non_empty_identity_fields(
    breakdowns: list[SentenceBreakdown],
) -> list[ValidationIssue]:
    """Surface form and gloss must be non-empty; readings may be omitted (normalized later)."""

    issues: list[ValidationIssue] = []
    for si, bd in enumerate(breakdowns):
        for ei, el in enumerate(bd.elements):
            prefix = f"breakdowns[{si}].elements[{ei}]"
            issues.extend(_element_nonempty_issues(prefix, el))
    return issues


def _normalize_breakdown_readings(breakdowns: list[SentenceBreakdown]) -> list[SentenceBreakdown]:
    """Strip readings; empty / missing readings become \"\" and emit a log warning."""

    normalized: list[SentenceBreakdown] = []
    for si, bd in enumerate(breakdowns):
        new_elements: list[BreakdownElement] = []
        for ei, el in enumerate(bd.elements):
            raw = el.reading if el.reading is not None else ""
            stripped = raw.strip()
            if stripped == "":
                logger.warning(
                    "breakdown element reading missing or whitespace-only: "
                    "breakdowns[%s].elements[%s] text=%r — using empty reading",
                    si,
                    ei,
                    el.text,
                )
            new_elements.append(el.model_copy(update={"reading": stripped}))
        normalized.append(bd.model_copy(update={"elements": new_elements}))
    return normalized


def _element_nonempty_issues(prefix: str, el: BreakdownElement) -> list[ValidationIssue]:
    items: list[ValidationIssue] = []
    if not el.text.strip():
        items.append(ValidationIssue(code="element_text_empty", message=f"{prefix}.text"))
    if not el.meaning.strip():
        items.append(ValidationIssue(code="element_meaning_empty", message=f"{prefix}.meaning"))
    return items


def validate_ha_roles_are_explicit(breakdowns: list[SentenceBreakdown]) -> list[ValidationIssue]:
    """`は` must never be surfaced as generic `other` — always topic vs contrast."""

    issues: list[ValidationIssue] = []
    for si, bd in enumerate(breakdowns):
        for ei, el in enumerate(bd.elements):
            if el.text.strip() == "は" and el.role == "other":
                issues.append(
                    ValidationIssue(
                        code="ha_role_generic_other",
                        message=(
                            f"breakdowns[{si}].elements[{ei}] uses は with role "
                            "`other`; choose topic_marker or contrast_marker."
                        ),
                    )
                )
    return issues


def validate_ni_particle_notes_when_needed(
    breakdowns: list[SentenceBreakdown],
) -> list[ValidationIssue]:
    """`に` must carry a concise function note inside `note`."""

    issues: list[ValidationIssue] = []
    for si, bd in enumerate(breakdowns):
        for ei, el in enumerate(bd.elements):
            if el.text.strip() != "に":
                continue
            if el.note is None or not el.note.strip():
                issues.append(
                    ValidationIssue(
                        code="ni_missing_function_note",
                        message=(
                            f"breakdowns[{si}].elements[{ei}] — specify に function "
                            "(location / direction / indirect_object / time / agent) "
                            "in `note`."
                        ),
                    )
                )
    return issues


def validate_breakdown_generation(raw: RawOutput) -> ValidationResult:
    """Runs every Phase 1 structural check once raw JSON survives parsing."""

    text = strip_code_fences(raw.raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidationResult(is_valid=False, issues=[issue_json_decode(exc)], breakdowns=None)

    try:
        envelope = AnalyseEnvelope.model_validate(payload)
    except ValidationError as exc:
        return ValidationResult(
            is_valid=False,
            issues=issue_pydantic_validation(exc),
            breakdowns=None,
        )

    issues: list[ValidationIssue] = []
    collectors = (
        validate_breakdown_collections_nonempty,
        validate_original_sentence_non_empty,
        validate_elements_non_empty_identity_fields,
        validate_ha_roles_are_explicit,
        validate_ni_particle_notes_when_needed,
    )
    for collector in collectors:
        issues.extend(collector(envelope.breakdowns))

    is_valid = len(issues) == 0
    breakdowns_out = (
        _normalize_breakdown_readings(envelope.breakdowns) if is_valid else None
    )
    return ValidationResult(
        is_valid=is_valid,
        issues=issues,
        breakdowns=breakdowns_out,
    )


def validate_plain_extract_text(raw: RawOutput) -> ValidationResult:
    """Vision/text extraction replies are plain Unicode — forbid empty payloads."""

    text = strip_code_fences(raw.raw_text).strip()
    if not text:
        return ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(
                    code="extract_empty",
                    message="Extraction produced no usable characters.",
                )
            ],
            breakdowns=None,
        )
    return ValidationResult(is_valid=True, issues=[], breakdowns=None)


def validate_practice_item_ids_unique(items: list[PracticeItem]) -> list[ValidationIssue]:
    seen: set[str] = set()
    issues: list[ValidationIssue] = []
    for item in items:
        if item.itemId in seen:
            issues.append(
                ValidationIssue(
                    code="practice_item_id_duplicate",
                    message=(
                        f"Duplicate itemId {item.itemId!r} — regenerate with fresh unique ids "
                        "per drill."
                    ),
                )
            )
        seen.add(item.itemId)
    return issues


def validate_practice_generation(raw: RawOutput) -> ValidationResult:
    """Structural parse + uniqueness for practice minting responses."""

    text = strip_code_fences(raw.raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidationResult(is_valid=False, issues=[issue_json_decode(exc)])

    try:
        envelope = PracticeGenerateEnvelope.model_validate(payload)
    except ValidationError as exc:
        return ValidationResult(
            is_valid=False,
            issues=issue_pydantic_validation(exc),
        )

    issues = validate_practice_item_ids_unique(envelope.items)
    is_valid = len(issues) == 0
    return ValidationResult(
        is_valid=is_valid,
        issues=issues,
        practice_items=envelope.items if is_valid else None,
    )


def validate_practice_evaluation(raw: RawOutput) -> ValidationResult:
    """Parse envelope wrapping `PracticeResult` for graded submissions."""

    text = strip_code_fences(raw.raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidationResult(is_valid=False, issues=[issue_json_decode(exc)])

    try:
        envelope = PracticeEvaluateEnvelope.model_validate(payload)
    except ValidationError as exc:
        return ValidationResult(
            is_valid=False,
            issues=issue_pydantic_validation(exc),
        )

    return ValidationResult(is_valid=True, issues=[], practice_result=envelope.result)


def _utc_assessment_stamp() -> str:
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return stamp.replace("+00:00", "Z")


def validate_onboarding_assessment(raw: RawOutput) -> ValidationResult:
    """Parse Claude onboarding envelope and stamp deterministic server timestamps."""

    text = strip_code_fences(raw.raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidationResult(is_valid=False, issues=[issue_json_decode(exc)])

    try:
        envelope = OnboardingAssessEnvelope.model_validate(payload)
    except ValidationError as exc:
        return ValidationResult(is_valid=False, issues=issue_pydantic_validation(exc))

    stamped = envelope.model_dump(mode="python", by_alias=True)
    ts = _utc_assessment_stamp()
    stamped["createdAt"] = ts
    stamped["updatedAt"] = ts

    try:
        profile = StudentProfile.model_validate(stamped)
    except ValidationError as exc:
        return ValidationResult(is_valid=False, issues=issue_pydantic_validation(exc))

    return ValidationResult(is_valid=True, issues=[], student_profile=profile)


def validate_srs_compute(raw: RawOutput) -> ValidationResult:
    """Flat JSON `{ suggestedIntervalDays, nextReviewAt, reasoning }` from SRS compute prompts."""

    text = strip_code_fences(raw.raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidationResult(is_valid=False, issues=[issue_json_decode(exc)])

    try:
        parsed = SrsComputeResponse.model_validate(payload)
    except ValidationError as exc:
        return ValidationResult(is_valid=False, issues=issue_pydantic_validation(exc))

    return ValidationResult(is_valid=True, issues=[], srs_compute=parsed)


def validate_explain_generation(raw: RawOutput) -> ValidationResult:
    """Parse `ExplainEnvelope` for element-level tutor copy."""

    text = strip_code_fences(raw.raw_text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidationResult(is_valid=False, issues=[issue_json_decode(exc)])

    try:
        envelope = ExplainEnvelope.model_validate(payload)
    except ValidationError as exc:
        return ValidationResult(
            is_valid=False,
            issues=issue_pydantic_validation(exc),
        )

    return ValidationResult(
        is_valid=True,
        issues=[],
        element_explanation=envelope.explanation,
    )
