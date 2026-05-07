"""Prompt layer — language-namespaced `prompts/<lang>/` files plus `{STUDENT_CONTEXT}` injection."""

import json
from dataclasses import dataclass

from yomitoku_api.config import Settings
from yomitoku_api.exceptions import PromptNotFoundError
from yomitoku_api.schemas import (
    BreakdownElement,
    KnowledgeGap,
    PracticeItem,
    PracticeResult,
    SentenceBreakdown,
)

_LANGUAGE_JAPANESE = "japanese"
_DEFAULT_FALLBACK_CTX = "Student: Japanese language learner. No profile available."


def build_student_context(profile: dict) -> str:
    """Flatten a persisted learner-profile dict into a deterministic multi-line string."""

    if not isinstance(profile, dict):
        raise TypeError("profile must be a mapping")
    chunks: list[str] = []
    for key in sorted(profile.keys(), key=str):
        value = profile[key]
        label = key if isinstance(key, str) else str(key)
        if isinstance(value, (list, tuple)):
            chunks.append(f"{label}: {', '.join(str(item) for item in value)}")
        elif isinstance(value, dict):
            chunks.append(f"{label}: {json.dumps(value, ensure_ascii=False)}")
        elif isinstance(value, bool):
            chunks.append(f"{label}: {'yes' if value else 'no'}")
        else:
            chunks.append(f"{label}: {value}")
    return "\n".join(chunks).strip()


def resolve_request_student_context(explicit_body_field: str | None) -> str:
    """Per-request Claude tone anchor — honours client `studentContext` or benign default."""

    text = (explicit_body_field or "").strip()
    return text if text else _DEFAULT_FALLBACK_CTX


@dataclass(frozen=True)
class PromptBundle:
    """Anthropic system/user pair plus feature → prompt version metadata."""

    system: str
    user: str
    prompt_versions: dict[str, str]


def _prompt_path(fragment: str) -> str:
    return f"{_LANGUAGE_JAPANESE}/{fragment}"


def _read_utf8(settings: Settings, fragment: str) -> str:
    path = settings.prompts_dir / fragment
    if not path.is_file():
        raise PromptNotFoundError(fragment)
    return path.read_text(encoding="utf-8")


def _inject_student_context(fragment: str, *, student_context: str) -> str:
    """Replace template token `{STUDENT_CONTEXT}` wherever prompt authors placed it."""

    return fragment.replace("{STUDENT_CONTEXT}", student_context)


def build_scan_extract_bundle(settings: Settings, *, student_context: str) -> PromptBundle:
    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("scan_extract_v1_system.txt")),
        student_context=student_context,
    )
    user = _inject_student_context(
        _read_utf8(settings, _prompt_path("scan_extract_v1_user.txt")),
        student_context=student_context,
    )
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"scan_extract": "v1"},
    )


def build_breakdown_analysis_bundle(
    settings: Settings,
    japanese_text: str,
    *,
    student_context: str,
) -> PromptBundle:
    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("breakdown_analysis_v3_system.txt")),
        student_context=student_context,
    )
    user_template = _read_utf8(settings, _prompt_path("breakdown_analysis_v3_user.txt"))
    user = _inject_student_context(user_template, student_context=student_context).replace(
        "{JAPANESE_TEXT}", japanese_text.strip()
    )
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"breakdown_analysis": "v3"},
    )


def build_practice_generate_bundle(
    settings: Settings,
    breakdown: SentenceBreakdown,
    *,
    student_context: str,
) -> PromptBundle:
    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("practice_generate_v1_system.txt")),
        student_context=student_context,
    )
    user_template = _read_utf8(settings, _prompt_path("practice_generate_v1_user.txt"))
    sd_json = breakdown.model_dump_json()
    user = _inject_student_context(user_template, student_context=student_context).replace(
        "{SENTENCE_BREAKDOWN_JSON}", sd_json
    )
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"practice_generate": "v1"},
    )


def build_practice_generate_tier2_bundle(
    settings: Settings,
    *,
    gap_specs_json: str,
    student_context: str,
) -> PromptBundle:
    """Batch Tier 2 items for application_mc / nuance_choice."""

    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("practice_generate_v2_system.txt")),
        student_context=student_context,
    )
    template = _read_utf8(settings, _prompt_path("practice_generate_v2_user.txt"))
    user = (
        _inject_student_context(template, student_context=student_context).replace(
            "{GAP_SPECS_JSON}",
            gap_specs_json,
        )
    )
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"practice_generate_tier2": "v2"},
    )


def build_practice_submit_bundle(
    settings: Settings,
    *,
    submission_rows_json: str,
    student_context: str,
) -> PromptBundle:
    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("practice_submit_v1_system.txt")),
        student_context=student_context,
    )
    template = _read_utf8(settings, _prompt_path("practice_submit_v1_user.txt"))
    user = _inject_student_context(template, student_context=student_context).replace(
        "{SUBMISSION_ROWS_JSON}",
        submission_rows_json,
    )
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"practice_submit": "v1"},
    )


def build_explain_element_bundle(
    settings: Settings,
    *,
    element: BreakdownElement,
    source_sentence: str,
    student_context: str,
) -> PromptBundle:
    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("explain_element_v1_system.txt")),
        student_context=student_context,
    )
    template = _read_utf8(settings, _prompt_path("explain_element_v1_user.txt"))
    user = (
        _inject_student_context(template, student_context=student_context)
        .replace("{SOURCE_SENTENCE}", source_sentence.strip())
        .replace("{ELEMENT_JSON}", element.model_dump_json())
    )
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"explain_element": "v1"},
    )


def build_srs_compute_bundle(
    settings: Settings,
    *,
    gap: KnowledgeGap,
    results: list[PracticeResult],
    student_context: str,
) -> PromptBundle:
    """Spacing hint from gap + chronological practice grades."""

    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("srs_compute_v1_system.txt")),
        student_context=student_context,
    )
    template = _read_utf8(settings, _prompt_path("srs_compute_v1_user.txt"))
    history_json = json.dumps(
        [r.model_dump(mode="json") for r in results],
        ensure_ascii=False,
    )
    user = (
        _inject_student_context(template, student_context=student_context)
        .replace("{GAP_JSON}", gap.model_dump_json())
        .replace("{PRACTICE_HISTORY_JSON}", history_json)
    )
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"srs_compute": "v1"},
    )


def build_targeted_scan_bundle(
    settings: Settings,
    passage: str,
    *,
    student_context: str,
) -> PromptBundle:
    """Phase 1.7 — grammar / vocabulary / expression highlights for a passage."""

    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("scan_v2_system.txt")),
        student_context=student_context,
    )
    user = _inject_student_context(
        _read_utf8(settings, _prompt_path("scan_v2_user.txt")),
        student_context=student_context,
    ).replace("{PASSAGE}", passage.strip())
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"targeted_scan": "v2"},
    )


def build_scan_ask_bundle(
    settings: Settings,
    *,
    passage: str,
    question: str,
    student_context: str,
) -> PromptBundle:
    """Follow-up Q&A grounded on the same passage as the scan."""

    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("ask_v2_system.txt")),
        student_context=student_context,
    )
    user = (
        _inject_student_context(
            _read_utf8(settings, _prompt_path("ask_v2_user.txt")),
            student_context=student_context,
        )
        .replace("{PASSAGE}", passage.strip())
        .replace("{QUESTION}", question.strip())
    )
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"scan_ask": "v2"},
    )


def build_onboard_assess_bundle(
    settings: Settings,
    *,
    native_languages_json: str,
    self_reported_level: str,
    answers_json: str,
    student_context: str,
) -> PromptBundle:
    system = _inject_student_context(
        _read_utf8(settings, _prompt_path("onboard_assess_v1_system.txt")),
        student_context=student_context,
    )
    template = _read_utf8(settings, _prompt_path("onboard_assess_v1_user.txt"))
    user = (
        _inject_student_context(template, student_context=student_context)
        .replace("{NATIVE_LANGS_JSON}", native_languages_json)
        .replace("{SELF_REPORTED_LEVEL}", self_reported_level.strip())
        .replace("{ANSWERS_JSON}", answers_json)
    )
    return PromptBundle(
        system=system,
        user=user,
        prompt_versions={"onboard_assess": "v1"},
    )
