"""
yomitoku_api/services/jmdict.py — Phase 3.1
JMdict lookup service. Three-layer pattern:
  LAYER 1 (prompt)     — build_fallback_prompt()
  LAYER 2 (generation) — generate_fallback_entry()
  LAYER 3 (validation) — validate_fallback_output()

Primary path: Supabase jmdict_entries lookup (fast, reliable).
Fallback path: AI-estimated JLPT level when term not in DB.
  source="jmdict"      → row came from Supabase
  source="fallback_ai" → row came from AI estimate (may be wrong, never cache)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, ConfigDict, Field
from supabase import Client

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "japanese"

JLPT_LEVELS = frozenset({"N1", "N2", "N3", "N4", "N5"})

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class JmdictEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    text: str
    reading: str
    jlpt_level: str | None = Field(default=None, alias="jlptLevel")
    pitch_accent: str | None = Field(default=None, alias="pitchAccent")
    meanings: list[str]
    parts_of_speech: list[str] = Field(alias="partsOfSpeech")


class JmdictLookupResult(BaseModel):
    entry: JmdictEntry | None
    source: Literal["jmdict", "fallback_ai"]


# ---------------------------------------------------------------------------
# LAYER 1 — Prompt
# ---------------------------------------------------------------------------


class PromptBundle(BaseModel):
    system: str
    user: str
    version: str


def build_fallback_prompt(term: str, version: str = "v1") -> PromptBundle:
    """
    Build the AI fallback prompt for a term not found in jmdict_entries.
    Loads versioned .txt files from prompts/japanese/.
    """
    system_path = PROMPT_DIR / f"jmdict_fallback_{version}_system.txt"
    user_path = PROMPT_DIR / f"jmdict_fallback_{version}_user.txt"

    system = system_path.read_text(encoding="utf-8")
    user = user_path.read_text(encoding="utf-8").replace("{{term}}", term)

    return PromptBundle(system=system, user=user, version=version)


# ---------------------------------------------------------------------------
# LAYER 2 — Generation
# ---------------------------------------------------------------------------


class RawFallbackOutput(BaseModel):
    content: str
    model: str
    prompt_version: str


def generate_fallback_entry(
    prompt: PromptBundle,
    settings,
) -> RawFallbackOutput:
    """
    Call the model to estimate JLPT level for a term not in jmdict_entries.
    Returns raw content — does NOT parse or validate. Does NOT catch errors.
    """
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=prompt.system,
        messages=[{"role": "user", "content": prompt.user}],
    )
    return RawFallbackOutput(
        content=response.content[0].text,
        model=MODEL,
        prompt_version=prompt.version,
    )


# ---------------------------------------------------------------------------
# LAYER 3 — Validation
# ---------------------------------------------------------------------------


class FallbackValidationResult(BaseModel):
    passed: bool
    jlpt_level: str | None = None
    failure_reason: str | None = None


def validate_fallback_output(raw: RawFallbackOutput) -> FallbackValidationResult:
    """
    Parse and validate AI fallback output.
    Expected JSON: {"jlpt_level": "N3"} or {"jlpt_level": null}
    Returns structured pass/fail — never raises.
    """
    try:
        text = (
            raw.content.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return FallbackValidationResult(passed=False, failure_reason=f"JSON parse failed: {exc}")

    if not isinstance(parsed, dict):
        return FallbackValidationResult(passed=False, failure_reason="Expected a JSON object")

    level = parsed.get("jlpt_level")

    if level is None:
        return FallbackValidationResult(passed=True, jlpt_level=None)

    if level not in JLPT_LEVELS:
        return FallbackValidationResult(
            passed=False,
            failure_reason=f"Invalid jlpt_level: {level!r} — must be one of {sorted(JLPT_LEVELS)} or null",
        )

    return FallbackValidationResult(passed=True, jlpt_level=level)


# ---------------------------------------------------------------------------
# Primary lookup — Supabase
# ---------------------------------------------------------------------------


def _row_to_entry(row: dict) -> JmdictEntry:
    return JmdictEntry.model_validate(
        {
            "id": row["id"],
            "text": row["text"],
            "reading": row["reading"],
            "jlptLevel": row.get("jlpt_level"),
            "pitchAccent": row.get("pitch_accent"),
            "meanings": row.get("meanings") or [],
            "partsOfSpeech": row.get("parts_of_speech") or [],
        }
    )


def lookup_in_db(term: str, supabase: Client) -> JmdictEntry | None:
    """
    Exact match on text (kanji form) first, then reading (kana form).
    Returns None if not found.
    """
    result = (
        supabase.table("jmdict_entries")
        .select("id, text, reading, jlpt_level, pitch_accent, meanings, parts_of_speech")
        .eq("text", term)
        .limit(1)
        .execute()
    )
    if result.data:
        return _row_to_entry(result.data[0])

    result = (
        supabase.table("jmdict_entries")
        .select("id, text, reading, jlpt_level, pitch_accent, meanings, parts_of_speech")
        .eq("reading", term)
        .limit(1)
        .execute()
    )
    if result.data:
        return _row_to_entry(result.data[0])

    return None


# ---------------------------------------------------------------------------
# Fallback — AI estimate
# ---------------------------------------------------------------------------


def lookup_via_ai_fallback(term: str, settings) -> JmdictEntry | None:
    """
    When a term is not in jmdict_entries, ask the model for a best-effort
    JLPT level estimate. Returns a minimal JmdictEntry or None.
    Rule: never cache fallback_ai results — they may be wrong.
    """
    try:
        prompt = build_fallback_prompt(term)
        raw = generate_fallback_entry(prompt, settings)
        validation = validate_fallback_output(raw)
    except Exception as exc:
        log.warning("AI fallback failed for %r: %s", term, exc)
        return None

    if not validation.passed:
        log.warning("AI fallback validation failed for %r: %s", term, validation.failure_reason)
        return None

    return JmdictEntry.model_validate(
        {
            "id": f"ai:{term}",
            "text": term,
            "reading": "",
            "jlptLevel": validation.jlpt_level,
            "pitchAccent": None,
            "meanings": [],
            "partsOfSpeech": [],
        }
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def lookup(term: str, supabase: Client, settings) -> JmdictLookupResult:
    """
    DB hit  → source="jmdict"
    DB miss → AI fallback → source="fallback_ai"
    AI miss → entry=None, source="fallback_ai"
    """
    entry = lookup_in_db(term, supabase)
    if entry is not None:
        return JmdictLookupResult(entry=entry, source="jmdict")

    log.info("jmdict: %r not in DB, trying AI fallback", term)
    fallback_entry = lookup_via_ai_fallback(term, settings)
    return JmdictLookupResult(entry=fallback_entry, source="fallback_ai")
