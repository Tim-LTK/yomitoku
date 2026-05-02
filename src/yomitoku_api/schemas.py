"""Pydantic request / response and domain shapes — Phase 1 only."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

GrammarRole = Literal[
    "topic_marker",
    "subject_marker",
    "object_marker",
    "location",
    "direction",
    "indirect_object",
    "time",
    "means_method",
    "contrast_marker",
    "verb_base",
    "verb_te_form",
    "verb_ending",
    "noun",
    "adjective_i",
    "adjective_na",
    "adverb",
    "conjunction",
    "sentence_final",
    "grammar_pattern",
    "other",
]

JlptBand = Literal["N5", "N4", "N3", "N2", "N1"]


class BreakdownElement(BaseModel):
    text: str
    reading: str
    role: GrammarRole
    meaning: str
    note: str | None = None


class GrammarNote(BaseModel):
    pattern: str
    explanation: str
    timInContext: str


class SentenceBreakdown(BaseModel):
    original: str
    elements: list[BreakdownElement]
    grammarNotes: list[GrammarNote]
    nuanceNote: str
    difficulty: JlptBand


class ExtractRequest(BaseModel):
    """Image bytes as base64; vision extraction runs in generation layer."""

    model_config = ConfigDict(populate_by_name=True)

    image_base64: Annotated[str, Field(min_length=1, alias="imageBase64")]
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = Field(
        default="image/jpeg",
        alias="mimeType",
    )


class ExtractResponse(BaseModel):
    text: str


class AnalyseRequest(BaseModel):
    text: Annotated[str, Field(min_length=1)]


class AnalyseResponse(BaseModel):
    breakdowns: list[SentenceBreakdown]


class HealthResponse(BaseModel):
    ok: bool
    service: Literal["yomitoku-api"] = "yomitoku-api"


class ProblemDetail(BaseModel):
    """Structured error payloads for stable clients."""

    title: str
    detail: str | None = None


class RawOutput(BaseModel):
    """Generation layer returns raw assistant text plus provenance."""

    raw_text: str
    model_id: str
    prompt_versions: dict[str, str]


class ValidationIssue(BaseModel):
    code: str
    message: str


class ValidationResult(BaseModel):
    """Idempotent aggregation of discrete validators."""

    is_valid: bool
    issues: list[ValidationIssue]
    breakdowns: list[SentenceBreakdown] | None = None


class AnalyseEnvelope(BaseModel):
    """JSON envelope expected from breakdown generation."""

    breakdowns: list[SentenceBreakdown]
