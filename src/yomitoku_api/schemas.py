"""Pydantic request / response and domain shapes — Phase 1 breakdown + Phase 2 practice."""

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


PracticeErrorTag = Literal[
    "particle",
    "conjugation",
    "vocabulary",
    "grammar_pattern",
    "register",
    "orthography",
    "listening",
    "other",
]


class PracticeItem(BaseModel):
    """Single AI-generated drill question derived from SentenceBreakdown."""

    itemId: Annotated[str, Field(min_length=1)]
    practiceType: Annotated[str, Field(min_length=1)]
    prompt: Annotated[str, Field(min_length=1)]
    hint: str | None = None


class PracticeResult(BaseModel):
    """Score + tagging per Phase 2 practice engine rules."""

    qualityScore: Annotated[int, Field(ge=0, le=5)]
    feedback: Annotated[str, Field(min_length=1)]
    errorTags: list[PracticeErrorTag]


class PracticeGenerateRequest(BaseModel):
    """Client sends one analysed sentence breakdown to spawn practice drills."""

    model_config = ConfigDict(populate_by_name=True)

    sentence_breakdown: SentenceBreakdown = Field(alias="sentenceBreakdown")


class PracticeGenerateResponse(BaseModel):
    items: Annotated[list[PracticeItem], Field(min_length=1)]


class PracticeEvaluateRequest(BaseModel):
    """Learner submission for the same breakdown + item Claude generated."""

    model_config = ConfigDict(populate_by_name=True)

    sentence_breakdown: SentenceBreakdown = Field(alias="sentenceBreakdown")
    practice_item: PracticeItem = Field(alias="practiceItem")
    user_answer: Annotated[str, Field(min_length=1, alias="userAnswer")]


class PracticeEvaluateResponse(BaseModel):
    result: PracticeResult


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
    practice_items: list[PracticeItem] | None = None
    practice_result: PracticeResult | None = None


class AnalyseEnvelope(BaseModel):
    """JSON envelope expected from breakdown generation."""

    breakdowns: list[SentenceBreakdown]


class PracticeGenerateEnvelope(BaseModel):
    """JSON Claude must emit when minting drills."""

    items: Annotated[list[PracticeItem], Field(min_length=1)]


class PracticeEvaluateEnvelope(BaseModel):
    """JSON Claude returns when grading a submission."""

    result: PracticeResult
