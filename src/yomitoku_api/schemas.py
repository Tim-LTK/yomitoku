"""Pydantic request / response and domain shapes — Phase 1 breakdown + Phase 2 practice."""

from __future__ import annotations

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

ScanItemType = Literal["grammar", "vocabulary", "expression"]
HighlightTier = Literal["consolidate", "stretch"]


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


class ElementExplanation(BaseModel):
    """AI tutor copy for exactly one surfaced token / span within a sentence."""

    headline: Annotated[str, Field(min_length=1)]
    detail: Annotated[str, Field(min_length=1)]
    commonPitfalls: str | None = None


class ExplainRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    breakdown_element: BreakdownElement = Field(alias="breakdownElement")
    source_sentence: Annotated[str, Field(min_length=1, alias="sourceSentence")]
    student_context: str | None = Field(default=None, alias="studentContext")


class ExplainResponse(BaseModel):
    explanation: ElementExplanation


class ExplainEnvelope(BaseModel):
    explanation: ElementExplanation


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
    student_context: str | None = Field(default=None, alias="studentContext")


class PracticeGenerateResponse(BaseModel):
    items: Annotated[list[PracticeItem], Field(min_length=1)]


class PracticeEvaluateRequest(BaseModel):
    """Learner submission for the same breakdown + item Claude generated."""

    model_config = ConfigDict(populate_by_name=True)

    sentence_breakdown: SentenceBreakdown = Field(alias="sentenceBreakdown")
    practice_item: PracticeItem = Field(alias="practiceItem")
    user_answer: Annotated[str, Field(min_length=1, alias="userAnswer")]
    student_context: str | None = Field(default=None, alias="studentContext")


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
    student_context: str | None = Field(default=None, alias="studentContext")


class ExtractResponse(BaseModel):
    text: str


class AnalyseRequest(BaseModel):
    text: Annotated[str, Field(min_length=1)]
    student_context: str | None = Field(default=None, alias="studentContext")


class AnalyseResponse(BaseModel):
    breakdowns: list[SentenceBreakdown]


class FlaggedItem(BaseModel):
    """Single highlighted learning target from a targeted reading scan."""

    id: str
    text: str
    reading: str
    type: ScanItemType
    jlptLevel: JlptBand
    briefExplanation: str
    inContext: str
    highlightTier: HighlightTier


class ScanResult(BaseModel):
    """Structured scan payload — same shape for model JSON, API body, and validation."""

    passage: Annotated[str, Field(min_length=1)]
    flaggedItems: list[FlaggedItem]
    overallDifficulty: JlptBand
    userLevel: str


class ScanEnvelope(ScanResult):
    """JSON object Claude returns for targeted scan (`POST /scan`)."""


class ScanRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: Annotated[str, Field(min_length=1)]
    student_context: str | None = Field(default=None, alias="studentContext")


class ScanResponse(ScanResult):
    """HTTP 200 body for `POST /scan`."""


class AskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: Annotated[str, Field(min_length=1)]
    passage: Annotated[str, Field(min_length=1)]
    student_context: str | None = Field(default=None, alias="studentContext")


class AskResponse(BaseModel):
    answer: str


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


class KnowledgeGap(BaseModel):
    """Learner-flagged weak spot — aligns with RN `KnowledgeGap`."""

    model_config = ConfigDict(populate_by_name=True)

    id: Annotated[str, Field(min_length=8)]
    createdAtIso: Annotated[str, Field(min_length=1)]
    breakdownRouteId: Annotated[str, Field(min_length=1)]
    sentenceIndex: Annotated[int, Field(ge=0)]
    sourceSentence: Annotated[str, Field(min_length=1)]
    element: BreakdownElement
    explanationSnapshot: ElementExplanation
    nextReviewAt: str | None = None
    intervalDays: Annotated[int | None, Field(default=None, ge=1, le=366)] = None


class KnowledgeGapPartial(BaseModel):
    """Sparse PATCH merge for SRS gap rows."""

    model_config = ConfigDict(populate_by_name=True)

    createdAtIso: str | None = None
    breakdownRouteId: str | None = None
    sentenceIndex: int | None = Field(None, ge=0)
    sourceSentence: str | None = None
    element: BreakdownElement | None = None
    explanationSnapshot: ElementExplanation | None = None
    nextReviewAt: str | None = None
    intervalDays: Annotated[int | None, Field(default=None, ge=1, le=366)] = None


class SrsComputeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    gap: KnowledgeGap
    results: Annotated[list[PracticeResult], Field(min_length=1)]
    student_context: str | None = Field(default=None, alias="studentContext")


class OnboardingAnswers(BaseModel):
    """Five free-text placement answers keyed q1–q5."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    q1: Annotated[str, Field(min_length=1)]
    q2: Annotated[str, Field(min_length=1)]
    q3: Annotated[str, Field(min_length=1)]
    q4: Annotated[str, Field(min_length=1)]
    q5: Annotated[str, Field(min_length=1)]


class OnboardingAssessRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    native_languages: Annotated[list[str], Field(min_length=1, alias="nativeLanguages")]
    self_reported_level: Annotated[str, Field(min_length=1, alias="selfReportedLevel")]
    answers: OnboardingAnswers
    student_context: str | None = Field(default=None, alias="studentContext")


class OnboardingAssessEnvelope(BaseModel):
    """Claude onboarding JSON — timestamps added server-side into `StudentProfile`."""

    model_config = ConfigDict(populate_by_name=True)

    target_language: str = Field(default="japanese", alias="targetLanguage")
    native_languages: Annotated[list[str], Field(min_length=1, alias="nativeLanguages")]
    self_reported_level: Annotated[str, Field(min_length=1, alias="selfReportedLevel")]
    assessed_level: Annotated[str, Field(min_length=1, alias="assessedLevel")]
    kanji_advantage: bool = Field(alias="kanjiAdvantage")
    listening_gap: bool = Field(alias="listeningGap")
    weak_areas: list[str] = Field(alias="weakAreas")
    known_grammar: list[str] = Field(alias="knownGrammar")
    notes: Annotated[str, Field(min_length=1)]


class StudentProfile(BaseModel):
    """Structured learner fingerprint returned from `/onboard/assess`."""

    model_config = ConfigDict(populate_by_name=True)

    target_language: str = Field(default="japanese", alias="targetLanguage")
    native_languages: Annotated[list[str], Field(min_length=1, alias="nativeLanguages")]
    self_reported_level: Annotated[str, Field(min_length=1, alias="selfReportedLevel")]
    assessed_level: Annotated[str, Field(min_length=1, alias="assessedLevel")]
    kanji_advantage: bool = Field(alias="kanjiAdvantage")
    listening_gap: bool = Field(alias="listeningGap")
    weak_areas: list[str] = Field(alias="weakAreas")
    known_grammar: list[str] = Field(alias="knownGrammar")
    notes: Annotated[str, Field(min_length=1)]
    created_at: Annotated[str, Field(min_length=1, alias="createdAt")]
    updated_at: Annotated[str, Field(min_length=1, alias="updatedAt")]


class SrsComputeResponse(BaseModel):
    """Claude-guided spacing — serialised verbatim to callers."""

    model_config = ConfigDict(populate_by_name=True)

    suggestedIntervalDays: Annotated[int, Field(ge=1, le=366)]
    nextReviewAt: Annotated[str, Field(min_length=1)]
    reasoning: Annotated[str, Field(min_length=1)]


class SrsComputeEnvelope(BaseModel):
    suggestedIntervalDays: Annotated[int, Field(ge=1, le=366)]
    nextReviewAt: Annotated[str, Field(min_length=1)]
    reasoning: Annotated[str, Field(min_length=1)]


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
    element_explanation: ElementExplanation | None = None
    srs_compute: SrsComputeResponse | None = None
    student_profile: StudentProfile | None = None
    scan_result: ScanResult | None = None


class AnalyseEnvelope(BaseModel):
    """JSON envelope expected from breakdown generation."""

    breakdowns: list[SentenceBreakdown]


class PracticeGenerateEnvelope(BaseModel):
    """JSON Claude must emit when minting drills."""

    items: Annotated[list[PracticeItem], Field(min_length=1)]


class PracticeEvaluateEnvelope(BaseModel):
    """JSON Claude returns when grading a submission."""

    result: PracticeResult
