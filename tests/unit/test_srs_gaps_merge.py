"""Unit tests for SRS gap JSON merge (PATCH parity with mobile)."""

from yomitoku_api.schemas import BreakdownElement, ElementExplanation, KnowledgeGap, KnowledgeGapPartial


def _sample_gap() -> KnowledgeGap:
    return KnowledgeGap(
        id="aaaaaaaa-bbbb-cccc-dddddddddddd",
        createdAtIso="2026-05-01T00:00:00Z",
        breakdownRouteId="route-1",
        sentenceIndex=0,
        sourceSentence="こんにちは。",
        element=BreakdownElement(
            text="は",
            reading="わ",
            role="topic_marker",
            meaning="topic",
            note=None,
        ),
        explanationSnapshot=ElementExplanation(
            headline="h",
            detail="d",
            commonPitfalls=None,
        ),
    )


def test_merge_gap_applies_srs_fields_into_json_shape() -> None:
    from yomitoku_api.services import srs_gaps as svc

    base = _sample_gap()
    patch = KnowledgeGapPartial(
        nextReviewAt="2026-06-01T00:00:00Z",
        intervalDays=7,
    )
    merged = svc.merge_gap(base, patch)
    assert merged.nextReviewAt == "2026-06-01T00:00:00Z"
    assert merged.intervalDays == 7
    assert merged.sourceSentence == base.sourceSentence
