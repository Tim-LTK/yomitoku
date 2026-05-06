import json

from yomitoku_api.schemas import RawOutput
from yomitoku_api.services import validate as validate_service


def _raw_with_json(
    payload: object,
    *,
    prompt_versions: dict[str, str] | None = None,
) -> RawOutput:
    return RawOutput(
        raw_text=json.dumps(payload, ensure_ascii=False),
        model_id="pytest",
        prompt_versions=prompt_versions or {"breakdown_analysis": "v3"},
    )


def test_validate_breakdown_accepts_minimal_sentence() -> None:
    payload = {
        "breakdowns": [
            {
                "original": "私は行きます。",
                "elements": [
                    {
                        "text": "私",
                        "reading": "わたし",
                        "role": "noun",
                        "meaning": "I",
                        "note": None,
                    },
                    {
                        "text": "は",
                        "reading": "は",
                        "role": "topic_marker",
                        "meaning": "topic marker",
                        "note": None,
                    },
                    {
                        "text": "行きます",
                        "reading": "いきます",
                        "role": "verb_base",
                        "meaning": "go / move",
                        "note": None,
                    },
                ],
                "grammarNotes": [],
                "nuanceNote": "neutral polite",
                "difficulty": "N5",
            }
        ]
    }
    result = validate_service.validate_breakdown_generation(_raw_with_json(payload))
    assert result.is_valid
    assert result.breakdowns is not None
    assert len(result.breakdowns) == 1


def test_validate_breakdown_flags_ha_as_other() -> None:
    payload = {
        "breakdowns": [
            {
                "original": "今日は暑い。",
                "elements": [
                    {
                        "text": "今日",
                        "reading": "きょう",
                        "role": "noun",
                        "meaning": "today",
                        "note": None,
                    },
                    {
                        "text": "は",
                        "reading": "は",
                        "role": "other",
                        "meaning": "particle",
                        "note": None,
                    },
                    {
                        "text": "暑い",
                        "reading": "あつい",
                        "role": "adjective_i",
                        "meaning": "hot",
                        "note": None,
                    },
                ],
                "grammarNotes": [],
                "nuanceNote": "",
                "difficulty": "N5",
            }
        ]
    }
    result = validate_service.validate_breakdown_generation(_raw_with_json(payload))
    assert not result.is_valid
    assert any(issue.code == "ha_role_generic_other" for issue in result.issues)


def test_validate_breakdown_requires_ni_note() -> None:
    payload = {
        "breakdowns": [
            {
                "original": "学校に行きます。",
                "elements": [
                    {
                        "text": "学校",
                        "reading": "がっこう",
                        "role": "noun",
                        "meaning": "school",
                        "note": None,
                    },
                    {
                        "text": "に",
                        "reading": "に",
                        "role": "location",
                        "meaning": "to",
                        "note": None,
                    },
                    {
                        "text": "行きます",
                        "reading": "いきます",
                        "role": "verb_base",
                        "meaning": "go",
                        "note": None,
                    },
                ],
                "grammarNotes": [],
                "nuanceNote": "",
                "difficulty": "N5",
            }
        ]
    }
    result = validate_service.validate_breakdown_generation(_raw_with_json(payload))
    assert not result.is_valid
    assert any(issue.code == "ni_missing_function_note" for issue in result.issues)


def test_validate_practice_generation_accepts_minimal_payload() -> None:
    payload = {
        "items": [
            {
                "itemId": "pr_1_a",
                "gapId": "kg_test_gap_01",
                "questionType": "fill_blank",
                "prompt": "Blank the particle.",
                "hint": None,
                "options": None,
            }
        ]
    }
    result = validate_service.validate_practice_generation(_raw_with_json(payload))
    assert result.is_valid
    assert result.practice_items is not None
    assert len(result.practice_items) == 1


def test_validate_practice_generation_flags_duplicate_ids() -> None:
    dup = {
        "itemId": "dup",
        "gapId": "kg_dup_01",
        "questionType": "translate",
        "prompt": "Q1",
        "hint": None,
        "options": None,
    }
    payload = {"items": [dup, dup]}
    result = validate_service.validate_practice_generation(_raw_with_json(payload))
    assert not result.is_valid
    assert any(issue.code == "practice_item_id_duplicate" for issue in result.issues)


def test_validate_session_submit_accepts_batch_envelope() -> None:
    payload = {
        "results": [
            {
                "qualityScore": 4,
                "feedback": "Particles line up cleanly.",
                "errorTags": ["particle"],
            }
        ],
        "tutorNotes": "Particles need one more rehearsal pass tomorrow.",
    }
    result = validate_service.validate_session_submit_generation(_raw_with_json(payload), expected_count=1)
    assert result.is_valid
    assert result.practice_submit_envelope is not None
    assert result.practice_submit_envelope.results[0].qualityScore == 4


def test_validate_session_submit_rejects_length_mismatch() -> None:
    payload = {
        "results": [
            {"qualityScore": 5, "feedback": "OK", "errorTags": []},
        ],
        "tutorNotes": "Good job.",
    }
    result = validate_service.validate_session_submit_generation(_raw_with_json(payload), expected_count=2)
    assert not result.is_valid
    assert any(i.code == "practice_submit_results_length" for i in result.issues)


def test_validate_srs_compute_accepts_flat_payload() -> None:
    payload = {
        "suggestedIntervalDays": 4,
        "nextReviewAt": "2026-06-07T09:30:00Z",
        "reasoning": "Scores clustered at 3–4 without particle regressions → modest stretch.",
    }
    result = validate_service.validate_srs_compute(_raw_with_json(payload))
    assert result.is_valid
    assert result.srs_compute is not None
    assert result.srs_compute.suggestedIntervalDays == 4


def test_validate_explain_accepts_envelope() -> None:
    payload = {
        "explanation": {
            "headline": "に as static location",
            "detail": "Here に marks where the action takes place; pair it with a stative predicate.",
            "commonPitfalls": "Do not confuse with に for time.",
        }
    }
    result = validate_service.validate_explain_generation(_raw_with_json(payload))
    assert result.is_valid
    assert result.element_explanation is not None
    assert result.element_explanation.headline.startswith("に")


def test_validate_scan_accepts_empty_flagged_items() -> None:
    payload = {
        "passage": "今日は暑い。",
        "flaggedItems": [],
        "overallDifficulty": "N5",
        "userLevel": "N5",
    }
    raw = _raw_with_json(payload, prompt_versions={"targeted_scan": "v1"})
    result = validate_service.validate_scan_generation(raw)
    assert result.is_valid
    assert result.scan_result is not None
    assert result.scan_result.flaggedItems == []


def test_validate_scan_rejects_flagged_items_not_list() -> None:
    payload = {
        "passage": "今日は暑い。",
        "flaggedItems": {"not": "a list"},
        "overallDifficulty": "N5",
        "userLevel": "N5",
    }
    raw = _raw_with_json(payload, prompt_versions={"targeted_scan": "v1"})
    result = validate_service.validate_scan_generation(raw)
    assert not result.is_valid
    assert any(i.code == "scan_flagged_items_not_list" for i in result.issues)


def test_validate_ask_accepts_null_suggestion() -> None:
    payload = {"answer": "The sentence describes the weather.", "suggestedFlaggedItem": None}
    raw = _raw_with_json(payload, prompt_versions={"scan_ask": "v2"})
    result = validate_service.validate_ask_generation(raw)
    assert result.is_valid
    assert result.ask_response is not None
    assert result.ask_response.answer.startswith("The sentence")
    assert result.ask_response.suggested_flagged_item is None


def test_validate_ask_drops_invalid_suggestion_keeps_answer() -> None:
    payload = {"answer": "Still valid.", "suggestedFlaggedItem": {"id": "1", "type": "bogus"}}
    raw = _raw_with_json(payload, prompt_versions={"scan_ask": "v2"})
    result = validate_service.validate_ask_generation(raw)
    assert result.is_valid
    assert result.ask_response is not None
    assert result.ask_response.answer == "Still valid."
    assert result.ask_response.suggested_flagged_item is None


def test_validate_ask_keeps_well_formed_suggestion() -> None:
    payload = {
        "answer": "に marks the location.",
        "suggestedFlaggedItem": {
            "id": "ask-suggest-1",
            "text": "に",
            "reading": "に",
            "type": "grammar",
            "jlptLevel": "N5",
            "briefExplanation": "Location particle.",
            "inContext": "机の上に本がある",
            "highlightTier": "consolidate",
        },
    }
    raw = _raw_with_json(payload, prompt_versions={"scan_ask": "v2"})
    result = validate_service.validate_ask_generation(raw)
    assert result.is_valid
    assert result.ask_response is not None
    assert result.ask_response.suggested_flagged_item is not None
    assert result.ask_response.suggested_flagged_item.text == "に"
