import json

from yomitoku_api.schemas import RawOutput
from yomitoku_api.services import validate as validate_service


def _raw_with_json(payload: object) -> RawOutput:
    return RawOutput(
        raw_text=json.dumps(payload, ensure_ascii=False),
        model_id="pytest",
        prompt_versions={"breakdown_analysis": "v2"},
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
                "practiceType": "cloze_gap",
                "prompt": "Blank the particle.",
                "hint": None,
            }
        ]
    }
    result = validate_service.validate_practice_generation(_raw_with_json(payload))
    assert result.is_valid
    assert result.practice_items is not None
    assert len(result.practice_items) == 1


def test_validate_practice_generation_flags_duplicate_ids() -> None:
    dup = {"itemId": "dup", "practiceType": "short_answer", "prompt": "Q1", "hint": None}
    payload = {"items": [dup, dup]}
    result = validate_service.validate_practice_generation(_raw_with_json(payload))
    assert not result.is_valid
    assert any(issue.code == "practice_item_id_duplicate" for issue in result.issues)


def test_validate_practice_evaluation_accepts_envelope() -> None:
    payload = {
        "result": {
            "qualityScore": 4,
            "feedback": "Solid word order; double-check the particle nuance.",
            "errorTags": ["particle"],
        }
    }
    result = validate_service.validate_practice_evaluation(_raw_with_json(payload))
    assert result.is_valid
    assert result.practice_result is not None
    assert result.practice_result.qualityScore == 4


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
