import json

from yomitoku_api.schemas import RawOutput
from yomitoku_api.services import validate as validate_service


def test_validate_onboarding_assessment_round_trip_stamp() -> None:
    envelope = {
        "targetLanguage": "japanese",
        "nativeLanguages": ["Cantonese"],
        "selfReportedLevel": "N5",
        "assessedLevel": "low N5",
        "kanjiAdvantage": True,
        "listeningGap": False,
        "weakAreas": ["particles_wa_ga"],
        "knownGrammar": ["verbs_polite_present"],
        "notes": "Short answers but accurate particles.",
    }
    raw = RawOutput(
        raw_text=json.dumps(envelope),
        model_id="pytest",
        prompt_versions={"onboard_assess": "v1"},
    )
    result = validate_service.validate_onboarding_assessment(raw)
    assert result.is_valid
    assert result.student_profile is not None
    assert result.student_profile.target_language == "japanese"
    assert result.student_profile.created_at.endswith("Z")
    assert result.student_profile.created_at == result.student_profile.updated_at

def test_resolve_request_student_context_falls_back() -> None:
    from yomitoku_api.services.prompts import resolve_request_student_context

    assert "No profile available" in resolve_request_student_context(None)
    assert resolve_request_student_context("  JLPT traveller  ").startswith("JLPT")
