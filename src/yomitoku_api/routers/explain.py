"""HTTP router — targeted element tutoring separate from analyse/extract/practice."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from yomitoku_api.config import Settings
from yomitoku_api.deps import get_settings_cached
from yomitoku_api.schemas import ExplainRequest, ExplainResponse
from yomitoku_api.services import explain as explain_gen
from yomitoku_api.services import prompts as prompt_service
from yomitoku_api.services import validate as validate_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explain"])
SettingsDep = Depends(get_settings_cached)


@router.post(
    "",
    response_model=ExplainResponse,
    summary="Explain one Breakdown element grounded on the source sentence",
)
def post_explain(body: ExplainRequest, settings: Settings = SettingsDep) -> ExplainResponse:
    bundle = prompt_service.build_explain_element_bundle(
        settings,
        element=body.breakdown_element,
        source_sentence=body.source_sentence.strip(),
    )
    raw = explain_gen.generate_element_explanation(settings, bundle)
    validation = validate_service.validate_explain_generation(raw)
    if not validation.is_valid or validation.element_explanation is None:
        logger.info(
            "validation.failed.explain_element",
            extra={"issue_count": len(validation.issues)},
        )
        raise HTTPException(
            status_code=422,
            detail=[issue.model_dump() for issue in validation.issues],
        )
    return ExplainResponse(explanation=validation.element_explanation)
