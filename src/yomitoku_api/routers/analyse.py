"""HTTP router for grammar breakdown."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from yomitoku_api.config import Settings
from yomitoku_api.deps import get_settings_cached
from yomitoku_api.schemas import AnalyseRequest, AnalyseResponse
from yomitoku_api.services import analyse as analyse_gen
from yomitoku_api.services import prompts as prompt_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyse", tags=["analyse"])
SettingsDep = Depends(get_settings_cached)


@router.post(
    "",
    response_model=AnalyseResponse,
    summary="Morphological + grammar-role breakdown",
)
def post_analyse(body: AnalyseRequest, settings: Settings = SettingsDep) -> AnalyseResponse:
    student_context = prompt_service.resolve_request_student_context(body.student_context)
    validation = analyse_gen.run_chunked_sentence_breakdown_analysis(
        settings,
        body.text.strip(),
        student_context=student_context,
    )
    if not validation.is_valid or validation.breakdowns is None:
        logger.info(
            "validation.failed",
            extra={"issue_count": len(validation.issues)},
        )
        raise HTTPException(
            status_code=422,
            detail=[issue.model_dump() for issue in validation.issues],
        )
    return AnalyseResponse(breakdowns=validation.breakdowns)
