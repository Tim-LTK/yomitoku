"""FastAPI composition root — routers only orchestrate layers defined in services."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from yomitoku_api.deps import get_settings_cached
from yomitoku_api.exceptions import GenerationFailedError, MissingApiKeyError, PromptNotFoundError
from yomitoku_api.routers import analyse, extract
from yomitoku_api.schemas import HealthResponse, ProblemDetail

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings_cached()
    if not settings.allowed_origins:
        logger.warning(
            "cors.origins_missing",
            extra={
                "detail": (
                    "ALLOWED_ORIGINS is unset or empty — no browser Origin passes CORS checks. "
                    "Native clients without an Origin header may still succeed."
                ),
            },
        )
    yield


def create_application() -> FastAPI:
    settings = get_settings_cached()
    app = FastAPI(
        title="Yomitoku API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(MissingApiKeyError)
    async def handle_missing_api_key(request: Request, exc: MissingApiKeyError) -> JSONResponse:
        _ = request
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ProblemDetail(title="AI backend unavailable", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(PromptNotFoundError)
    async def handle_prompt_missing(request: Request, exc: PromptNotFoundError) -> JSONResponse:
        _ = request
        logger.error("prompt.missing", exc_info=False, extra={"detail": str(exc)})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ProblemDetail(title="Prompt bundle incomplete", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(GenerationFailedError)
    async def handle_generation_failed(
        request: Request,
        exc: GenerationFailedError,
    ) -> JSONResponse:
        _ = request
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ProblemDetail(title="Model generation failed", detail=str(exc)).model_dump(),
        )

    app.include_router(extract.router)
    app.include_router(analyse.router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        return HealthResponse(ok=True)

    return app


app = create_application()
