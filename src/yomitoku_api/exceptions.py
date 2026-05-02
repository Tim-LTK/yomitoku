"""Domain-specific exceptions; routers map these to HTTP responses."""


class YomitokuApiError(Exception):
    """Base class for API-layer failures with a stable user-facing message."""


class MissingApiKeyError(YomitokuApiError):
    """Raised when `ANTHROPIC_API_KEY` is unset — configure Railway or local `.env`."""

    def __init__(self) -> None:
        super().__init__(
            "Anthropic API key is not configured. Set ANTHROPIC_API_KEY in the server "
            "environment and restart."
        )


class PromptNotFoundError(YomitokuApiError):
    """A versioned `.txt` prompt file is missing from `src/prompts/`."""

    def __init__(self, relative_path: str) -> None:
        super().__init__(
            f"Prompt missing: {relative_path}. "
            "Add the versioned file under src/prompts/."
        )


class GenerationFailedError(YomitokuApiError):
    """The model returned no usable completion or refused the request."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "Anthropic returned an empty completion. Retry with clearer input "
            "or inspect server logs."
        )
