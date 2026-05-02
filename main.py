"""Shim module so `uvicorn main:app` works once the package is installed."""

from yomitoku_api.main import app

__all__ = ["app"]
