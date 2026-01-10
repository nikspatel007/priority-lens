"""CORS middleware configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from fastapi import FastAPI

    from priority_lens.api.config import APIConfig


def setup_cors(app: FastAPI, config: APIConfig) -> None:
    """Configure CORS middleware for the application.

    Args:
        app: FastAPI application instance.
        config: API configuration with CORS settings.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Correlation-ID"],
    )
