"""SDUI schema definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Types of actions that can be triggered by UI interactions."""

    NAVIGATE = "navigate"
    API_CALL = "api_call"
    DISMISS = "dismiss"
    SNOOZE = "snooze"
    COMPLETE = "complete"
    REPLY = "reply"
    ARCHIVE = "archive"
    DELETE = "delete"


class ActionProps(BaseModel):
    """Properties for an action that can be triggered by the UI."""

    id: str = Field(..., description="Unique identifier for the action")
    type: ActionType = Field(..., description="Type of action")
    label: str = Field(..., description="Display label for the action")
    endpoint: str | None = Field(None, description="API endpoint for api_call type")
    method: str = Field("POST", description="HTTP method for api_call")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters for the action")
    confirm: str | None = Field(None, description="Confirmation message before executing")
    icon: str | None = Field(None, description="Icon name for the action")


class GridProps(BaseModel):
    """Grid layout properties."""

    columns: int = Field(1, ge=1, le=12, description="Number of grid columns")
    rows: int | None = Field(None, ge=1, description="Number of grid rows")
    gap: int = Field(8, ge=0, description="Gap between grid items in pixels")
    areas: list[str] | None = Field(None, description="CSS grid-template-areas value")


class LayoutProps(BaseModel):
    """Layout properties for UI blocks."""

    grid: GridProps | None = Field(None, description="Grid layout configuration")
    grid_area: str | None = Field(None, alias="gridArea", description="CSS grid-area value")
    padding: int | list[int] | None = Field(
        None, description="Padding in pixels (single or [top, right, bottom, left])"
    )
    margin: int | list[int] | None = Field(
        None, description="Margin in pixels (single or [top, right, bottom, left])"
    )
    flex: str | None = Field(None, description="CSS flex value")
    align_items: str | None = Field(None, alias="alignItems", description="CSS align-items value")
    justify_content: str | None = Field(
        None, alias="justifyContent", description="CSS justify-content value"
    )
    width: str | int | None = Field(None, description="Width (px or CSS value)")
    height: str | int | None = Field(None, description="Height (px or CSS value)")
    max_width: str | int | None = Field(None, alias="maxWidth", description="Max width")
    background: str | None = Field(None, description="Background color or CSS value")
    border_radius: int | None = Field(
        None, alias="borderRadius", description="Border radius in pixels"
    )

    model_config = {"populate_by_name": True}


class UIBlock(BaseModel):
    """Server-driven UI block definition."""

    id: str = Field(..., description="Unique identifier for the block")
    type: str = Field(..., description="Component type (e.g., 'card', 'list', 'text')")
    props: dict[str, Any] = Field(default_factory=dict, description="Component-specific properties")
    layout: LayoutProps | None = Field(None, description="Layout configuration")
    children: list[UIBlock] = Field(default_factory=list, description="Nested child blocks")
    actions: list[ActionProps] = Field(
        default_factory=list, description="Available actions for this block"
    )

    model_config = {"populate_by_name": True}
