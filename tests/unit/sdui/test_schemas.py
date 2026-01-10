"""Tests for SDUI schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from priority_lens.sdui.schemas import (
    ActionProps,
    ActionType,
    GridProps,
    LayoutProps,
    UIBlock,
)


class TestActionType:
    """Tests for ActionType enum."""

    def test_all_action_types_exist(self) -> None:
        """Test that all expected action types exist."""
        assert ActionType.NAVIGATE == "navigate"
        assert ActionType.API_CALL == "api_call"
        assert ActionType.DISMISS == "dismiss"
        assert ActionType.SNOOZE == "snooze"
        assert ActionType.COMPLETE == "complete"
        assert ActionType.REPLY == "reply"
        assert ActionType.ARCHIVE == "archive"
        assert ActionType.DELETE == "delete"


class TestActionProps:
    """Tests for ActionProps schema."""

    def test_minimal_action(self) -> None:
        """Test creating minimal action props."""
        action = ActionProps(
            id="test-1",
            type=ActionType.NAVIGATE,
            label="Go",
        )

        assert action.id == "test-1"
        assert action.type == ActionType.NAVIGATE
        assert action.label == "Go"
        assert action.endpoint is None
        assert action.method == "POST"
        assert action.params == {}

    def test_full_action(self) -> None:
        """Test creating action with all fields."""
        action = ActionProps(
            id="api-1",
            type=ActionType.API_CALL,
            label="Archive",
            endpoint="/api/emails/1/archive",
            method="POST",
            params={"confirm": True},
            confirm="Are you sure?",
            icon="archive",
        )

        assert action.endpoint == "/api/emails/1/archive"
        assert action.confirm == "Are you sure?"
        assert action.icon == "archive"
        assert action.params == {"confirm": True}

    def test_action_type_from_string(self) -> None:
        """Test that action type can be created from string."""
        action = ActionProps(
            id="test",
            type="navigate",  # type: ignore[arg-type]
            label="Test",
        )
        assert action.type == ActionType.NAVIGATE


class TestGridProps:
    """Tests for GridProps schema."""

    def test_default_grid(self) -> None:
        """Test default grid values."""
        grid = GridProps()

        assert grid.columns == 1
        assert grid.rows is None
        assert grid.gap == 8
        assert grid.areas is None

    def test_custom_grid(self) -> None:
        """Test custom grid configuration."""
        grid = GridProps(
            columns=3,
            rows=2,
            gap=16,
            areas=["header header", "main sidebar"],
        )

        assert grid.columns == 3
        assert grid.rows == 2
        assert grid.gap == 16
        assert grid.areas == ["header header", "main sidebar"]

    def test_columns_validation(self) -> None:
        """Test columns must be between 1 and 12."""
        with pytest.raises(ValidationError):
            GridProps(columns=0)

        with pytest.raises(ValidationError):
            GridProps(columns=13)

    def test_gap_validation(self) -> None:
        """Test gap must be non-negative."""
        with pytest.raises(ValidationError):
            GridProps(gap=-1)


class TestLayoutProps:
    """Tests for LayoutProps schema."""

    def test_empty_layout(self) -> None:
        """Test empty layout props."""
        layout = LayoutProps()

        assert layout.grid is None
        assert layout.padding is None
        assert layout.margin is None

    def test_layout_with_grid(self) -> None:
        """Test layout with grid configuration."""
        layout = LayoutProps(
            grid=GridProps(columns=2),
            padding=16,
        )

        assert layout.grid is not None
        assert layout.grid.columns == 2
        assert layout.padding == 16

    def test_layout_with_padding_array(self) -> None:
        """Test layout with padding as array."""
        layout = LayoutProps(padding=[10, 20, 10, 20])

        assert layout.padding == [10, 20, 10, 20]

    def test_layout_alias_fields(self) -> None:
        """Test that alias fields work."""
        layout = LayoutProps(
            gridArea="main",
            alignItems="center",
            justifyContent="space-between",
            maxWidth="800px",
            borderRadius=8,
        )

        assert layout.grid_area == "main"
        assert layout.align_items == "center"
        assert layout.justify_content == "space-between"
        assert layout.max_width == "800px"
        assert layout.border_radius == 8


class TestUIBlock:
    """Tests for UIBlock schema."""

    def test_minimal_block(self) -> None:
        """Test creating minimal UI block."""
        block = UIBlock(id="block-1", type="card")

        assert block.id == "block-1"
        assert block.type == "card"
        assert block.props == {}
        assert block.layout is None
        assert block.children == []
        assert block.actions == []

    def test_block_with_props(self) -> None:
        """Test block with custom props."""
        block = UIBlock(
            id="email-1",
            type="email_card",
            props={
                "subject": "Test Email",
                "from": "test@example.com",
            },
        )

        assert block.props["subject"] == "Test Email"
        assert block.props["from"] == "test@example.com"

    def test_block_with_layout(self) -> None:
        """Test block with layout configuration."""
        block = UIBlock(
            id="container",
            type="container",
            layout=LayoutProps(padding=16, border_radius=8),
        )

        assert block.layout is not None
        assert block.layout.padding == 16
        assert block.layout.border_radius == 8

    def test_block_with_children(self) -> None:
        """Test block with nested children."""
        parent = UIBlock(
            id="list",
            type="list",
            children=[
                UIBlock(id="item-1", type="list_item"),
                UIBlock(id="item-2", type="list_item"),
            ],
        )

        assert len(parent.children) == 2
        assert parent.children[0].id == "item-1"
        assert parent.children[1].id == "item-2"

    def test_block_with_actions(self) -> None:
        """Test block with actions."""
        block = UIBlock(
            id="task-1",
            type="task_card",
            actions=[
                ActionProps(
                    id="complete-1",
                    type=ActionType.COMPLETE,
                    label="Complete",
                ),
                ActionProps(
                    id="dismiss-1",
                    type=ActionType.DISMISS,
                    label="Dismiss",
                ),
            ],
        )

        assert len(block.actions) == 2
        assert block.actions[0].type == ActionType.COMPLETE
        assert block.actions[1].type == ActionType.DISMISS

    def test_block_serialization(self) -> None:
        """Test that block serializes correctly."""
        block = UIBlock(
            id="test",
            type="card",
            props={"title": "Test"},
            layout=LayoutProps(padding=8),
        )

        data = block.model_dump()

        assert data["id"] == "test"
        assert data["type"] == "card"
        assert data["props"]["title"] == "Test"
        assert data["layout"]["padding"] == 8
