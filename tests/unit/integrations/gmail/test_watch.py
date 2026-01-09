"""Tests for GmailClient watch methods."""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from rl_emails.integrations.gmail.client import GmailApiError, GmailClient


class TestGmailClientWatch:
    """Tests for GmailClient.watch method."""

    @pytest.fixture
    def mock_response(self) -> mock.MagicMock:
        """Create mock HTTP response."""
        response = mock.MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.content = b'{"historyId": "12345", "expiration": "1609459200000"}'
        response.json.return_value = {"historyId": "12345", "expiration": "1609459200000"}
        return response

    @pytest.mark.asyncio
    async def test_watch_basic(self, mock_response: mock.MagicMock) -> None:
        """Test basic watch setup."""
        with mock.patch("httpx.AsyncClient.request", return_value=mock_response):
            async with GmailClient("test_token") as client:
                history_id, expiration = await client.watch(topic_name="projects/test/topics/gmail")

        assert history_id == "12345"
        assert expiration == 1609459200000

    @pytest.mark.asyncio
    async def test_watch_with_labels(self, mock_response: mock.MagicMock) -> None:
        """Test watch with custom labels."""
        with mock.patch("httpx.AsyncClient.request", return_value=mock_response) as mock_req:
            async with GmailClient("test_token") as client:
                await client.watch(
                    topic_name="projects/test/topics/gmail",
                    label_ids=["INBOX", "IMPORTANT"],
                )

            # Check the request body contains the labels
            call_args = mock_req.call_args
            assert call_args.kwargs["json"]["labelIds"] == ["INBOX", "IMPORTANT"]

    @pytest.mark.asyncio
    async def test_watch_default_label(self, mock_response: mock.MagicMock) -> None:
        """Test watch uses INBOX by default."""
        with mock.patch("httpx.AsyncClient.request", return_value=mock_response) as mock_req:
            async with GmailClient("test_token") as client:
                await client.watch(topic_name="projects/test/topics/gmail")

            call_args = mock_req.call_args
            assert call_args.kwargs["json"]["labelIds"] == ["INBOX"]

    @pytest.mark.asyncio
    async def test_watch_label_filter_action(self, mock_response: mock.MagicMock) -> None:
        """Test watch with label filter action."""
        with mock.patch("httpx.AsyncClient.request", return_value=mock_response) as mock_req:
            async with GmailClient("test_token") as client:
                await client.watch(
                    topic_name="projects/test/topics/gmail",
                    label_filter_action="exclude",
                )

            call_args = mock_req.call_args
            assert call_args.kwargs["json"]["labelFilterAction"] == "exclude"

    @pytest.mark.asyncio
    async def test_watch_api_error(self) -> None:
        """Test watch with API error."""
        error_response = mock.MagicMock(spec=httpx.Response)
        error_response.status_code = 403
        error_response.content = b'{"error": {"message": "Permission denied"}}'
        error_response.text = "Permission denied"
        error_response.json.return_value = {"error": {"message": "Permission denied"}}

        with mock.patch("httpx.AsyncClient.request", return_value=error_response):
            async with GmailClient("test_token") as client:
                with pytest.raises(GmailApiError) as exc_info:
                    await client.watch(topic_name="projects/test/topics/gmail")

        assert exc_info.value.status_code == 403


class TestGmailClientStopWatch:
    """Tests for GmailClient.stop_watch method."""

    @pytest.mark.asyncio
    async def test_stop_watch_success(self) -> None:
        """Test successful watch stop."""
        response = mock.MagicMock(spec=httpx.Response)
        response.status_code = 200
        response.content = b""  # stop returns empty body

        with mock.patch("httpx.AsyncClient.request", return_value=response) as mock_req:
            async with GmailClient("test_token") as client:
                await client.stop_watch()

            # Verify POST request was made
            call_args = mock_req.call_args
            assert call_args.args[0] == "POST"
            assert "stop" in call_args.args[1]

    @pytest.mark.asyncio
    async def test_stop_watch_api_error(self) -> None:
        """Test stop_watch with API error."""
        error_response = mock.MagicMock(spec=httpx.Response)
        error_response.status_code = 404
        error_response.content = b'{"error": {"message": "No watch found"}}'
        error_response.text = "No watch found"
        error_response.json.return_value = {"error": {"message": "No watch found"}}

        with mock.patch("httpx.AsyncClient.request", return_value=error_response):
            async with GmailClient("test_token") as client:
                with pytest.raises(GmailApiError) as exc_info:
                    await client.stop_watch()

        assert exc_info.value.status_code == 404
