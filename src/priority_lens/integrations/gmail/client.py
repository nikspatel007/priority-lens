"""Gmail API client for email operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from priority_lens.integrations.gmail.models import GmailMessageRef
from priority_lens.integrations.gmail.rate_limiter import RateLimiter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class GmailApiError(Exception):
    """Exception raised for Gmail API errors.

    Attributes:
        status_code: HTTP status code from the API.
        error_code: Error code from Gmail API response.
        message: Human-readable error message.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


class GmailClient:
    """Client for Gmail API operations.

    This client handles communication with the Gmail API, including
    authentication, rate limiting, and error handling.

    Typical usage:
        client = GmailClient(access_token="...", rate_limiter=RateLimiter())

        # List messages
        async for ref in client.list_messages():
            print(ref.id)

        # Get a single message
        raw = await client.get_message("msg123")

        # Batch get messages
        refs = [GmailMessageRef(id="msg1", thread_id="t1"), ...]
        results = await client.batch_get_messages(refs)

    Attributes:
        BASE_URL: Gmail API base URL.
    """

    BASE_URL = "https://gmail.googleapis.com/gmail/v1"

    def __init__(
        self,
        access_token: str,
        rate_limiter: RateLimiter | None = None,
        user_id: str = "me",
    ) -> None:
        """Initialize Gmail client.

        Args:
            access_token: Valid OAuth2 access token for Gmail API.
            rate_limiter: Optional rate limiter for API throttling.
            user_id: Gmail user ID (default: "me" for authenticated user).
        """
        self.access_token = access_token
        self.rate_limiter = rate_limiter or RateLimiter()
        self.user_id = user_id
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30.0,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> GmailClient:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context manager."""
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, str] | None = None,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Make an API request with rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: API path relative to base URL.
            params: Query parameters.
            json_body: JSON body for POST/PUT requests.

        Returns:
            JSON response as dict.

        Raises:
            GmailApiError: If the API returns an error.
        """
        await self.rate_limiter.acquire()

        url = f"{self.BASE_URL}/users/{self.user_id}/{path}"
        response = await self._client.request(method, url, params=params, json=json_body)

        if response.status_code >= 400:
            error_data: dict[str, object] = response.json() if response.content else {}
            error_info = error_data.get("error", {})
            if isinstance(error_info, dict):
                raise GmailApiError(
                    message=str(error_info.get("message", response.text)),
                    status_code=response.status_code,
                    error_code=str(error_info.get("code", "")),
                )
            raise GmailApiError(
                message=response.text,
                status_code=response.status_code,
            )

        # Some endpoints return empty response (like stop)
        if not response.content:
            return {}

        result: dict[str, object] = response.json()
        return result

    async def list_messages(
        self,
        max_results: int = 100,
        page_token: str | None = None,
        query: str | None = None,
        label_ids: list[str] | None = None,
    ) -> tuple[list[GmailMessageRef], str | None]:
        """List message IDs with pagination.

        Args:
            max_results: Maximum number of results per page (1-500).
            page_token: Token for fetching next page.
            query: Gmail search query (e.g., "is:unread").
            label_ids: Filter by label IDs.

        Returns:
            Tuple of (list of message refs, next page token or None).

        Raises:
            GmailApiError: If the API returns an error.
        """
        params: dict[str, str] = {"maxResults": str(min(max_results, 500))}

        if page_token:
            params["pageToken"] = page_token
        if query:
            params["q"] = query
        if label_ids:
            params["labelIds"] = ",".join(label_ids)

        result = await self._request("GET", "messages", params)

        messages = []
        raw_messages = result.get("messages", [])
        if isinstance(raw_messages, list):
            for msg in raw_messages:
                if isinstance(msg, dict):
                    msg_id = str(msg.get("id", ""))
                    thread_id = str(msg.get("threadId", ""))
                    if msg_id:
                        messages.append(GmailMessageRef(id=msg_id, thread_id=thread_id))

        next_page = result.get("nextPageToken")
        return messages, str(next_page) if next_page else None

    async def list_all_messages(
        self,
        query: str | None = None,
        label_ids: list[str] | None = None,
        max_messages: int | None = None,
    ) -> AsyncIterator[GmailMessageRef]:
        """Iterate over all messages with automatic pagination.

        This is a convenience method that handles pagination automatically.

        Args:
            query: Gmail search query (e.g., "is:unread").
            label_ids: Filter by label IDs.
            max_messages: Maximum total messages to return (None for all).

        Yields:
            GmailMessageRef for each message.

        Raises:
            GmailApiError: If the API returns an error.
        """
        page_token = None
        count = 0

        while True:
            refs, next_token = await self.list_messages(
                max_results=500,
                page_token=page_token,
                query=query,
                label_ids=label_ids,
            )

            for ref in refs:
                yield ref
                count += 1
                if max_messages and count >= max_messages:
                    return

            if not next_token:
                break
            page_token = next_token

    async def get_message(
        self,
        message_id: str,
        format: str = "full",  # noqa: A002
    ) -> dict[str, object]:
        """Get a single message by ID.

        Args:
            message_id: Gmail message ID.
            format: Response format ("minimal", "full", "raw", "metadata").

        Returns:
            Raw Gmail API message response.

        Raises:
            GmailApiError: If the API returns an error.
        """
        params = {"format": format}
        return await self._request("GET", f"messages/{message_id}", params)

    async def batch_get_messages(
        self,
        message_refs: list[GmailMessageRef],
        format: str = "full",  # noqa: A002
    ) -> list[dict[str, object] | GmailApiError]:
        """Get multiple messages in parallel.

        This method fetches multiple messages concurrently while
        respecting rate limits.

        Args:
            message_refs: List of message references to fetch.
            format: Response format ("minimal", "full", "raw", "metadata").

        Returns:
            List of results in same order as input.
            Each item is either the raw message dict or a GmailApiError.
        """
        import asyncio

        async def fetch_one(ref: GmailMessageRef) -> dict[str, object] | GmailApiError:
            try:
                return await self.get_message(ref.id, format=format)
            except GmailApiError as e:
                return e

        results = await asyncio.gather(*[fetch_one(ref) for ref in message_refs])
        return list(results)

    async def get_profile(self) -> dict[str, object]:
        """Get the authenticated user's Gmail profile.

        Returns:
            User profile including email address and messages total.

        Raises:
            GmailApiError: If the API returns an error.
        """
        return await self._request("GET", "profile")

    async def get_history(
        self,
        start_history_id: str,
        max_results: int = 100,
        page_token: str | None = None,
        label_ids: list[str] | None = None,
    ) -> tuple[list[dict[str, object]], str | None, str | None]:
        """Get message history changes since a history ID.

        This is used for incremental sync to get changes since last sync.

        Args:
            start_history_id: History ID to start from.
            max_results: Maximum number of history records per page.
            page_token: Token for fetching next page.
            label_ids: Filter by label IDs.

        Returns:
            Tuple of (list of history records, next page token, latest history ID).

        Raises:
            GmailApiError: If the API returns an error.
        """
        params: dict[str, str] = {
            "startHistoryId": start_history_id,
            "maxResults": str(min(max_results, 500)),
        }

        if page_token:
            params["pageToken"] = page_token
        if label_ids:
            params["labelIds"] = ",".join(label_ids)

        result = await self._request("GET", "history", params)

        history_records: list[dict[str, object]] = []
        raw_history = result.get("history", [])
        if isinstance(raw_history, list):
            for record in raw_history:
                if isinstance(record, dict):
                    history_records.append(record)

        next_page = result.get("nextPageToken")
        history_id = result.get("historyId")

        return (
            history_records,
            str(next_page) if next_page else None,
            str(history_id) if history_id else None,
        )

    async def watch(
        self,
        topic_name: str,
        label_ids: list[str] | None = None,
        label_filter_action: str = "include",
    ) -> tuple[str, int]:
        """Set up Gmail push notifications via Pub/Sub.

        Creates a watch on the user's mailbox. Gmail will send
        notifications to the specified Pub/Sub topic when changes occur.

        Args:
            topic_name: Full Pub/Sub topic name (projects/{project}/topics/{topic}).
            label_ids: Labels to watch. If None, watches INBOX.
            label_filter_action: How to apply label filter ("include" or "exclude").

        Returns:
            Tuple of (history_id, expiration_timestamp_ms).
            The expiration is typically 7 days from now.

        Raises:
            GmailApiError: If the API returns an error.

        Note:
            The topic must grant publish rights to gmail-api-push@system.gserviceaccount.com.
            Watch must be renewed before expiration (typically every 7 days).
        """
        body: dict[str, object] = {
            "topicName": topic_name,
            "labelFilterAction": label_filter_action,
        }

        if label_ids:
            body["labelIds"] = label_ids
        else:
            body["labelIds"] = ["INBOX"]

        result = await self._request("POST", "watch", json_body=body)

        history_id = str(result.get("historyId", ""))
        expiration_value = result.get("expiration")
        expiration = int(str(expiration_value)) if expiration_value else 0

        return history_id, expiration

    async def stop_watch(self) -> None:
        """Stop receiving push notifications.

        Stops the current watch on the user's mailbox. After calling this,
        no more notifications will be sent to the Pub/Sub topic.

        Raises:
            GmailApiError: If the API returns an error.
        """
        await self._request("POST", "stop")
