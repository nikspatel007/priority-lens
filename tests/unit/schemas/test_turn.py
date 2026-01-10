"""Unit tests for turn schemas."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from priority_lens.schemas.turn import (
    TextInput,
    TextSubmitPayload,
    TurnClosePayload,
    TurnCreate,
    TurnInputType,
    TurnOpenPayload,
    TurnResponse,
    VoiceInput,
    VoiceTranscriptPayload,
)


class TestTurnInputType:
    """Tests for TurnInputType enum."""

    def test_text_value(self) -> None:
        """Test TEXT enum value."""
        assert TurnInputType.TEXT.value == "text"

    def test_voice_value(self) -> None:
        """Test VOICE enum value."""
        assert TurnInputType.VOICE.value == "voice"


class TestTextInput:
    """Tests for TextInput schema."""

    def test_valid_text_input(self) -> None:
        """Test valid text input."""
        input_data = TextInput(text="Hello, world!")

        assert input_data.type == "text"
        assert input_data.text == "Hello, world!"

    def test_empty_text_raises(self) -> None:
        """Test that empty text raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TextInput(text="")

        assert "text" in str(exc_info.value)

    def test_text_too_long_raises(self) -> None:
        """Test that text over 10000 chars raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TextInput(text="a" * 10001)

        assert "text" in str(exc_info.value)

    def test_text_max_length_valid(self) -> None:
        """Test that 10000 char text is valid."""
        input_data = TextInput(text="a" * 10000)
        assert len(input_data.text) == 10000


class TestVoiceInput:
    """Tests for VoiceInput schema."""

    def test_valid_voice_input(self) -> None:
        """Test valid voice input with all fields."""
        input_data = VoiceInput(
            transcript="Hello, world!",
            confidence=0.95,
            duration_ms=1500,
        )

        assert input_data.type == "voice"
        assert input_data.transcript == "Hello, world!"
        assert input_data.confidence == 0.95
        assert input_data.duration_ms == 1500

    def test_voice_input_defaults(self) -> None:
        """Test voice input with default values."""
        input_data = VoiceInput(transcript="Hello")

        assert input_data.confidence == 1.0
        assert input_data.duration_ms is None

    def test_empty_transcript_raises(self) -> None:
        """Test that empty transcript raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            VoiceInput(transcript="")

        assert "transcript" in str(exc_info.value)

    def test_confidence_out_of_range_raises(self) -> None:
        """Test that confidence outside 0-1 raises validation error."""
        with pytest.raises(ValidationError):
            VoiceInput(transcript="test", confidence=1.5)

        with pytest.raises(ValidationError):
            VoiceInput(transcript="test", confidence=-0.1)

    def test_negative_duration_raises(self) -> None:
        """Test that negative duration raises validation error."""
        with pytest.raises(ValidationError):
            VoiceInput(transcript="test", duration_ms=-100)


class TestTurnCreate:
    """Tests for TurnCreate schema."""

    def test_valid_text_turn(self) -> None:
        """Test creating turn with text input."""
        session_id = uuid4()
        turn = TurnCreate(
            session_id=session_id,
            input=TextInput(text="Hello!"),
        )

        assert turn.session_id == session_id
        assert isinstance(turn.input, TextInput)
        assert turn.input.text == "Hello!"

    def test_valid_voice_turn(self) -> None:
        """Test creating turn with voice input."""
        session_id = uuid4()
        turn = TurnCreate(
            session_id=session_id,
            input=VoiceInput(transcript="Hello there!"),
        )

        assert turn.session_id == session_id
        assert isinstance(turn.input, VoiceInput)
        assert turn.input.transcript == "Hello there!"

    def test_missing_session_id_raises(self) -> None:
        """Test that missing session_id raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TurnCreate(input=TextInput(text="test"))  # type: ignore[call-arg]

        assert "session_id" in str(exc_info.value)

    def test_missing_input_raises(self) -> None:
        """Test that missing input raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            TurnCreate(session_id=uuid4())  # type: ignore[call-arg]

        assert "input" in str(exc_info.value)


class TestTurnResponse:
    """Tests for TurnResponse schema."""

    def test_valid_response(self) -> None:
        """Test valid turn response."""
        correlation_id = uuid4()
        thread_id = uuid4()
        session_id = uuid4()

        response = TurnResponse(
            correlation_id=correlation_id,
            accepted=True,
            thread_id=thread_id,
            session_id=session_id,
            seq=5,
        )

        assert response.correlation_id == correlation_id
        assert response.accepted is True
        assert response.thread_id == thread_id
        assert response.session_id == session_id
        assert response.seq == 5

    def test_accepted_default_true(self) -> None:
        """Test that accepted defaults to True."""
        response = TurnResponse(
            correlation_id=uuid4(),
            thread_id=uuid4(),
            session_id=uuid4(),
            seq=0,
        )

        assert response.accepted is True

    def test_negative_seq_raises(self) -> None:
        """Test that negative seq raises validation error."""
        with pytest.raises(ValidationError):
            TurnResponse(
                correlation_id=uuid4(),
                thread_id=uuid4(),
                session_id=uuid4(),
                seq=-1,
            )


class TestTurnOpenPayload:
    """Tests for TurnOpenPayload schema."""

    def test_valid_payload(self) -> None:
        """Test valid turn open payload."""
        payload = TurnOpenPayload(
            correlation_id="abc-123",
            input_type="text",
        )

        assert payload.correlation_id == "abc-123"
        assert payload.input_type == "text"


class TestTextSubmitPayload:
    """Tests for TextSubmitPayload schema."""

    def test_valid_payload(self) -> None:
        """Test valid text submit payload."""
        payload = TextSubmitPayload(
            correlation_id="abc-123",
            text="Hello, world!",
        )

        assert payload.correlation_id == "abc-123"
        assert payload.text == "Hello, world!"


class TestVoiceTranscriptPayload:
    """Tests for VoiceTranscriptPayload schema."""

    def test_valid_payload(self) -> None:
        """Test valid voice transcript payload."""
        payload = VoiceTranscriptPayload(
            correlation_id="abc-123",
            transcript="Hello there!",
            confidence=0.95,
            duration_ms=1500,
        )

        assert payload.correlation_id == "abc-123"
        assert payload.transcript == "Hello there!"
        assert payload.confidence == 0.95
        assert payload.duration_ms == 1500

    def test_duration_optional(self) -> None:
        """Test that duration_ms is optional."""
        payload = VoiceTranscriptPayload(
            correlation_id="abc-123",
            transcript="test",
            confidence=1.0,
        )

        assert payload.duration_ms is None


class TestTurnClosePayload:
    """Tests for TurnClosePayload schema."""

    def test_valid_payload(self) -> None:
        """Test valid turn close payload."""
        payload = TurnClosePayload(
            correlation_id="abc-123",
            reason="submit",
        )

        assert payload.correlation_id == "abc-123"
        assert payload.reason == "submit"

    def test_reason_default(self) -> None:
        """Test that reason defaults to submit."""
        payload = TurnClosePayload(correlation_id="abc-123")

        assert payload.reason == "submit"
