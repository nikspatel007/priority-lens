"""Tests for EmailAction dataclass and utilities."""

import pytest
from datetime import datetime

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from email_action import (
    EmailAction,
    ActionProbabilities,
    ActionTypeIndex,
    ResponseTimeIndex,
    TaskPriorityIndex,
    ACTION_TYPE_TO_INDEX,
    INDEX_TO_ACTION_TYPE,
    LABEL_TO_ACTION_TYPE,
    compute_action_similarity,
    is_similar_action,
    response_time_to_hours,
    urgency_from_response_time,
)


class TestEmailAction:
    """Test EmailAction dataclass."""

    def test_create_basic_action(self):
        """Test creating a basic EmailAction."""
        action = EmailAction(action_type='reply_now', priority=0.8)
        assert action.action_type == 'reply_now'
        assert action.priority == 0.8
        assert action.suggested_response_time == 'same_day'
        assert action.task_priority == 'medium'
        assert action.confidence == 1.0

    def test_create_full_action(self):
        """Test creating EmailAction with all fields."""
        deadline = datetime(2026, 1, 15, 17, 0)
        action = EmailAction(
            action_type='create_task',
            priority=0.9,
            suggested_response_time='next_day',
            task_priority='high',
            task_deadline=deadline,
            task_assignee='bob@example.com',
            confidence=0.85,
        )
        assert action.action_type == 'create_task'
        assert action.priority == 0.9
        assert action.suggested_response_time == 'next_day'
        assert action.task_priority == 'high'
        assert action.task_deadline == deadline
        assert action.task_assignee == 'bob@example.com'
        assert action.confidence == 0.85

    def test_priority_validation(self):
        """Test that priority must be between 0 and 1."""
        with pytest.raises(ValueError):
            EmailAction(action_type='archive', priority=1.5)
        with pytest.raises(ValueError):
            EmailAction(action_type='archive', priority=-0.1)

    def test_confidence_validation(self):
        """Test that confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            EmailAction(action_type='archive', confidence=1.5)

    def test_action_type_index(self):
        """Test action type to index conversion."""
        action = EmailAction(action_type='forward')
        assert action.action_type_index == ActionTypeIndex.FORWARD
        assert action.action_type_index == 2

    def test_response_time_index(self):
        """Test response time to index conversion."""
        action = EmailAction(action_type='reply_now', suggested_response_time='immediate')
        assert action.response_time_index == ResponseTimeIndex.IMMEDIATE
        assert action.response_time_index == 0

    def test_requires_response(self):
        """Test requires_response property."""
        assert EmailAction(action_type='reply_now').requires_response
        assert EmailAction(action_type='reply_later').requires_response
        assert not EmailAction(action_type='archive').requires_response
        assert not EmailAction(action_type='forward').requires_response

    def test_requires_immediate_attention(self):
        """Test requires_immediate_attention property."""
        assert EmailAction(action_type='reply_now').requires_immediate_attention
        assert EmailAction(
            action_type='create_task', task_priority='high'
        ).requires_immediate_attention
        assert EmailAction(
            action_type='reply_later', suggested_response_time='immediate'
        ).requires_immediate_attention
        assert not EmailAction(action_type='archive').requires_immediate_attention

    def test_can_be_automated(self):
        """Test can_be_automated property."""
        assert EmailAction(action_type='archive', confidence=0.95).can_be_automated
        assert EmailAction(action_type='delete', confidence=0.95).can_be_automated
        assert not EmailAction(action_type='archive', confidence=0.8).can_be_automated
        assert not EmailAction(action_type='reply_now', confidence=0.95).can_be_automated

    def test_to_dict(self):
        """Test serialization to dictionary."""
        deadline = datetime(2026, 1, 15, 17, 0)
        action = EmailAction(
            action_type='create_task',
            priority=0.9,
            task_deadline=deadline,
        )
        d = action.to_dict()
        assert d['action_type'] == 'create_task'
        assert d['priority'] == 0.9
        assert d['task_deadline'] == '2026-01-15T17:00:00'

    def test_to_dict_no_deadline(self):
        """Test to_dict with no deadline."""
        action = EmailAction(action_type='archive')
        d = action.to_dict()
        assert d['task_deadline'] is None


class TestEmailActionVector:
    """Test vector conversion for EmailAction."""

    def test_to_action_vector(self):
        """Test conversion to numpy array."""
        np = pytest.importorskip('numpy')

        action = EmailAction(
            action_type='reply_now',
            priority=0.8,
            suggested_response_time='same_day',
            confidence=0.9,
        )
        vector = action.to_action_vector()

        assert vector.shape == (9,)
        assert vector[0] == ActionTypeIndex.REPLY_NOW
        assert vector[1] == 0.8  # priority
        assert vector[2] == ResponseTimeIndex.SAME_DAY
        assert vector[6] == 0.9  # confidence

    def test_from_action_vector(self):
        """Test creation from numpy array."""
        np = pytest.importorskip('numpy')

        vector = np.array([
            0.0,  # REPLY_NOW
            0.75,  # priority
            1.0,  # SAME_DAY
            0.0,  # HIGH
            0.0, 0.0,
            0.85,  # confidence
            0.0, 0.0,
        ], dtype=np.float32)

        action = EmailAction.from_action_vector(vector)
        assert action.action_type == 'reply_now'
        assert action.priority == 0.75
        assert action.suggested_response_time == 'same_day'
        assert action.confidence == pytest.approx(0.85)

    def test_vector_roundtrip(self):
        """Test that vector conversion is reversible."""
        np = pytest.importorskip('numpy')

        original = EmailAction(
            action_type='forward',
            priority=0.6,
            suggested_response_time='next_day',
            task_priority='low',
            confidence=0.7,
        )

        vector = original.to_action_vector()
        restored = EmailAction.from_action_vector(vector)

        assert restored.action_type == original.action_type
        assert abs(restored.priority - original.priority) < 0.001
        assert restored.suggested_response_time == original.suggested_response_time
        assert abs(restored.confidence - original.confidence) < 0.001


class TestEmailActionFactory:
    """Test factory methods for EmailAction."""

    def test_from_training_label_replied(self):
        """Test creating action from REPLIED label."""
        action = EmailAction.from_training_label('REPLIED')
        assert action.action_type == 'reply_now'
        assert action.priority == 0.8

    def test_from_training_label_archived(self):
        """Test creating action from ARCHIVED label."""
        action = EmailAction.from_training_label('ARCHIVED')
        assert action.action_type == 'archive'
        assert action.priority == 0.3

    def test_from_training_label_deleted(self):
        """Test creating action from DELETED label."""
        action = EmailAction.from_training_label('DELETED')
        assert action.action_type == 'delete'
        assert action.priority == 0.1

    def test_from_training_label_forwarded(self):
        """Test creating action from FORWARDED label."""
        action = EmailAction.from_training_label('FORWARDED')
        assert action.action_type == 'forward'

    def test_from_training_label_with_priority(self):
        """Test overriding default priority."""
        action = EmailAction.from_training_label('ARCHIVED', priority=0.9)
        assert action.priority == 0.9

    def test_from_training_label_with_response_hours(self):
        """Test inferring response time from hours."""
        # Immediate (< 1 hour)
        action = EmailAction.from_training_label('REPLIED', response_hours=0.5)
        assert action.suggested_response_time == 'immediate'

        # Same day (< 8 hours)
        action = EmailAction.from_training_label('REPLIED', response_hours=4.0)
        assert action.suggested_response_time == 'same_day'

        # Next day (< 24 hours)
        action = EmailAction.from_training_label('REPLIED', response_hours=20.0)
        assert action.suggested_response_time == 'next_day'

        # This week
        action = EmailAction.from_training_label('REPLIED', response_hours=100.0)
        assert action.suggested_response_time == 'this_week'

    def test_from_policy_output(self):
        """Test creating action from neural network output."""
        np = pytest.importorskip('numpy')

        # Mock policy output - highest logit for archive
        action_logits = np.array([0.1, 0.2, 0.1, 2.0, 0.1, 0.1])  # Archive has highest
        timing_logits = np.array([0.1, 0.1, 1.5, 0.1, 0.1])  # next_day has highest
        priority = 0.4

        action = EmailAction.from_policy_output(action_logits, priority, timing_logits)
        assert action.action_type == 'archive'
        assert action.priority == 0.4
        assert action.suggested_response_time == 'next_day'
        assert 0 < action.confidence <= 1.0


class TestActionProbabilities:
    """Test ActionProbabilities class."""

    def test_sample_action(self):
        """Test sampling action from probability distribution."""
        np = pytest.importorskip('numpy')

        probs = ActionProbabilities(
            action_probs={
                'reply_now': 0.1,
                'archive': 0.8,
                'delete': 0.1,
            },
            timing_probs={
                'same_day': 0.9,
                'this_week': 0.1,
            },
            priority_mean=0.5,
            priority_std=0.1,
        )

        rng = np.random.default_rng(42)
        action = probs.sample_action(rng)

        assert action.action_type in probs.action_probs
        assert action.suggested_response_time in probs.timing_probs
        assert 0.0 <= action.priority <= 1.0

    def test_entropy(self):
        """Test entropy calculation."""
        np = pytest.importorskip('numpy')

        # Uniform distribution has max entropy
        uniform = ActionProbabilities(
            action_probs={
                'reply_now': 0.25,
                'archive': 0.25,
                'delete': 0.25,
                'forward': 0.25,
            }
        )

        # Concentrated distribution has low entropy
        concentrated = ActionProbabilities(
            action_probs={
                'archive': 0.97,
                'delete': 0.01,
                'forward': 0.01,
                'reply_now': 0.01,
            }
        )

        assert uniform.entropy() > concentrated.entropy()


class TestActionSimilarity:
    """Test action similarity functions."""

    def test_identical_actions(self):
        """Test similarity of identical actions."""
        action1 = EmailAction(action_type='reply_now', priority=0.8)
        action2 = EmailAction(action_type='reply_now', priority=0.8)
        assert compute_action_similarity(action1, action2) == pytest.approx(1.0)

    def test_different_action_types(self):
        """Test similarity of completely different actions."""
        action1 = EmailAction(action_type='reply_now', priority=0.9)
        action2 = EmailAction(action_type='delete', priority=0.1)
        similarity = compute_action_similarity(action1, action2)
        assert similarity < 0.3  # Very different

    def test_similar_action_types(self):
        """Test partial credit for similar actions."""
        action1 = EmailAction(action_type='reply_now', priority=0.7)
        action2 = EmailAction(action_type='reply_later', priority=0.7)
        similarity = compute_action_similarity(action1, action2)
        assert 0.4 < similarity < 0.8  # Partial credit

    def test_is_similar_action(self):
        """Test is_similar_action helper."""
        # Response actions are similar
        assert is_similar_action('reply_now', 'reply_later')
        # Passive actions are similar
        assert is_similar_action('archive', 'delete')
        # Different categories are not similar
        assert not is_similar_action('reply_now', 'delete')


class TestUtilityFunctions:
    """Test utility functions."""

    def test_response_time_to_hours(self):
        """Test response time category to hours conversion."""
        assert response_time_to_hours('immediate') == 0.5
        assert response_time_to_hours('same_day') == 4.0
        assert response_time_to_hours('next_day') == 16.0
        assert response_time_to_hours('this_week') == 72.0
        assert response_time_to_hours('when_possible') == 168.0

    def test_urgency_from_response_time(self):
        """Test urgency score calculation."""
        # Quick response = high urgency
        assert urgency_from_response_time(0.5) == 1.0
        # Slower response = lower urgency
        assert urgency_from_response_time(50) == 0.4
        # Very slow = low urgency
        assert urgency_from_response_time(200) == 0.1


class TestMappings:
    """Test constant mappings."""

    def test_action_type_mappings(self):
        """Test bidirectional action type mappings."""
        for action_type, idx in ACTION_TYPE_TO_INDEX.items():
            assert INDEX_TO_ACTION_TYPE[idx] == action_type

    def test_label_to_action_coverage(self):
        """Test that all training labels have mappings."""
        training_labels = ['REPLIED', 'FORWARDED', 'DELETED', 'ARCHIVED', 'KEPT', 'COMPOSED', 'JUNK']
        for label in training_labels:
            assert label in LABEL_TO_ACTION_TYPE
