#!/usr/bin/env python3
"""Tests for ensemble module."""

import sys
from pathlib import Path
import tempfile

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ensemble import (
    EnsemblePredictor,
    EnsembleMethod,
    ModelMember,
)
from policy_network import create_policy_network, NUM_ACTION_TYPES
from features.combined import CombinedFeatureExtractor, FEATURE_DIMS


class TestModelMember:
    """Tests for ModelMember dataclass."""

    def test_model_member_creation(self):
        """Test creating a ModelMember."""
        model = create_policy_network(input_dim=69)
        extractor = CombinedFeatureExtractor(include_content=False)

        member = ModelMember(
            name='test_model',
            model=model,
            extractor=extractor,
            input_dim=69,
            weight=1.0,
            validation_accuracy=0.75,
        )

        assert member.name == 'test_model'
        assert member.input_dim == 69
        assert member.weight == 1.0
        assert member.validation_accuracy == 0.75


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor class."""

    @pytest.fixture
    def sample_email(self):
        """Sample email for testing."""
        return {
            'from': 'john@example.com',
            'to': 'jane@example.com',
            'cc': '',
            'subject': 'URGENT: Need your input',
            'body': 'Please review and respond by EOD.',
            'x_from': 'Smith, John',
            'x_to': 'Doe, Jane',
        }

    @pytest.fixture
    def mock_ensemble(self):
        """Create a mock ensemble with random weights."""
        members = []
        for name in ['balanced', 'focal', 'oversampled']:
            model = create_policy_network(input_dim=69)
            extractor = CombinedFeatureExtractor(include_content=False)
            member = ModelMember(
                name=name,
                model=model,
                extractor=extractor,
                input_dim=69,
                validation_accuracy=0.6 + 0.1 * len(members),
            )
            members.append(member)

        return EnsemblePredictor(members=members, device='cpu')

    def test_ensemble_creation(self, mock_ensemble):
        """Test creating an ensemble."""
        assert len(mock_ensemble.members) == 3
        assert mock_ensemble.device == torch.device('cpu')

    def test_weight_computation(self, mock_ensemble):
        """Test that weights are computed from validation accuracy."""
        total_weight = sum(m.weight for m in mock_ensemble.members)
        assert abs(total_weight - 1.0) < 1e-6  # Weights should sum to 1

        # Higher accuracy should have higher weight
        weights = {m.name: m.weight for m in mock_ensemble.members}
        assert weights['oversampled'] > weights['balanced']

    def test_predict_proba_single(self, mock_ensemble, sample_email):
        """Test probability prediction for a single email."""
        probs = mock_ensemble.predict_proba(sample_email)

        assert probs.shape == (1, NUM_ACTION_TYPES)
        assert probs.sum().item() == pytest.approx(1.0, rel=1e-5)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_proba_batch(self, mock_ensemble, sample_email):
        """Test probability prediction for a batch of emails."""
        emails = [sample_email] * 5
        probs = mock_ensemble.predict_proba(emails)

        assert probs.shape == (5, NUM_ACTION_TYPES)
        assert all(probs[i].sum().item() == pytest.approx(1.0, rel=1e-5) for i in range(5))

    def test_predict(self, mock_ensemble, sample_email):
        """Test action prediction."""
        pred = mock_ensemble.predict(sample_email)

        assert pred.shape == (1,)
        assert 0 <= pred.item() < NUM_ACTION_TYPES

    def test_hard_voting(self, mock_ensemble, sample_email):
        """Test hard voting ensemble method."""
        probs = mock_ensemble.predict_proba(sample_email, method=EnsembleMethod.HARD_VOTING)

        assert probs.shape == (1, NUM_ACTION_TYPES)
        # Hard voting produces vote counts normalized to probabilities
        assert probs.sum().item() == pytest.approx(1.0, rel=1e-5)

    def test_soft_voting(self, mock_ensemble, sample_email):
        """Test soft voting ensemble method."""
        probs = mock_ensemble.predict_proba(sample_email, method=EnsembleMethod.SOFT_VOTING)

        assert probs.shape == (1, NUM_ACTION_TYPES)
        assert probs.sum().item() == pytest.approx(1.0, rel=1e-5)

    def test_weighted_avg(self, mock_ensemble, sample_email):
        """Test weighted averaging ensemble method."""
        probs = mock_ensemble.predict_proba(sample_email, method=EnsembleMethod.WEIGHTED_AVG)

        assert probs.shape == (1, NUM_ACTION_TYPES)
        assert probs.sum().item() == pytest.approx(1.0, rel=1e-5)

    def test_get_member_predictions(self, mock_ensemble, sample_email):
        """Test getting individual member predictions."""
        predictions = mock_ensemble.get_member_predictions(sample_email)

        assert len(predictions) == 3
        for name, pred in predictions.items():
            assert pred.shape == (1,)
            assert 0 <= pred.item() < NUM_ACTION_TYPES

    def test_get_agreement_score(self, mock_ensemble, sample_email):
        """Test agreement score computation."""
        agreement = mock_ensemble.get_agreement_score(sample_email)

        assert agreement.shape == (1,)
        assert 0.0 <= agreement.item() <= 1.0

    def test_batch_agreement_score(self, mock_ensemble, sample_email):
        """Test agreement score for batch."""
        emails = [sample_email] * 3
        agreement = mock_ensemble.get_agreement_score(emails)

        assert agreement.shape == (3,)
        assert (agreement >= 0.0).all()
        assert (agreement <= 1.0).all()


class TestEnsembleMethods:
    """Tests for different ensemble combination methods."""

    @pytest.fixture
    def deterministic_ensemble(self):
        """Create ensemble with predictable outputs for testing."""
        torch.manual_seed(42)

        members = []
        for i, name in enumerate(['model_a', 'model_b', 'model_c']):
            model = create_policy_network(input_dim=69)
            extractor = CombinedFeatureExtractor(include_content=False)
            member = ModelMember(
                name=name,
                model=model,
                extractor=extractor,
                input_dim=69,
                validation_accuracy=0.7 + 0.05 * i,
            )
            members.append(member)

        return EnsemblePredictor(members=members, device='cpu')

    def test_methods_produce_valid_output(self, deterministic_ensemble):
        """All methods should produce valid probability distributions."""
        email = {
            'from': 'test@example.com',
            'to': 'user@example.com',
            'subject': 'Test',
            'body': 'Test body',
        }

        for method in [EnsembleMethod.HARD_VOTING, EnsembleMethod.SOFT_VOTING, EnsembleMethod.WEIGHTED_AVG]:
            probs = deterministic_ensemble.predict_proba(email, method=method)

            assert probs.shape == (1, NUM_ACTION_TYPES)
            assert probs.sum().item() == pytest.approx(1.0, rel=1e-5)
            assert (probs >= 0).all()

    def test_different_methods_may_differ(self, deterministic_ensemble):
        """Different methods may produce different predictions."""
        email = {
            'from': 'test@example.com',
            'to': 'user@example.com',
            'subject': 'Test',
            'body': 'Test body',
        }

        results = {}
        for method in [EnsembleMethod.HARD_VOTING, EnsembleMethod.SOFT_VOTING, EnsembleMethod.WEIGHTED_AVG]:
            probs = deterministic_ensemble.predict_proba(email, method=method)
            results[method] = probs

        # Methods can differ (though not guaranteed for any specific input)
        # Just verify they all produce valid outputs
        for method, probs in results.items():
            assert probs.shape == (1, NUM_ACTION_TYPES)


class TestEnsembleCheckpointing:
    """Tests for ensemble save/load functionality."""

    @pytest.fixture
    def mock_ensemble_dir(self):
        """Create a temporary directory with mock checkpoint structure."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create variant directories
            for name in ['balanced', 'focal']:
                variant_dir = tmpdir / name
                variant_dir.mkdir()

                # Create mock config
                config = {
                    'name': name,
                    'description': f'Mock {name} model',
                    'input_dim': 69,
                    'include_content': False,
                    'content_model': 'all-MiniLM-L6-v2',
                    'final_metrics': {
                        'val_accuracy': 0.7 + (0.05 if name == 'focal' else 0),
                    },
                }
                with open(variant_dir / 'config.json', 'w') as f:
                    json.dump(config, f)

                # Create mock checkpoint
                model = create_policy_network(input_dim=69)
                torch.save({
                    'model_state_dict': model.state_dict(),
                }, variant_dir / 'final.pt')

            # Create summary
            summary = {
                'variants': ['balanced', 'focal'],
                'results': {
                    'balanced': {'val_accuracy': 0.7},
                    'focal': {'val_accuracy': 0.75},
                },
            }
            with open(tmpdir / 'ensemble_summary.json', 'w') as f:
                json.dump(summary, f)

            yield tmpdir

    def test_load_from_checkpoint_dir(self, mock_ensemble_dir):
        """Test loading ensemble from checkpoint directory."""
        ensemble = EnsemblePredictor.from_checkpoint_dir(mock_ensemble_dir, device='cpu')

        assert len(ensemble.members) == 2
        member_names = {m.name for m in ensemble.members}
        assert member_names == {'balanced', 'focal'}

    def test_loaded_models_work(self, mock_ensemble_dir):
        """Test that loaded models can make predictions."""
        ensemble = EnsemblePredictor.from_checkpoint_dir(mock_ensemble_dir, device='cpu')

        email = {
            'from': 'test@example.com',
            'to': 'user@example.com',
            'subject': 'Test',
            'body': 'Test body',
        }

        pred = ensemble.predict(email)
        assert pred.shape == (1,)
        assert 0 <= pred.item() < NUM_ACTION_TYPES


class TestStackingMetaClassifier:
    """Tests for stacking ensemble method."""

    @pytest.fixture
    def simple_ensemble(self):
        """Create simple ensemble for stacking tests."""
        members = []
        for name in ['model_a', 'model_b']:
            model = create_policy_network(input_dim=69)
            extractor = CombinedFeatureExtractor(include_content=False)
            member = ModelMember(
                name=name,
                model=model,
                extractor=extractor,
                input_dim=69,
                validation_accuracy=0.7,
            )
            members.append(member)

        return EnsemblePredictor(members=members, device='cpu')

    def test_meta_classifier_not_trained_by_default(self, simple_ensemble):
        """Meta-classifier should not exist before training."""
        assert simple_ensemble._meta_classifier is None

    def test_stacking_requires_trained_meta_classifier(self, simple_ensemble):
        """Stacking should fall back to weighted avg without trained meta-classifier."""
        email = {
            'from': 'test@example.com',
            'to': 'user@example.com',
            'subject': 'Test',
            'body': 'Test body',
        }

        # Without trained meta-classifier, should use weighted avg
        probs = simple_ensemble.predict_proba(email, method=EnsembleMethod.STACKING)
        assert probs.shape == (1, NUM_ACTION_TYPES)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
