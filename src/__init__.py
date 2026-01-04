"""Email RL source modules.

This package provides the core components for email prioritization RL:
- features/: Feature extraction (project, topic, task, people, combined)
- policy_network: PyTorch policy network for action selection

Optional modules (require additional dependencies):
- email_action: Action representation (requires numpy)
- email_state: State representation (requires numpy)
- reward: Multi-signal reward function (requires numpy)
"""

# Feature extraction is always available
from .features import (
    CombinedFeatures,
    CombinedFeatureExtractor,
    extract_combined_features,
    FEATURE_DIMS,
)

# Policy network requires torch
try:
    from .policy_network import (
        EmailPolicyNetwork,
        PolicyConfig,
        PolicyOutput,
        ActionSample,
        create_policy_network,
        DuelingPolicyNetwork,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__all__ = [
    # Features
    'CombinedFeatures',
    'CombinedFeatureExtractor',
    'extract_combined_features',
    'FEATURE_DIMS',
]

if HAS_TORCH:
    __all__.extend([
        'EmailPolicyNetwork',
        'PolicyConfig',
        'PolicyOutput',
        'ActionSample',
        'create_policy_network',
        'DuelingPolicyNetwork',
    ])
