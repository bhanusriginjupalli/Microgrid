"""Phase 4.5 Explainable MARL (X-MARL) Package."""

from experiments.phase4_5.explainability.decision_explainer import (
    DecisionExplainer,
    verify_checkpoint_metadata,
)
from experiments.phase4_5.explainability.feature_attribution import (
    get_agent_feature_names,
    compute_integrated_gradients,
    compute_perturbation_sensitivity,
    run_controlled_perturbation_validation,
)
from experiments.phase4_5.explainability.counterfactuals import (
    COUNTERFACTUAL_SPECS,
    run_counterfactual_suite,
)

__all__ = [
    "DecisionExplainer",
    "verify_checkpoint_metadata",
    "get_agent_feature_names",
    "compute_integrated_gradients",
    "compute_perturbation_sensitivity",
    "run_controlled_perturbation_validation",
    "COUNTERFACTUAL_SPECS",
    "run_counterfactual_suite",
]
