from __future__ import annotations

from importlib import import_module

from .base import Estimator, UserState

_EXPORTS: dict[str, str] = {
    "Estimator": "base",
    "UserState": "base",
    "GlobalWordPriorEstimator": "baselines",
    "OnlineThresholdWordPriorEstimator": "calibrated",
    "OnlineThresholdCalibratedEstimator": "calibrated",
    "BatchMedianCenteredEstimator": "calibrated",
    "ProbabilityPowerCalibratedEstimator": "calibrated",
    "RaschIRTOnlineEstimator": "irt",
    "GroupedResidualIRTOnlineEstimator": "irt",
    "BasicRaschFromAccuracyEstimator": "irt",
    "TwoPLIRTOnlineEstimator": "irt",
    "GraphLabelPropagationEstimator": "graph",
    "WeightedAveragedEnsembleEstimator": "ensemble",
    "FastTextKernelLogisticConfig": "fasttext_kernel",
    "FastTextKernelLogisticEstimator": "fasttext_kernel",
    "FastTextSVDRerankerConfig": "fasttext_kernel",
    "FastTextSVDRerankerEstimator": "fasttext_kernel",
    "FastTextSemanticConfig": "fasttext_semantic",
    "FastTextSemanticPrototypeEstimator": "fasttext_semantic",
    "NeuralEstimatorConfig": "neural",
    "NeuralEncoderDecoderEstimator": "neural",
    "AveragedEnsembleEstimator": "neural",
    "ObservedMatchUserVoteEstimator": "observed_user_vote",
    "OnlineUserLogisticEstimator": "online_user_logistic",
    "PersonalizedKNNPriorEstimator": "personalized",
    "PersonalizedHybridSignalEstimator": "personalized",
    "SVDRidgeUserEstimator": "svd",
    "UserKNNResponseEstimator": "user_knn",
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{_EXPORTS[name]}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(globals().keys() | set(__all__))
