"""
Ensemble Models Module
Ensemble learning models for stock prediction
"""

from .ensemble_model import EnsembleModel, StackingEnsemble

__all__ = [
    'EnsembleModel',
    'StackingEnsemble'
]