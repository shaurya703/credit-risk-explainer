"""SHAP explanations for the trained tree model.

The headline feature: every prediction comes with a breakdown of *which features
pushed the risk up or down, and by how much*. Built directly from the saved
pipeline so the API/dashboard and the notebooks share identical explanation logic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from . import config
from .features import OUTPUT_FEATURES


class CreditExplainer:
    """Wraps a SHAP TreeExplainer over the pipeline's final tree model.

    The pipeline's preprocessing steps are applied first, then SHAP explains the
    booster in the *engineered* feature space (the space the model reasons in).
    """

    def __init__(self, model: Pipeline):
        self.model = model
        self.preprocess = model[:-1]            # everything but the classifier
        self.estimator = model[-1]              # the tree model
        self.explainer = shap.TreeExplainer(self.estimator)
        base = self.explainer.expected_value
        # XGBoost returns a scalar margin; normalize to a float.
        self.base_value = float(np.ravel(base)[0])

    def _transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        Xt = self.preprocess.transform(X_raw)
        if not isinstance(Xt, pd.DataFrame):
            Xt = pd.DataFrame(Xt, columns=OUTPUT_FEATURES, index=X_raw.index)
        return Xt

    def explain_one(self, X_raw: pd.DataFrame) -> dict:
        """Return the per-feature SHAP contributions for a single applicant row.

        Contributions are in log-odds (margin) space, the additive space SHAP
        guarantees: base_value + sum(shap_values) = model margin for this row.
        """
        Xt = self._transform(X_raw.iloc[[0]])
        sv = self._shap_values(Xt)[0]
        contributions = [
            {
                "feature": feat,
                "label": config.FEATURE_LABELS.get(feat, feat),
                "value": float(Xt.iloc[0][feat]),
                "shap_value": float(sv[i]),
            }
            for i, feat in enumerate(OUTPUT_FEATURES)
        ]
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)
        return {"base_value": self.base_value, "contributions": contributions}

    def global_importance(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Mean(|SHAP|) per feature across a sample — global feature importance."""
        Xt = self._transform(X_raw)
        sv = self._shap_values(Xt)
        imp = np.abs(sv).mean(axis=0)
        return (pd.DataFrame({"feature": OUTPUT_FEATURES, "mean_abs_shap": imp})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True))

    def shap_values_matrix(self, X_raw: pd.DataFrame):
        """Raw SHAP value matrix + transformed features (for beeswarm plots)."""
        Xt = self._transform(X_raw)
        return self._shap_values(Xt), Xt

    def _shap_values(self, Xt: pd.DataFrame) -> np.ndarray:
        sv = self.explainer.shap_values(Xt)
        sv = np.asarray(sv)
        if sv.ndim == 3:  # some versions return (n, features, classes)
            sv = sv[..., -1]
        return sv
