"""End-to-end training pipeline: data -> models -> evaluation -> artifacts.

Run with `python -m src.train` (or `make pipeline`). Produces:
  models/model.joblib   the chosen pipeline (preprocessing + classifier)
  models/metrics.json   model comparison, test metrics, threshold, feature stats
  reports/figures/*.png  ROC/PR/confusion/calibration + SHAP global & example plots

This is the single reproducible path that turns raw data into everything the API,
dashboard, and README rely on.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
)

from . import config, data, model as model_mod
from .explain import CreditExplainer
from .features import OUTPUT_FEATURES

MODEL_KINDS = ["logreg", "rf", "xgboost"]
SHAP_SAMPLE = 2000  # rows used for SHAP global plots (keeps it fast)


def _feature_stats(df) -> dict:
    """Median / 5th / 95th percentile per raw feature, for dashboard form defaults."""
    stats = {}
    for col in config.RAW_FEATURES:
        s = df[col].dropna()
        stats[col] = {
            "median": float(s.median()),
            "p05": float(s.quantile(0.05)),
            "p95": float(s.quantile(0.95)),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    return stats


def main() -> None:
    config.ensure_dirs()
    print("Loading data ...")
    df = data.load_raw()
    X_train, X_val, X_test, y_train, y_val, y_test = data.split_data(df)
    spw = model_mod.pos_weight(y_train)
    print(f"  train={len(X_train)} val={len(X_val)} test={len(X_test)} | scale_pos_weight={spw:.2f}")

    # --- train & compare on validation ---
    # Imbalance handling: LR/RF use class_weight='balanced'; XGBoost is trained
    # UNWEIGHTED so its probabilities stay calibrated for use as a risk score, and
    # imbalance is handled at decision time via the cost-based threshold. The
    # scale_pos_weight alternative is explored as an ablation in notebook 03.
    trained, comparison = {}, {}
    for kind in MODEL_KINDS:
        print(f"Training {kind} ...")
        m = model_mod.build_model(kind)
        m.fit(X_train, y_train)
        trained[kind] = m
        val_metrics = model_mod.evaluate(m, X_val, y_val, threshold=0.5)
        comparison[kind] = {"roc_auc": val_metrics["roc_auc"], "pr_auc": val_metrics["pr_auc"]}
        print(f"  val ROC-AUC={val_metrics['roc_auc']:.4f}  PR-AUC={val_metrics['pr_auc']:.4f}")

    # --- pick best by PR-AUC (imbalance-appropriate) ---
    best_kind = max(comparison, key=lambda k: comparison[k]["pr_auc"])
    best = trained[best_kind]
    print(f"Best model: {best_kind}")

    # --- choose decision threshold from business cost on validation ---
    val_prob = best.predict_proba(X_val)[:, 1]
    threshold = model_mod.find_cost_threshold(y_val, val_prob)
    print(f"Cost-optimal threshold ({config.COST_FALSE_NEGATIVE:.0f}:1 FN:FP) = {threshold:.3f}")

    # --- final honest evaluation on the untouched test set ---
    test_metrics = model_mod.evaluate(best, X_test, y_test, threshold=threshold)
    print(f"TEST  ROC-AUC={test_metrics['roc_auc']:.4f}  PR-AUC={test_metrics['pr_auc']:.4f}  "
          f"recall={test_metrics['recall']:.3f}  precision={test_metrics['precision']:.3f}")

    # --- persist model + metrics ---
    model_mod.save_model(best)
    explainer = CreditExplainer(best)
    Xs = X_test.sample(min(SHAP_SAMPLE, len(X_test)), random_state=config.RANDOM_STATE)
    global_imp = explainer.global_importance(Xs)

    metrics = {
        "dataset": {"rows": int(len(df)), "default_rate": float(df[config.TARGET].mean())},
        "best_model": best_kind,
        "scale_pos_weight": spw,
        "threshold": threshold,
        "cost_assumption": {"fn": config.COST_FALSE_NEGATIVE, "fp": config.COST_FALSE_POSITIVE},
        "validation_comparison": comparison,
        "test": test_metrics,
        "global_importance": global_imp.to_dict(orient="records"),
        "feature_stats": _feature_stats(df),
    }
    model_mod.save_metrics(metrics)
    print(f"Saved model -> {config.MODEL_PATH.name}, metrics -> {config.METRICS_PATH.name}")

    # --- figures ---
    _make_figures(trained, best, best_kind, threshold, X_test, y_test, explainer, Xs)
    print(f"Saved figures -> {config.FIGURES}")


def _make_figures(trained, best, best_kind, threshold, X_test, y_test, explainer, Xs) -> None:
    # ROC curves (all models, on test)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for kind, m in trained.items():
        p = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        from sklearn.metrics import roc_auc_score
        ax.plot(fpr, tpr, lw=2, label=f"{kind} (AUC={roc_auc_score(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(config.FIGURES / "roc_curves.png", dpi=140); plt.close(fig)

    # PR curves
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for kind, m in trained.items():
        p = m.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, p)
        from sklearn.metrics import average_precision_score
        ax.plot(rec, prec, lw=2, label=f"{kind} (AP={average_precision_score(y_test, p):.3f})")
    ax.axhline(y_test.mean(), color="grey", ls="--", lw=1, label=f"baseline ({y_test.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(config.FIGURES / "pr_curves.png", dpi=140); plt.close(fig)

    # Confusion matrix (best, at chosen threshold)
    prob = best.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No default", "Default"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix — {best_kind} @ threshold {threshold:.2f}")
    fig.tight_layout(); fig.savefig(config.FIGURES / "confusion_matrix.png", dpi=140); plt.close(fig)

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(mean_pred, frac_pos, "o-", lw=2, label=best_kind)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (test set)"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(config.FIGURES / "calibration.png", dpi=140); plt.close(fig)

    # SHAP global: bar + beeswarm
    sv, Xt = explainer.shap_values_matrix(Xs)
    shap.summary_plot(sv, Xt, plot_type="bar", show=False, max_display=15)
    plt.title("Global feature importance (mean |SHAP|)")
    plt.tight_layout(); plt.savefig(config.FIGURES / "shap_global_bar.png", dpi=140); plt.close()

    shap.summary_plot(sv, Xt, show=False, max_display=15)
    plt.title("SHAP summary (impact on default risk)")
    plt.tight_layout(); plt.savefig(config.FIGURES / "shap_global_beeswarm.png", dpi=140); plt.close()

    # Example local explanation (a high-risk applicant) via waterfall
    high_idx = int(np.argmax(prob))
    exp = shap.Explanation(
        values=explainer.shap_values_matrix(X_test.iloc[[high_idx]])[0][0],
        base_values=explainer.base_value,
        data=explainer._transform(X_test.iloc[[high_idx]]).iloc[0].values,
        feature_names=OUTPUT_FEATURES,
    )
    shap.plots.waterfall(exp, max_display=12, show=False)
    plt.title("Why this applicant was flagged (SHAP)")
    plt.tight_layout(); plt.savefig(config.FIGURES / "shap_example_waterfall.png", dpi=140); plt.close()


if __name__ == "__main__":
    main()
