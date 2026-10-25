"""Evaluation -- stratified k-fold cross-validation of the whole pipeline."""
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """Build sklearn scorers, including configured ``fbeta_<beta>`` metrics."""
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            scorers[name] = make_scorer(
                fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0
            )
        else:
            scorers[name] = get_scorer(name)
    return scorers


def cross_validate_pipeline(
    pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1
):
    """
    Fits a fresh copy of `pipeline` on each fold's training part and scores it on that
    fold's validation part. Because `pipeline` contains the preprocessing too, imputation
    medians, encoder statistics and scaler means are re-learned inside every fold.

    Returns (fold_scores, y_oof):
      - fold_scores: one row per fold, with all validation metrics and the train-validation
        gap of the selection metric.
      - y_oof: out-of-fold probabilities of the positive class, from the fold model that
        did not train on each row.
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    y_oof = np.empty(len(X))
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        y_oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return fold_scores, y_oof


def cv_report(
    fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str
) -> str:
    """Format per-fold results and mean and standard deviation for every metric."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(
            float_format=lambda v: f"{v:.3f}"
        ),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """Confusion matrix and classification report from out-of-fold probabilities."""
    y_pred = (oof_proba >= threshold).astype(int)
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=["true 0", "true 1"],
        columns=["predicted 0", "predicted 1"],
    )
    return (
        f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
        f"{matrix.to_string()}\n\n"
        f"{classification_report(y_true, y_pred, zero_division=0)}"
    )


def fairness_report(
    y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
    threshold: float = 0.5,
) -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough
    to show that "accurate" and "fair" are not the same thing.

    For each race group, the false positive rate (share of people who did NOT reoffend
    but were predicted to) for:
        - our own model (out-of-fold predictions on the development set)
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk"
          prediction), on the same rows, for comparison
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (oof_proba >= threshold).astype(int)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]

    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {str(group):<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text
