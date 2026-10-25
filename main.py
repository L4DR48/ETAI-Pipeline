"""Entry point for the predictive pipeline."""
from datetime import datetime

import optuna
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.evaluate import (
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.model import build_model
from src.preprocessing import (
    build_preprocessor,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.results import save_run, save_scores
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Stateless cleaning and de-duplication happen before the locked test split.
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    X_dev, X_test, y_dev, _y_test, extras_dev, _extras_test = split_dev_test(
        X,
        y,
        extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)
    model_type = config["model"]["type"]

    print(
        f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
        f"{cv_config['n_splits']}-fold CV"
    )
    print(
        f"Data:  {len(X_dev)} development rows   |   "
        f"{len(X_test)} test rows locked away (not used)"
    )

    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])
    floor_pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model({"type": "dummy"})),
    ])
    floor_scores, _ = cross_validate_pipeline(
        floor_pipeline,
        X_dev,
        y_dev,
        cv,
        {selection_metric: scorers[selection_metric]},
        selection_metric,
        n_jobs,
    )
    fold_scores, oof_proba = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs
    )
    floor = floor_scores["validation"].mean()
    untuned = fold_scores["validation"].mean()

    print("\nDefault hyperparameters (config.yaml)")
    print(
        f"  {selection_metric}: {untuned:.3f} +/- "
        f"{fold_scores['validation'].std(ddof=1):.3f}   "
        f"train-validation gap: {fold_scores['gap'].mean():+.3f}   "
        f"no-model floor: {floor:.3f}"
    )
    report = (
        f"Hyperparameters from config.yaml: {config['model'].get('params')}\n\n"
        + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
        + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}"
    )
    final_scores = fold_scores
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    search_space = tuning_config.get("search_spaces", {}).get(model_type)
    study = None

    if tuning_enabled and search_space:
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(
            n_splits=tuning_config["n_splits"],
            shuffle=True,
            random_state=tuning_seed,
        )

        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv,
            inner_cv,
            scorers,
            selection_metric,
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )
        print(
            f"  {selection_metric}: {final_scores['validation'].mean():.3f} +/- "
            f"{final_scores['validation'].std(ddof=1):.3f}   "
            "(honest estimate of the tuned model)"
        )
        report += "\n\n" + cv_report(
            final_scores,
            metrics,
            selection_metric,
            "Nested cross-validation (tuning procedure, estimated honestly)",
        )

        pipeline, study = tune_pipeline(
            pipeline,
            X_dev,
            y_dev,
            inner_cv,
            scorers[selection_metric],
            search_space,
            n_trials,
            tuning_seed,
            n_jobs=n_jobs,
        )
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            label = name.replace("model__", "")
            if isinstance(value, float):
                print(f"    {label:<24s} {value:.4g}")
            else:
                print(f"    {label:<24s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)
    elif tuning_enabled:
        print(f"Tuning skipped for {model_type}: no search space is configured.")
        report += f"\n\nTuning skipped for {model_type}: no search space is configured."

    print(
        f"\nAll metrics ({'tuned model' if study else 'default hyperparameters'}, "
        "mean +/- std)"
    )
    for name in metrics:
        print(
            f"  {name:<18s} {final_scores[name].mean():.3f} +/- "
            f"{final_scores[name].std(ddof=1):.3f}"
        )
    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(
        y_dev,
        oof_proba,
        extras_dev,
        sensitive_attr=config["data"]["sensitive_attr"],
    )

    pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {model_type} refit on all {len(X_dev)} development rows."
    print(refit)
    report += "\n" + refit + "\n"

    row = {
        "run": run_id,
        "data": config["data"]["path"],
        "model": model_type,
        "tuned": bool(study),
        "encoder": config["preprocessing"]["encoder"],
        "scaler": config["preprocessing"]["scaler"],
        "params": str(config["model"].get("params")),
        "tuned_params": str(study.best_params) if study else "",
        "metric": selection_metric,
        "mean": final_scores["validation"].mean(),
        "std": final_scores["validation"].std(ddof=1),
        "untuned": untuned,
        "floor": floor,
        **{
            f"fold_{i}": score
            for i, score in enumerate(final_scores["validation"], start=1)
        },
    }
    scoreboard_path = config.get("output", {}).get("scoreboard", "results/runs.csv")
    runs = save_scores(scoreboard_path, row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
    board = (
        same.sort_values("mean", ascending=False)[
            ["run", "model", "tuned", "encoder", "scaler", "mean", "std", "untuned", "floor"]
        ]
        .rename(columns={"mean": selection_metric})
        .to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    path = save_run(
        config.get("output", {}).get("results_dir", "results"),
        config,
        report,
        run_id,
    )
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
