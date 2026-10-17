"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean -> drop duplicates -> lock test set
    -> cross-validate (preprocessing + model) on the development set
    -> nested CV and tune -> refit selected model on development rows -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    build_preprocessor,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.model import build_model
from src.evaluate import cross_validate_pipeline, cv_report, fairness_report, oof_classification_report
from src.results import save_run
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # stateless cleaning + training-only de-duplication may see all rows; anything that
    # learns from data (imputer, encoder, scaler, model) lives inside the Pipeline
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # locked test set: set aside once, never used for fitting, tuning or comparing
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    pipeline = Pipeline([
        ("prep", build_preprocessor(config["preprocessing"])),
        ("model", build_model(config["model"])),
    ])

    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=shuffle,
        random_state=cv_config.get("random_state") if shuffle else None,
    )
    scoring = cv_config.get("scoring", "accuracy")
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scoring, n_jobs=cv_config.get("n_jobs", 1)
    )

    report = cv_report(fold_scores, scoring)
    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    tuning_text = ""
    final_pipeline = pipeline
    tuning_config = config.get("tuning", {})
    if tuning_config.get("enabled", False):
        search_space = tuning_config.get("search_spaces", {}).get(config["model"]["type"])
        if search_space:
            inner_cv = StratifiedKFold(
                n_splits=tuning_config["n_splits"],
                shuffle=True,
                random_state=tuning_config["random_state"],
            )
            nested_scores, _ = nested_cross_validate(
                pipeline,
                X_dev,
                y_dev,
                cv,
                inner_cv,
                scoring,
                search_space,
                tuning_config["n_trials"],
                tuning_config["random_state"],
                n_jobs=cv_config.get("n_jobs", 1),
            )
            final_pipeline, study = tune_pipeline(
                pipeline,
                X_dev,
                y_dev,
                inner_cv,
                scoring,
                search_space,
                tuning_config["n_trials"],
                tuning_config["random_state"],
                n_jobs=cv_config.get("n_jobs", 1),
            )
            tuning_text = tuning_report(study, nested_scores, scoring)
        else:
            tuning_text = (
                f"Tuning skipped for {config['model']['type']}: "
                "no search space is configured for this model."
            )
            print(tuning_text)

    # CV estimates the recipe; refit the selected recipe on all development rows.
    final_model = final_pipeline.fit(X_dev, y_dev)
    print(f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows.")

    results_dir = config.get("output", {}).get("results_dir", "results")
    if tuning_text:
        report += "\n\n" + tuning_text
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
