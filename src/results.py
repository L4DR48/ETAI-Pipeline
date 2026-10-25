"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment
you scroll past it or close the window. This module writes the full metrics, confusion
matrix, classification report and fairness table to a timestamped file in `results/`, and
keeps comparable run summaries in the scoreboard CSV.
"""
import os
from datetime import datetime
from typing import Optional

import pandas as pd


def save_run(
    results_dir: str, config: dict, report_text: str, run_id: Optional[str] = None
) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(results_dir, f"run_{run_id}.txt")

    header = (
        f"Run: {run_id}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  scaler={config['preprocessing']['scaler']}  "
        f"imputation={config['preprocessing'].get('imputation')}\n"
        f"Locked test set: size={config['test_set']['size']}  random_state={config['test_set']['random_state']}\n"
        f"CV metrics: {config['cv']['metrics']}\n"
        f"Selection metric: {config['cv']['selection_metric']}\n"
        f"CV: {config['cv']}\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path


def save_scores(path: str, row: dict):
    """Append one model run to the scoreboard CSV and return all recorded runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs
