# Baseline Predictive Pipeline -- ETAI 
Author: Vasco Rodrigues,  nº20231676

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # cross-validation metrics, out-of-fold reports + fairness check
│   ├── tuning.py           # Optuna search + nested cross-validation
│   └── results.py          # saves reports and appends scoreboard rows
├── results/                # timestamped reports + runs.csv (local outputs; not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/`. **Observation:** at the start, logistic regression already scored better than the decision tree |
| 3 | EDA and data diagnosis | Diagnosis-driven cleaning applied through `clean_dataset()` and the `diagnostics` section of `config.yaml`: placeholder tokens and out-of-range values (`validity_rules`) become `NaN`, category spellings are canonicalised, duplicate rows/ids and redundant (multicollinear) columns are removed; imputation matched to the missingness mechanism (median for MCAR numerics, most frequent for categoricals) plus `_was_missing` flags for the MNAR columns (`priors_count`, `c_charge_degree`); configurable encoder (`onehot`/`ordinal`/`count`/`target`) and scaler (`none`/`standard`/`minmax`/`robust`) in a `ColumnTransformer` placed inside the model `Pipeline`, so everything is fitted on training rows only; evaluation still a single holdout split. **Observation:** after preprocessing with the class diagnosis, logistic regression beat the decision tree (test accuracy 0.658 vs. 0.605; train/test gap +0.018 vs. +0.189). The large gap shows the tree is overfitting |
| 4 | Preprocessing recipe + cross-validation | Locked test set (`test_set` in `config.yaml`, 20%, seed 42, never changed); stratified 5-fold CV of the whole pipeline (preprocessing + model) on the development set (`cv` section); out-of-fold predictions for the classification report and fairness check; final model refit on all development rows; row-preserving `clean_dataset()` + training-only `drop_duplicate_rows()`; sklearn `TargetEncoder`, robust scaling; new `dummy` and `random_forest` models; new `knn` numeric imputation option. **Observation:** under 5-fold CV, logistic regression is still the best model (0.672 ± 0.013) and the only one that does not overfit (gap +0.003); the tree (0.610) and the untuned random forest (0.650) both have a train–val gap of about 0.08. A single holdout split was misleading (the tree moved 0.591 -> 0.610 under CV). KNN imputation gave no real improvement over the median, so the median stays|
| 5 | Leakage-safe hyperparameter tuning | Optuna tunes pipeline parameters using cross-validation of the complete preprocessing/model pipeline; nested stratified CV estimates the tuning procedure on untouched outer folds; the final best recipe is tuned on all development rows and refit there. The locked test set is never used. |
| 6 | Tree ensembles | Added oblique random forest (bagged sparse-projection trees) and AdaBoost (decision-tree weak learners); configured search spaces for decision trees and all three ensembles; added nested-CV and full-development tuning reports. Select the model to run with `model.type` in `config.yaml` (the Week 6 default was `oblique_random_forest`). |
| 7 | Probability-based evaluation and model comparison | Added Gradient Boosting, HistGradientBoosting and Gaussian Naive Bayes; reports eight classification metrics from out-of-fold probabilities, the confusion matrix and fairness checks; tunes on ROC-AUC, estimates tuning honestly with nested CV, and appends every run (including the dummy floor and untuned score) to `results/runs.csv`. |

## Week 6 ensemble evaluation (historical accuracy benchmark)

All six model types were run from this checkout on the same 5,771-row development set, with the configured five-fold CV and (where a search space exists) 30 Optuna trials inside five-fold nested CV. The default and nested-CV columns show mean ± standard deviation and mean train-validation gap. The full-development best-trial score is the maximum of 30 CV scores and is optimistic; use nested CV as the honest tuned estimate. Hyperparameters below are from each fresh full-development tuning run.

| Model | Default CV accuracy (mean ± std; gap) | Tuned nested CV accuracy (mean ± std; gap) | Best full-development CV score | Best full-development parameters |
|---|---|---|---:|---|
| Dummy | 0.549 ± 0.000 (-0.000) | — | — | — |
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | — |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013) | 0.679 | depth=6, min leaf=101, criterion=gini |
| Random forest | 0.650 ± 0.019 (+0.082) | 0.681 ± 0.015 (+0.009) | 0.683 | depth=22, min leaf=42, max features=0.5 |
| Oblique random forest | 0.647 ± 0.011 (+0.114) | 0.679 ± 0.013 (+0.013) | 0.683 | tree depth=6, tree min leaf=3, projection components=5 |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.017 (+0.004) | 0.684 | estimator depth=4, estimators=220, learning rate=0.0251 |

AdaBoost has the highest nested-CV mean at 0.682, but its lead over the random forest (0.681) is only 0.001, so the ensembles are effectively tied at this precision. Tuning substantially reduces the forests' overfitting gaps; AdaBoost with stumps is already close to its tuned result. The selected AdaBoost estimator is still shallow (depth 4) compared with the forest trees (depth 22); the oblique forest uses depth 6 trees and five projected components.

## Week 7 model scoreboard

All nine models below use the same cleaned development data, preprocessing, and five stratified folds. Tuned rows show the nested-CV ROC-AUC (five Optuna trials per fold); the untuned ROC-AUC and dummy floor use those same outer folds.

**Selection metric: ROC-AUC.** It measures how well the model ranks reoffenders above non-reoffenders across decision thresholds, which avoids choosing a particular classification threshold during model selection. Accuracy, balanced accuracy, precision, recall, F1, F2, and average precision are reported alongside it, while the confusion matrix and fairness check use an explicit 0.5 threshold.

| Model | Tuned? | ROC-AUC (mean ± std) | Untuned ROC-AUC | Dummy floor |
|---|---:|---:|---:|---:|
| Oblique random forest | Yes | 0.728 ± 0.012 | 0.685 | 0.500 |
| Random forest | Yes | 0.727 ± 0.016 | 0.684 | 0.500 |
| Gradient Boosting | Yes | 0.727 ± 0.014 | 0.727 | 0.500 |
| AdaBoost | Yes | 0.727 ± 0.014 | 0.723 | 0.500 |
| HistGradientBoosting | Yes | 0.726 ± 0.014 | 0.717 | 0.500 |
| Logistic regression | No | 0.718 ± 0.014 | 0.718 | 0.500 |
| Decision tree | Yes | 0.712 ± 0.014 | 0.610 | 0.500 |
| Naive Bayes | Yes | 0.685 ± 0.011 | 0.685 | 0.500 |
| Dummy | No | 0.500 ± 0.000 | 0.500 | 0.500 |

## Model evaluation

Evaluation uses a locked test set (20%, never touched) plus stratified 5-fold CV of the whole pipeline on the remaining development set. Imputation, encoding and scaling are re-fitted inside every fold, so validation rows never influence them. The holdout figures below are historical Week 4 results, not results from the current ensemble runs; current like-for-like model comparisons are in the Ensemble evaluation table above.

| Model | Holdout accuracy | CV accuracy (mean ± std) | CV train–val gap |
|---|---|---|---|
| Dummy | 0.550 | 0.549 ± 0.000 | -0.000 |
| Logistic regression | 0.674 | 0.672 ± 0.013 | +0.003 |
| Decision tree | 0.591 | 0.610 ± 0.018 | +0.085 |
| Random forest (300 trees) | 0.644 | 0.650 ± 0.018 | +0.083 |

The CV mean is the number to trust: it uses every development row for validation and reports its spread, whereas one holdout split is a single draw (the tree moves 0.591 -> 0.610). The week 2/3 conclusion is the same: logistic regression is the best and the only model that does not overfit; the untuned forest and the tree beat the dummy but now have a smaller train–val gap of about 0.08.

### Imputation experiment: median vs KNN

Same folds (`cv.random_state: 42`), only `imputation.numeric_strategy` changed (`knn` = `KNNImputer(n_neighbors=5)`, scaling applied before imputing).

| Model | Holdout median / KNN | CV median | CV KNN | Mean per-fold difference (KNN − median) |
|---|---|---|---|
| Logistic regression | 0.674 / 0.676 | 0.672 ± 0.013 | 0.673 ± 0.013 | +0.0005 |
| Decision tree | 0.591 / 0.608 | 0.610 ± 0.018 | 0.596 ± 0.014 | -0.013 |
| Random forest | 0.644 / 0.633 | 0.650 ± 0.018 | 0.647 ± 0.017 | -0.004 |

The holdout suggested KNN helps the tree (+0.017) and hurts the forest (-0.011); CV shows the opposite sign for the tree and differences within one fold-std for every model. KNN imputation brings no real gain here (few missing values), so `median` stays in `config.yaml`.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
