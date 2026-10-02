# Baseline Predictive Pipeline -- ETAI 
Author: Vasco Rodrigues,  nº20231676

Week1:
In the beggining the logistic regression is giving better scores than the decision tree

Week2:

After applying the preprocessing steps in the pipeline , taking into account diagnostic results seen in class, logistic regression is performing better than decision trees:

  Test accuracy: 0.658 vs. 0.605 for Decision Tree
  Train/test gap: +0.018 vs. +0.189

The big gap between training and testing for DT shows the model is overfitting.

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
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA and data diagnosis | Diagnosis-driven cleaning applied through `clean_dataset()` and the `diagnostics` section of `config.yaml`: placeholder tokens and out-of-range values (`validity_rules`) become `NaN`, category spellings are canonicalised, duplicate rows/ids and redundant (multicollinear) columns are removed; imputation matched to the missingness mechanism (median for MCAR numerics, most frequent for categoricals) plus `_was_missing` flags for the MNAR columns (`priors_count`, `c_charge_degree`); configurable encoder (`onehot`/`ordinal`/`count`/`target`) and scaler (`none`/`standard`/`minmax`/`robust`) in a `ColumnTransformer` placed inside the model `Pipeline`, so everything is fitted on training rows only; evaluation still a single holdout split |
| 4 | Preprocessing recipe + cross-validation | Locked test set (`test_set` in `config.yaml`, 20%, seed 42, never changed); stratified 5-fold CV of the whole pipeline (preprocessing + model) on the development set (`cv` section); out-of-fold predictions for the classification report and fairness check; final model refit on all development rows; row-preserving `clean_dataset()` + training-only `drop_duplicate_rows()`; sklearn `TargetEncoder`, robust scaling; new `dummy` and `random_forest` models; new `knn` numeric imputation option |

## Model evaluation

Evaluation uses a locked test set (20%, never touched) plus stratified 5-fold CV of the whole pipeline on the remaining development set. Imputation, encoding and scaling are re-fitted inside every fold, so validation rows never influence them. Holdout = one 75/25 split of the development set (seed 42).

| Model | Holdout accuracy | CV accuracy (mean ± std) | CV train–val gap |
|---|---|---|---|
| Dummy | 0.550 | 0.549 ± 0.000 | -0.000 |
| Logistic regression | 0.674 | 0.672 ± 0.013 | +0.003 |
| Decision tree | 0.591 | 0.610 ± 0.018 | +0.085 |
| Random forest (300 trees) | 0.644 | 0.650 ± 0.018 | +0.083 |

The CV mean is the number to trust: it uses every development row for validation and reports its spread, whereas one holdout split is a single draw (the tree moves 0.591 -> 0.610). The week 2/3 conclusion holds: logistic regression is best and the only model that does not overfit; the untuned forest and the tree beat the dummy but have a train–val gap of about 0.08.

### Imputation experiment: median vs KNN

Same folds (`cv.random_state: 42`), only `imputation.numeric_strategy` changed (`knn` = `KNNImputer(n_neighbors=5)`, scaling applied before imputing).

| Model | Holdout median / KNN | CV median | CV KNN | Mean per-fold difference (KNN − median) |
|---|---|---|---|---|
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
