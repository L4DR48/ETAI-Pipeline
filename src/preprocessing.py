"""
Preprocessing -- data cleaning and train/test preparation.

The original week 2 `preprocess` function remains available while the
diagnosis-driven cleaning functions are integrated into the pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from category_encoders import CountEncoder, TargetEncoder

def canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnosis: dict) -> pd.DataFrame:
    """
    Apply the EDA notebook's diagnosis: category cleanup, domain-rule / placeholder ->
    NaN conversion, de-duplication, redundant-column removal. Target-column-agnostic --
    safe to call on label-free inference data.
    """
    out = df.copy()
    placeholder_tokens = set(diagnosis["placeholder_tokens"])

    # numeric columns that loaded as text because of placeholder tokens
    for col in diagnosis.get("numeric_text_columns", ["priors_count", "prior_offenses"]):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    # domain-rule violations -> NaN
    for col, rule in diagnosis.get("validity_rules", {}).items():
        if col not in out.columns:
            continue
        invalid = out[col].notna()
        if "min" in rule:
            invalid &= out[col] < rule["min"]
        if "max" in rule:
            invalid |= out[col].notna() & (out[col] > rule["max"])
        out.loc[invalid, col] = np.nan

    # category canonicalization (also folds placeholder tokens to NaN)
    category_maps = diagnosis.get("canonical_categories", diagnosis.get("canonical_maps", {}))
    out = canonicalize_categories(out, category_maps, placeholder_tokens)

    # duplicates: exact row dupes and repeated ids point at the same rows here -- drop, keep first
    out = out.drop_duplicates()
    if "id" in out.columns:
        out = out.drop_duplicates(subset="id", keep="first")

    # redundant columns found via multicollinearity
    columns_to_drop = diagnosis.get("redundant_columns", diagnosis.get("columns_to_drop", []))
    cols_to_drop = [c for c in columns_to_drop if c in out.columns and c != "id"]
    out = out.drop(columns=cols_to_drop)

    return out


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """Add missingness flags before the corresponding values are imputed."""
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """Return features, target, and audit columns for training or inference."""
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None
    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    X = df[feature_cols]
    return X, y, extras


_SCALERS = {
    "none": "passthrough",
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}
_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda: TargetEncoder(handle_unknown="value", handle_missing="value"),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """Build the configured transformer; fitting happens inside the model pipeline."""
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    imputation = preprocessing_config.get("imputation", {})

    if scaler_name not in _SCALERS:
        raise ValueError(f"Unknown scaler: {scaler_name}. Options: {list(_SCALERS)}")
    if encoder_name not in _ENCODERS:
        raise ValueError(f"Unknown encoder: {encoder_name}. Options: {list(_ENCODERS)}")

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[encoder_name]()
    indicator_cols = [f"{c}_was_missing" for c in mnar_indicator_sources]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_train_test(X, y, extras, test_size: float, random_state: int):
    """Keep features, labels, and audit columns aligned through a stratified split."""
    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, extras_train, extras_test