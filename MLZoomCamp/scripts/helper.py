from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# -------------------------
# Column groups
# -------------------------
BOOL_COLS = ["Gender", "Ever_Married", "Graduated"]
NUM_COLS  = ["Age", "Work_Experience", "Family_Size"]
ORD_COLS  = ["Spending_Score"]                 # Low < Average < High
CAT_COLS  = ["Profession", "Var_1"]
TARGET    = "Segmentation"

ORD_MAP = {"low": 0, "average": 1, "high": 2}

# -------------------------
# Preprocessing helper functions #-------------------------
# Booleans
def _boolean_map(col: pd.Series, yes_no_map: Dict[str, int]) -> pd.Series:
    """"Map booleans to integers"""
    s = col.astype("string").str.strip().str.lower()
    return s.map(yes_no_map)


def _fit_bool_modes(X: pd.DataFrame) -> Dict[str, int]:
    """Compute modes for boolean columns"""
    modes = {}
    maps = {
        "Gender": {"male": 1, "female": 0},
        "Ever_Married": {"yes": 1, "no": 0},
        "Graduated": {"yes": 1, "no": 0}
    }
    for c in [col for col in BOOL_COLS if col in X.columns]:
        s = _boolean_map(X[c], maps.get(c, {}))
        modes[c] = int(s.mode(dropna=True).iloc[0]) if s.notna().any() else 0
    return modes


def _transform_booleans(X: pd.DataFrame, modes: Dict[str, int]) -> pd.DataFrame:
    """Transform boolean columns using mode"""
    out = pd.DataFrame(index=X.index)
    maps = {
        "Gender": {"male": 1, "female": 0},
        "Ever_Married": {"yes": 1, "no": 0},
        "Graduated": {"yes": 1, "no": 0}
    }
    for c in [col for col in BOOL_COLS if col in X.columns]:
        s = _boolean_map(X[c], maps.get(c, {}))
        out[c] = s.fillna(modes[c]).astype(int)
    return out


# Numerical
def _fit_numeric_stats(X: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute median/mean/std per numeric column"""
    stats = {}
    for c in cols:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.dropna().empty:
            stats[c] = {"median": 0.0, "mean": 0.0, "std": 1.0}
        else:
            med = float(s.median())
            mu  = float(s.fillna(med).mean())
            sd  = float(s.fillna(med).std(ddof=0))
            if not np.isfinite(sd) or sd == 0:
                sd = 1.0
            stats[c] = {"median": med, "mean": mu, "std": sd}
    return stats


def _transform_numeric_imputed(X: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Apply median imputation to numeric columns"""
    out = pd.DataFrame(index=X.index)
    for c, st in stats.items():
        s = pd.to_numeric(X[c], errors="coerce").fillna(st["median"])
        out[c] = s
    return out


def _transform_numeric_standardized(X: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Apply standardization to numeric columns"""
    out = pd.DataFrame(index=X.index)
    for c, st in stats.items():
        s = pd.to_numeric(X[c], errors="coerce").fillna(st["median"])
        out[c + "_z"] = (s - st["mean"]) / st["std"]
    return out


# Ordinal
def _transform_ordinal(X: pd.DataFrame) -> pd.DataFrame:
    """Transform ordinal columns to integers"""
    out = pd.DataFrame(index=X.index)
    if "Spending_Score" in X.columns:
        s = X["Spending_Score"].astype("string").str.strip().str.lower().map(ORD_MAP)
        out["Spending_Score_ordinal"] = s.fillna(-1).astype(int)  # no NaN for estimators
    return out

# Categorical
def _fit_categorical_levels(X: pd.DataFrame, cat_cols: List[str]) -> Dict[str, List[str]]:
    """Compute levels for each categorical column"""
    levels = {}
    for c in [col for col in cat_cols if col in X.columns]:
        s = X[c].astype("string").fillna("Missing")
        levels[c] = sorted(s.unique().tolist())
        if "Missing" not in levels[c]:
            levels[c].append("Missing")
    return levels


def _transform_categoricals_onehot(X: pd.DataFrame, levels: Dict[str, List[str]]) -> pd.DataFrame:
    """One-hot encode categorical columns using levels"""
    parts = []
    for c, cats in levels.items():
        s = X[c].astype("string").fillna("Missing")
        s = pd.Categorical(s, categories=cats, ordered=False)
        d = pd.get_dummies(s, prefix=c, dtype=int)
        parts.append(d)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)


# ================================================
# LOGISTIC REGRESSION PREPROCESSOR 
# ================================================
def preprocess_logreg_fit(X_train: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Fit preprocessing pipeline for logistic regression model"""
    X_train = X_train.copy()
    state: Dict = {}

    state["num_stats"]  = _fit_numeric_stats(X_train, [c for c in NUM_COLS if c in X_train.columns])
    state["bool_modes"] = _fit_bool_modes(X_train)
    state["cat_levels"] = _fit_categorical_levels(X_train, CAT_COLS)

    base_index = X_train.index

    parts = []
    if state["bool_modes"]:
        parts.append(_transform_booleans(X_train, state["bool_modes"]))
    parts.append(_transform_ordinal(X_train))
    if state["cat_levels"]:
        parts.append(_transform_categoricals_onehot(X_train, state["cat_levels"]))
    if state["num_stats"]:
        parts.append(_transform_numeric_standardized(X_train, state["num_stats"]))

    parts = [p.reindex(base_index) for p in parts if not p.empty]

    Xtr = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=base_index)
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    state["columns"] = Xtr.columns.tolist()
    return Xtr, state


def preprocess_logreg_transform(X_any: pd.DataFrame, state: Dict) -> pd.DataFrame:
    """Transform data using fitted logistic regression preprocessing pipeline"""
    X_any = X_any.copy()
    base_index = X_any.index

    parts = []
    if state.get("bool_modes"):
        parts.append(_transform_booleans(X_any, state["bool_modes"]))
    parts.append(_transform_ordinal(X_any))
    if state.get("cat_levels"):
        parts.append(_transform_categoricals_onehot(X_any, state["cat_levels"]))
    if state.get("num_stats"):
        parts.append(_transform_numeric_standardized(X_any, state["num_stats"]))

    parts = [p.reindex(base_index) for p in parts if not p.empty]

    Xout = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=base_index)
    Xout = Xout.reindex(columns=state["columns"], fill_value=0.0)
    Xout = Xout.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xout


# ===============================================
# XGBOOST PREPROCESSOR 
# ===============================================
def preprocess_xgb_fit(X_train: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Fit preprocessing pipeline for XGBoost model"""
    X_train = X_train.copy()
    state: Dict = {}

    state["num_stats"]  = _fit_numeric_stats(X_train, [c for c in NUM_COLS if c in X_train.columns])
    state["bool_modes"] = _fit_bool_modes(X_train)
    state["cat_levels"] = _fit_categorical_levels(X_train, CAT_COLS)

    base_index = X_train.index

    parts = []
    if state["bool_modes"]:
        parts.append(_transform_booleans(X_train, state["bool_modes"]))
    parts.append(_transform_ordinal(X_train))
    if state["cat_levels"]:
        parts.append(_transform_categoricals_onehot(X_train, state["cat_levels"]))
    if state["num_stats"]:
        parts.append(_transform_numeric_imputed(X_train, state["num_stats"]))

    parts = [p.reindex(base_index) for p in parts if not p.empty]

    Xtr = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=base_index)
    Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    state["columns"] = Xtr.columns.tolist()
    return Xtr, state


def preprocess_xgb_transform(X_any: pd.DataFrame, state: Dict) -> pd.DataFrame:
    """Transform data using fitted XGBoost preprocessing pipeline"""
    X_any = X_any.copy()
    base_index = X_any.index

    parts = []
    if state.get("bool_modes"):
        parts.append(_transform_booleans(X_any, state["bool_modes"]))
    parts.append(_transform_ordinal(X_any))
    if state.get("cat_levels"):
        parts.append(_transform_categoricals_onehot(X_any, state["cat_levels"]))
    if state.get("num_stats"):
        parts.append(_transform_numeric_imputed(X_any, state["num_stats"]))

    parts = [p.reindex(base_index) for p in parts if not p.empty]

    Xout = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=base_index)
    Xout = Xout.reindex(columns=state["columns"], fill_value=0.0)
    Xout = Xout.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return Xout


# =============================================
# Pipeline wrappers
# =============================================
def preprocess_pipeline_logreg(
    df: pd.DataFrame,
    state: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """Fit on df if state is None; otherwise transform with provided state."""
    if state is None:
        return preprocess_logreg_fit(df)
    else:
        return preprocess_logreg_transform(df, state), state


def preprocess_pipeline_xgb(
    df: pd.DataFrame,
    state: Optional[Dict] = None
) -> Tuple[pd.DataFrame, Dict]:
    """Fit on df if state is None; otherwise transform with provided state."""
    if state is None:
        return preprocess_xgb_fit(df)
    else:
        return preprocess_xgb_transform(df, state), state

