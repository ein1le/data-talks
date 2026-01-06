#!/usr/bin/env python3
import os
import sys
import time
import argparse
import pickle
from pathlib import Path
from typing import List
from types import SimpleNamespace


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from helper import (
    preprocess_pipeline_xgb,
    TARGET
)

# =========================
# Hyperparameters
# =========================
DEFAULT_RANDOM_STATE = 23

DEFAULT_TEMP_SIZE = 0.20      # 20% val + test
DEFAULT_VAL_RATIO_WITHIN_TEMP = 0.50  # 10% val / 10% test

# CV settings
DEFAULT_CV_FOLDS = 5
DEFAULT_SCORING = "accuracy"

# XGBoost base params
DEFAULT_OBJECTIVE = "multi:softprob"
DEFAULT_EVAL_METRIC = "mlogloss"
DEFAULT_N_JOBS = -1

# XGBoost CV Grid
DEFAULT_MAX_DEPTHS: List[int] = [3, 4, 5, 6]
DEFAULT_MIN_CHILD_WEIGHT: List[int] = [1, 3, 5]
DEFAULT_LEARNING_RATES: List[float] = [0.03, 0.05, 0.1]
DEFAULT_N_ESTIMATORS: List[int] = [200, 400, 600]
DEFAULT_REG_LAMBDA: List[float] = [1.0, 2.0]
DEFAULT_REG_ALPHA: List[float] = [0.0, 0.5]

# Registry
DEFAULT_REGISTRY_DIR = "../model_registry"
DEFAULT_MODEL_PREFIX = "xgb"


# =========================
# Small CLI parsing helpers
# =========================
def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


# =========================
# 1) Train/Val/Test split
# =========================
def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    temp_size: float = DEFAULT_TEMP_SIZE,
    val_ratio_within_temp: float = DEFAULT_VAL_RATIO_WITHIN_TEMP,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Split into: Train, Val, Test.
    Stratified by y.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1.0 - val_ratio_within_temp),
        stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# =========================
# 2) Label encoding
# =========================
def encode_labels(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
    """
    Fit LabelEncoder on the union of train/val/test
    """
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_val, y_test], axis=0))
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)
    num_classes = len(le.classes_)
    return y_train_enc, y_val_enc, y_test_enc, le, num_classes


# =========================
# 3) Preprocess (fit on TRAIN)
# =========================
def xgb_preprocess(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Fit preprocessing on TRAIN only, then transform VAL and TEST.
    Returns: Xtr_xgb, Xva_xgb, Xte_xgb, xgb_state
    """
    Xtr_xgb, xgb_state = preprocess_pipeline_xgb(X_train)          # FIT
    Xva_xgb, _         = preprocess_pipeline_xgb(X_val,  xgb_state)  # TRANSFORM
    Xte_xgb, _         = preprocess_pipeline_xgb(X_test, xgb_state)  # TRANSFORM

    # Defensive alignment to ensure identical row counts & order
    Xtr_xgb = Xtr_xgb.loc[X_train.index.intersection(Xtr_xgb.index)]
    Xva_xgb = Xva_xgb.loc[X_val.index.intersection(Xva_xgb.index)]
    Xte_xgb = Xte_xgb.loc[X_test.index.intersection(Xte_xgb.index)]
    return Xtr_xgb, Xva_xgb, Xte_xgb, xgb_state


# =========================
# 4) CV training
# =========================

def xgb_train_cv(
    Xtr_xgb: pd.DataFrame,
    y_train_enc: np.ndarray,
    num_classes: int,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    random_state: int = DEFAULT_RANDOM_STATE,
    # grids (two at a time)
    max_depths: List[int] = DEFAULT_MAX_DEPTHS,
    min_child_weight: List[int] = DEFAULT_MIN_CHILD_WEIGHT,
    learning_rates: List[float] = DEFAULT_LEARNING_RATES,
    n_estimators_list: List[int] = DEFAULT_N_ESTIMATORS,
    reg_lambda_list: List[float] = DEFAULT_REG_LAMBDA,
    reg_alpha_list: List[float] = DEFAULT_REG_ALPHA,
):
    """
    Sequential tuning in three 2D stages:

      1) (max_depth, min_child_weight)
      2) (learning_rate, n_estimators)  with Stage 1 fixed
      3) (reg_lambda, reg_alpha)        with Stages 1-2 fixed

    Returns a SimpleNamespace with fields:
      - best_params_     (dict of all 6 tuned params)
      - best_score_      (CV score of final params, via cross_val_score)
      - best_estimator_  (XGBClassifier fitted on TRAIN with final params)
      - cv               (cv_folds)
      - scoring          (scoring)
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Stage 1: tree structure
    base_stage1 = XGBClassifier(
        objective=DEFAULT_OBJECTIVE,
        num_class=num_classes,
        eval_metric=DEFAULT_EVAL_METRIC,
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
        # leave the rest to grid
    )
    grid1 = GridSearchCV(
        estimator=base_stage1,
        param_grid={
            "max_depth": max_depths,
            "min_child_weight": min_child_weight,
        },
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        verbose=0,
    )
    grid1.fit(Xtr_xgb, y_train_enc)
    best_stage1 = grid1.best_params_
    # print(f"[XGB][Stage 1] {best_stage1}  score={grid1.best_score_:.4f}")

    # Stage 2: learning dynamics
    base_stage2 = XGBClassifier(
        objective=DEFAULT_OBJECTIVE,
        num_class=num_classes,
        eval_metric=DEFAULT_EVAL_METRIC,
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
        # fix Stage 1
        max_depth=best_stage1["max_depth"],
        min_child_weight=best_stage1["min_child_weight"],
    )
    grid2 = GridSearchCV(
        estimator=base_stage2,
        param_grid={
            "learning_rate": learning_rates,
            "n_estimators": n_estimators_list,
        },
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        verbose=0,
    )
    grid2.fit(Xtr_xgb, y_train_enc)
    best_stage2 = {
        "learning_rate": grid2.best_params_["learning_rate"],
        "n_estimators": grid2.best_params_["n_estimators"],
    }
    # print(f"[XGB][Stage 2] {best_stage2}  score={grid2.best_score_:.4f}")

    # Stage 3: regularization
    base_stage3 = XGBClassifier(
        objective=DEFAULT_OBJECTIVE,
        num_class=num_classes,
        eval_metric=DEFAULT_EVAL_METRIC,
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
        # fix Stage 1 + Stage 2
        max_depth=best_stage1["max_depth"],
        min_child_weight=best_stage1["min_child_weight"],
        learning_rate=best_stage2["learning_rate"],
        n_estimators=best_stage2["n_estimators"],
    )
    grid3 = GridSearchCV(
        estimator=base_stage3,
        param_grid={
            "reg_lambda": reg_lambda_list,
            "reg_alpha": reg_alpha_list,
        },
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        verbose=0,
    )
    grid3.fit(Xtr_xgb, y_train_enc)
    best_stage3 = {
        "reg_lambda": grid3.best_params_["reg_lambda"],
        "reg_alpha": grid3.best_params_["reg_alpha"],
    }
    # print(f"[XGB][Stage 3] {best_stage3}  score={grid3.best_score_:.4f}")

    # Final params & estimator
    final_params = {
        **best_stage1,
        **best_stage2,
        **best_stage3,
    }

    final_estimator = XGBClassifier(
        objective=DEFAULT_OBJECTIVE,
        num_class=num_classes,
        eval_metric=DEFAULT_EVAL_METRIC,
        random_state=random_state,
        n_jobs=DEFAULT_N_JOBS,
        **final_params,
    )

    final_cv_scores = cross_val_score(
        final_estimator, Xtr_xgb, y_train_enc,
        scoring=scoring, cv=cv, n_jobs=-1
    )
    final_score = float(np.mean(final_cv_scores))

    # fit on full TRAIN (not train+val; you refit on train+val later in save_model)
    final_estimator.fit(Xtr_xgb, y_train_enc)

    # mimic the GridSearchCV attributes your downstream code expects
    result = SimpleNamespace(
        best_params_=final_params,
        best_score_=final_score,
        best_estimator_=final_estimator,
        cv=cv_folds,
        scoring=scoring,
    )
    return result



# =========================
# 5) Validation
# =========================
def xgb_validate(grid_xgb: GridSearchCV) -> dict:
    """
    Return the best params and best CV score from the fitted GridSearchCV.
    """
    return {
        "cv_best_params": grid_xgb.best_params_,
        "cv_best_score": grid_xgb.best_score_,
    }


# =========================
# 6) Refitting on TRAIN+VAL and saving
# =========================
def save_model(
    grid_xgb: GridSearchCV,
    Xtr_xgb: pd.DataFrame,
    Xva_xgb: pd.DataFrame,
    y_train_enc: np.ndarray,
    y_val_enc: np.ndarray,
    xgb_state: dict,
    label_encoder: LabelEncoder,
    registry_dir: str = DEFAULT_REGISTRY_DIR,
    model_prefix: str = DEFAULT_MODEL_PREFIX,
    meta: dict | None = None
) -> str:
    """
    Refit the best XGB on TRAIN+VAL, package with preprocess state + label encoder,
    and save as pickle. Returns output path.
    """
    best_params = grid_xgb.best_params_

    # Refit final model on TRAIN+VAL
    Xtrval_xgb = pd.concat([Xtr_xgb, Xva_xgb], axis=0)
    y_trval_enc = np.concatenate([y_train_enc, y_val_enc], axis=0)

    base_estimator: XGBClassifier = grid_xgb.best_estimator_
    final_kwargs = base_estimator.get_params(deep=False)
    # best_params already reflected in best_estimator_, but we keep the override explicit:
    final_kwargs.update(best_params)

    final_xgb = XGBClassifier(**final_kwargs).fit(Xtrval_xgb, y_trval_enc)

    # Prepare artifact
    Path(registry_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(registry_dir) / f"{model_prefix}_{ts}.pkl")

    artifact = {
        "model": final_xgb,
        "preprocess_state": xgb_state,
        "label_encoder": label_encoder,
        "feature_columns": Xtrval_xgb.columns.tolist(),
        "target_name": TARGET,
        "cv_best_params": grid_xgb.best_params_,
        "cv_best_score": grid_xgb.best_score_,
        "metadata": {
            "created_at": ts,
            "algorithm": "XGBClassifier",
            "cv_n_splits": grid_xgb.cv,
            "scoring": grid_xgb.scoring,
            **(meta or {})
        }
    }

    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)

    return out_path


# =========================
# Main
# =========================
def main(
    csv_path: str,
    temp_size: float = DEFAULT_TEMP_SIZE,
    val_ratio_within_temp: float = DEFAULT_VAL_RATIO_WITHIN_TEMP,
    random_state: int = DEFAULT_RANDOM_STATE,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    # grid lists
    max_depths: List[int] = DEFAULT_MAX_DEPTHS,
    min_child_weight: List[int] = DEFAULT_MIN_CHILD_WEIGHT,
    learning_rates: List[float] = DEFAULT_LEARNING_RATES,
    n_estimators_list: List[int] = DEFAULT_N_ESTIMATORS,
    reg_lambda_list: List[float] = DEFAULT_REG_LAMBDA,
    reg_alpha_list: List[float] = DEFAULT_REG_ALPHA,
    registry_dir: str = DEFAULT_REGISTRY_DIR,
    model_prefix: str = DEFAULT_MODEL_PREFIX,
):
    # Load
    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in CSV.")
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # 1) Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y,
        temp_size=temp_size,
        val_ratio_within_temp=val_ratio_within_temp,
        random_state=random_state
    )

    # 2) Encode labels
    y_train_enc, y_val_enc, y_test_enc, le, num_classes = encode_labels(y_train, y_val, y_test)

    # 3) Preprocess (fit on TRAIN)
    Xtr_xgb, Xva_xgb, Xte_xgb, xgb_state = xgb_preprocess(X_train, X_val, X_test)

    # 4) Train with CV
    grid_xgb = xgb_train_cv(
        Xtr_xgb=Xtr_xgb, y_train_enc=y_train_enc, num_classes=num_classes,
        cv_folds=cv_folds, scoring=scoring, random_state=random_state,
        max_depths=max_depths, min_child_weight=min_child_weight,
        learning_rates=learning_rates, n_estimators_list=n_estimators_list,
        reg_lambda_list=reg_lambda_list, reg_alpha_list=reg_alpha_list
    )

    # 5) Validate (no predictions)
    val_info = xgb_validate(grid_xgb)
    print(f"[CV] Best params: {val_info['cv_best_params']}")
    print(f"[CV] Best {scoring}: {val_info['cv_best_score']:.4f}")

    # 6) Save final refit model
    out_path = save_model(
        grid_xgb=grid_xgb,
        Xtr_xgb=Xtr_xgb, Xva_xgb=Xva_xgb,
        y_train_enc=y_train_enc, y_val_enc=y_val_enc,
        xgb_state=xgb_state, label_encoder=le,
        registry_dir=registry_dir, model_prefix=model_prefix,
        meta={"random_state": random_state}
    )
    print(f"[XGB] Saved trained model to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost with CV and save model artifact.")
    parser.add_argument("csv_path", type=str, help="Path to input CSV.")

    # Split / CV
    parser.add_argument("--temp-size", type=float, default=DEFAULT_TEMP_SIZE, help="Fraction of data set aside for val+test (default: 0.2).")
    parser.add_argument("--val-ratio-within-temp", type=float, default=DEFAULT_VAL_RATIO_WITHIN_TEMP, help="How much of temp goes to validation (default: 0.5 → 10% val / 10% test).")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, help="StratifiedKFold splits (default: 5).")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed (default: 42).")
    parser.add_argument("--scoring", type=str, default=DEFAULT_SCORING, help="CV scoring metric (default: accuracy).")

    # Grid lists 
    parser.add_argument("--max-depths", type=str, default=",".join(map(str, DEFAULT_MAX_DEPTHS)),
                        help=f"Comma-separated max_depth values (default: {','.join(map(str, DEFAULT_MAX_DEPTHS))}).")
    parser.add_argument("--min-child-weight", type=str, default=",".join(map(str, DEFAULT_MIN_CHILD_WEIGHT)),
                        help=f"Comma-separated min_child_weight values (default: {','.join(map(str, DEFAULT_MIN_CHILD_WEIGHT))}).")
    parser.add_argument("--learning-rates", type=str, default=",".join(map(str, DEFAULT_LEARNING_RATES)),
                        help=f"Comma-separated learning_rate values (default: {','.join(map(str, DEFAULT_LEARNING_RATES))}).")
    parser.add_argument("--n-estimators", type=str, default=",".join(map(str, DEFAULT_N_ESTIMATORS)),
                        help=f"Comma-separated n_estimators values (default: {','.join(map(str, DEFAULT_N_ESTIMATORS))}).")
    parser.add_argument("--reg-lambda", type=str, default=",".join(map(str, DEFAULT_REG_LAMBDA)),
                        help=f"Comma-separated reg_lambda values (default: {','.join(map(str, DEFAULT_REG_LAMBDA))}).")
    parser.add_argument("--reg-alpha", type=str, default=",".join(map(str, DEFAULT_REG_ALPHA)),
                        help=f"Comma-separated reg_alpha values (default: {','.join(map(str, DEFAULT_REG_ALPHA))}).")

    # Registry
    parser.add_argument("--registry-dir", type=str, default=DEFAULT_REGISTRY_DIR, help="Model registry directory (default: ../model_registry).")
    parser.add_argument("--model-prefix", type=str, default=DEFAULT_MODEL_PREFIX, help="Saved model filename prefix (default: xgb).")

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"CSV not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    # Parse comma-separated lists to Python list
    max_depths = _parse_int_list(args.max_depths)
    min_child_weight = _parse_int_list(args.min_child_weight)
    learning_rates = _parse_float_list(args.learning_rates)
    n_estimators_list = _parse_int_list(args.n_estimators)
    reg_lambda_list = _parse_float_list(args.reg_lambda)
    reg_alpha_list = _parse_float_list(args.reg_alpha)

    main(
        csv_path=args.csv_path,
        temp_size=args.temp_size,
        val_ratio_within_temp=args.val_ratio_within_temp,
        random_state=args.random_state,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        max_depths=max_depths,
        min_child_weight=min_child_weight,
        learning_rates=learning_rates,
        n_estimators_list=n_estimators_list,
        reg_lambda_list=reg_lambda_list,
        reg_alpha_list=reg_alpha_list,
        registry_dir=args.registry_dir,
        model_prefix=args.model_prefix,
    )
