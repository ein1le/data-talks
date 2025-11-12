#!/usr/bin/env python3
import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression

# <-- your helper module from the notebook
from helper import (
    preprocess_pipeline_logreg,
    TARGET
)

# =========================
# Defaults / Hyperparameters
# =========================
DEFAULT_RANDOM_STATE = 42

# Split ratios: 80% train, 10% val, 10% test
DEFAULT_TEMP_SIZE = 0.20      # portion carved out of full dataset to later split into val & test
DEFAULT_VAL_RATIO_WITHIN_TEMP = 0.50  # of the temp set, how much goes to VAL (the rest goes to TEST)

# CV settings
DEFAULT_C_MIN_EXP = -3        # 1e-3
DEFAULT_C_MAX_EXP = 2         # 1e+2
DEFAULT_C_NUM = 12            # how many C values between 1e-3 and 1e+2
DEFAULT_CV_FOLDS = 5
DEFAULT_SCORING = "balanced_accuracy"

# Logistic Regression hyperparameters
DEFAULT_LR_SOLVER = "lbfgs"
DEFAULT_LR_PENALTY = "l2"
DEFAULT_LR_MAX_ITER = 1000
# Leave multi_class=None to avoid sklearn deprecation warning;
# if you want to force "multinomial", set it via CLI flag.
DEFAULT_LR_MULTI_CLASS = None

# Registry
DEFAULT_REGISTRY_DIR = "../model_registry"
DEFAULT_MODEL_PREFIX = "logreg"


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
    Split into: Train (1 - temp_size), Validation (temp_size * val_ratio_within_temp),
                Test (temp_size * (1 - val_ratio_within_temp)).
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
# 2) Preprocess (fit on TRAIN)
# =========================
def logreg_preprocess(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
):
    """
    Fit preprocessing on TRAIN only, then transform VAL and TEST.
    Returns: Xtr_lr, Xva_lr, Xte_lr, lr_state
    """
    Xtr_lr, lr_state = preprocess_pipeline_logreg(X_train)          # FIT
    Xva_lr, _        = preprocess_pipeline_logreg(X_val,  lr_state) # TRANSFORM
    Xte_lr, _        = preprocess_pipeline_logreg(X_test, lr_state) # TRANSFORM

    # Enforce index alignment (defensive)
    Xtr_lr = Xtr_lr.loc[X_train.index.intersection(Xtr_lr.index)]
    Xva_lr = Xva_lr.loc[X_val.index.intersection(Xva_lr.index)]
    Xte_lr = Xte_lr.loc[X_test.index.intersection(Xte_lr.index)]
    return Xtr_lr, Xva_lr, Xte_lr, lr_state


# =========================
# 3) CV training
# =========================
def logistic_train_cv(
    Xtr_lr: pd.DataFrame,
    y_train: pd.Series,
    c_min_exp: int = DEFAULT_C_MIN_EXP,
    c_max_exp: int = DEFAULT_C_MAX_EXP,
    c_num: int = DEFAULT_C_NUM,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    solver: str = DEFAULT_LR_SOLVER,
    penalty: str = DEFAULT_LR_PENALTY,
    max_iter: int = DEFAULT_LR_MAX_ITER,
    multi_class: str | None = DEFAULT_LR_MULTI_CLASS,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Grid-search over C with StratifiedKFold CV. Returns fitted GridSearchCV.
    """
    C_values = np.logspace(c_min_exp, c_max_exp, num=c_num)

    # Build model kwargs conditionally to avoid deprecation warnings
    model_kwargs = dict(
        solver=solver,
        penalty=penalty,
        max_iter=max_iter,
        random_state=random_state,
    )
    if multi_class is not None:
        model_kwargs["multi_class"] = multi_class

    logreg = LogisticRegression(**model_kwargs)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    param_grid = {"C": C_values}

    grid_lr = GridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True
    )
    grid_lr.fit(Xtr_lr, y_train)
    return grid_lr


# =========================
# 4) “Validate” (no predictions)
# =========================
def logistic_validate(grid_lr: GridSearchCV) -> dict:
    """
    Return the best params and best CV score from the fitted GridSearchCV.
    (No predictions are made here.)
    """
    return {
        "cv_best_params": grid_lr.best_params_,
        "cv_best_score": grid_lr.best_score_,
    }


# =========================
# 5) Save final model (refit on train+val)
# =========================
def save_model(
    grid_lr: GridSearchCV,
    Xtr_lr: pd.DataFrame,
    Xva_lr: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    lr_state: dict,
    registry_dir: str = DEFAULT_REGISTRY_DIR,
    model_prefix: str = DEFAULT_MODEL_PREFIX,
    meta: dict | None = None
) -> str:
    """
    Refit the best LR on TRAIN+VAL, package with preprocess state, and save as pickle.
    Returns output path.
    """
    best_C = grid_lr.best_params_["C"]

    # Refit final model on TRAIN+VAL
    Xtrval_lr = pd.concat([Xtr_lr, Xva_lr], axis=0)
    y_trval   = pd.concat([y_train, y_val], axis=0)

    # Use the same kwargs as grid’s estimator, but override C to best_C
    base_estimator: LogisticRegression = grid_lr.best_estimator_
    final_kwargs = base_estimator.get_params(deep=False)
    final_kwargs["C"] = best_C

    final_logreg = LogisticRegression(**final_kwargs).fit(Xtrval_lr, y_trval)

    # Prepare artifact
    Path(registry_dir).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = str(Path(registry_dir) / f"{model_prefix}_{ts}.pkl")

    artifact = {
        "model": final_logreg,
        "preprocess_state": lr_state,
        "feature_columns": Xtrval_lr.columns.tolist(),
        "target_name": TARGET,
        "cv_best_params": grid_lr.best_params_,
        "cv_best_score": grid_lr.best_score_,
        "metadata": {
            "created_at": ts,
            "algorithm": "LogisticRegression",
            "cv_n_splits": grid_lr.cv,
            "scoring": grid_lr.scoring,
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
    c_min_exp: int = DEFAULT_C_MIN_EXP,
    c_max_exp: int = DEFAULT_C_MAX_EXP,
    c_num: int = DEFAULT_C_NUM,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    solver: str = DEFAULT_LR_SOLVER,
    penalty: str = DEFAULT_LR_PENALTY,
    max_iter: int = DEFAULT_LR_MAX_ITER,
    multi_class: str | None = DEFAULT_LR_MULTI_CLASS,
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

    # 2) Preprocess
    Xtr_lr, Xva_lr, Xte_lr, lr_state = logreg_preprocess(X_train, X_val, X_test)

    # 3) Train with CV
    grid_lr = logistic_train_cv(
        Xtr_lr=Xtr_lr, y_train=y_train,
        c_min_exp=c_min_exp, c_max_exp=c_max_exp, c_num=c_num,
        cv_folds=cv_folds, scoring=scoring,
        solver=solver, penalty=penalty, max_iter=max_iter,
        multi_class=multi_class, random_state=random_state
    )

    # 4) Validate (no predictions)
    val_info = logistic_validate(grid_lr)
    print(f"[CV] Best params: {val_info['cv_best_params']}")
    print(f"[CV] Best {scoring}: {val_info['cv_best_score']:.4f}")

    # 5) Save final refit model
    out_path = save_model(
        grid_lr=grid_lr,
        Xtr_lr=Xtr_lr, Xva_lr=Xva_lr,
        y_train=y_train, y_val=y_val,
        lr_state=lr_state,
        registry_dir=registry_dir,
        model_prefix=model_prefix,
        meta={"random_state": random_state}
    )
    print(f"[LOGREG] Saved trained model to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Logistic Regression with CV and save model artifact.")
    parser.add_argument("csv_path", type=str, help="Path to input CSV.")

    # Split / CV
    parser.add_argument("--temp-size", type=float, default=DEFAULT_TEMP_SIZE, help="Fraction of data set aside for val+test (default: 0.2).")
    parser.add_argument("--val-ratio-within-temp", type=float, default=DEFAULT_VAL_RATIO_WITHIN_TEMP, help="How much of temp goes to validation (default: 0.5, i.e., 10% val / 10% test).")
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS, help="StratifiedKFold splits (default: 5).")
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE, help="Random seed (default: 42).")

    # Hyperparameters
    parser.add_argument("--c-min-exp", type=int, default=DEFAULT_C_MIN_EXP, help="min exponent for logspace C (default: -3).")
    parser.add_argument("--c-max-exp", type=int, default=DEFAULT_C_MAX_EXP, help="max exponent for logspace C (default: 2).")
    parser.add_argument("--c-num", type=int, default=DEFAULT_C_NUM, help="number of C values between 10^min and 10^max (default: 12).")
    parser.add_argument("--solver", type=str, default=DEFAULT_LR_SOLVER, help="LogReg solver (default: lbfgs).")
    parser.add_argument("--penalty", type=str, default=DEFAULT_LR_PENALTY, help="Penalty (default: l2).")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_LR_MAX_ITER, help="Max iterations (default: 1000).")
    parser.add_argument("--multi-class", type=str, default=None, help="Set to 'multinomial' to force that mode (default: None).")
    parser.add_argument("--scoring", type=str, default=DEFAULT_SCORING, help="CV scoring metric (default: balanced_accuracy).")

    # Registry
    parser.add_argument("--registry-dir", type=str, default=DEFAULT_REGISTRY_DIR, help="Model registry directory (default: ../model_registry).")
    parser.add_argument("--model-prefix", type=str, default=DEFAULT_MODEL_PREFIX, help="Saved model filename prefix (default: logreg).")

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"CSV not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    main(
        csv_path=args.csv_path,
        temp_size=args.temp_size,
        val_ratio_within_temp=args.val_ratio_within_temp,
        random_state=args.random_state,
        c_min_exp=args.c_min_exp,
        c_max_exp=args.c_max_exp,
        c_num=args.c_num,
        cv_folds=args.cv_folds,
        scoring=args.scoring,
        solver=args.solver,
        penalty=args.penalty,
        max_iter=args.max_iter,
        multi_class=args.multi_class,
        registry_dir=args.registry_dir,
        model_prefix=args.model_prefix,
    )
