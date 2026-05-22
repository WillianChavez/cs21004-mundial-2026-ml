"""
model.py - Entrenamiento, tuning y persistencia.

Modelos:
    LogisticRegression multinomial (baseline interpretable)
    XGBoost (multi:softprob)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def train_logreg(X, y, sample_weight=None, random_state=42):
    """Baseline: Logistic Regression multinomial con regularizacion L2."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
            max_iter=2000,
            random_state=random_state,
        )),
    ])
    fit_kwargs = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
    pipe.fit(X, y, **fit_kwargs)
    return pipe


def train_xgboost(X, y, sample_weight=None, n_iter=15, n_splits=4, random_state=42):
    """XGBoost multinomial con RandomizedSearchCV + TimeSeriesSplit."""
    param_dist = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.05, 0.07, 0.1],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [1.0, 1.5, 2.0],
    }
    base = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=2,
        verbosity=0,
    )
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=tscv,
        n_jobs=1,
        random_state=random_state,
        verbose=0,
        refit=True,
    )
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    search.fit(X, y, **fit_kwargs)
    return search


def calibrate(model, X_cal, y_cal, method="isotonic"):
    """Isotonic calibration sobre un modelo ya entrenado.

    Usa cv='prefit' para no re-entrenar el modelo base.
    """
    cal = CalibratedClassifierCV(model, method=method, cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def save_model(model, name: str) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path
