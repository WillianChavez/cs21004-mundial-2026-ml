"""
evaluation.py - Metricas y validacion.

Metricas principales:
    Log-Loss multinomial (calibracion de probabilidades)
    Brier Score multi-clase
    Reliability diagram
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def multiclass_brier(y_true, y_proba):
    """Brier Score multi-clase = (1/N) sum sum (p_ij - y_ij)^2."""
    y_true = np.asarray(y_true, dtype=int)
    n_classes = y_proba.shape[1]
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1)))


def reliability_data(y_true, y_proba_class, n_bins=10):
    """Para una clase: bins por confianza vs frecuencia observada.

    Args:
        y_true: 0/1 si la observacion pertenece a la clase
        y_proba_class: P(clase=k)
    """
    y_true = np.asarray(y_true)
    y_proba_class = np.asarray(y_proba_class)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_proba_class, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    mean_pred = np.zeros(n_bins)
    mean_true = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            mean_pred[b] = y_proba_class[mask].mean()
            mean_true[b] = y_true[mask].mean()
            count[b] = mask.sum()
    return mean_pred, mean_true, count


def evaluate(name: str, y_true, y_proba, classes=(0, 1, 2)) -> dict:
    """Calcula Log-Loss, Brier y Accuracy."""
    n = len(y_true)
    ll = log_loss(y_true, y_proba, labels=list(classes))
    br = multiclass_brier(y_true, y_proba)
    y_pred = y_proba.argmax(axis=1)
    acc = float((y_pred == np.asarray(y_true)).mean())
    return {"model": name, "n": n, "log_loss": ll, "brier": br, "accuracy": acc}


def baseline_uniform_proba(n_rows, n_classes=3):
    """Predice 1/n_classes para todo."""
    return np.full((n_rows, n_classes), 1.0 / n_classes)


def baseline_elo_proba(df, elo_diff_col="elo_diff", expected_col="elo_expected_home"):
    """Convierte ELO_expected en P(W,D,L) usando una regla simple.

    P(W) = expected; P(L) = 1 - expected
    P(D) se estima como 0.22 + 0.06 * exp(-(elo_diff/250)^2) (modelo simple)
    Luego renormaliza.
    """
    exp_h = df[expected_col].values
    elo_diff = df[elo_diff_col].values
    p_d = 0.22 + 0.06 * np.exp(-(elo_diff / 250.0) ** 2)
    p_w = exp_h * (1 - p_d)
    p_l = (1 - exp_h) * (1 - p_d)
    P = np.stack([p_w, p_d, p_l], axis=1)
    P = P / P.sum(axis=1, keepdims=True)
    return P
