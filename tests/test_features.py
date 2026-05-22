"""Tests mínimos para src/features.py — completar conforme se implementa la Fase 2."""
import pytest
import numpy as np
from src.features import decay_weight, expected_score, DECAY_HALFLIFE_YEARS
import pandas as pd


def test_decay_weight_today_is_one():
    now = pd.Timestamp.utcnow()
    assert abs(decay_weight(now, now) - 1.0) < 1e-9


def test_decay_weight_halflife():
    """A 4 años el peso debe ser ~0.5."""
    ref = pd.Timestamp("2025-01-01")
    past = ref - pd.Timedelta(days=int(DECAY_HALFLIFE_YEARS * 365.25))
    assert abs(decay_weight(past, ref) - 0.5) < 1e-3


def test_expected_score_symmetry():
    """E(A vs B) + E(B vs A) = 1 sin ventaja de local."""
    assert abs(expected_score(1500, 1800) + expected_score(1800, 1500) - 1.0) < 1e-9


def test_expected_score_home_advantage():
    """Con ventaja de local de 100 puntos, equipos iguales: E > 0.5."""
    assert expected_score(1500, 1500, home_adv=100) > 0.5
