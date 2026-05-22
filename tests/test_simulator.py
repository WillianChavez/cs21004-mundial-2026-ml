"""Tests del simulador Monte Carlo."""
import numpy as np
from src.simulator import sample_outcome, sample_outcome_no_draw, wilson_ci


def test_sample_outcome_returns_valid_class():
    rng = np.random.default_rng(0)
    for _ in range(100):
        r = sample_outcome(0.5, 0.3, 0.2, rng)
        assert r in (0, 1, 2)


def test_sample_outcome_no_draw_never_returns_draw():
    rng = np.random.default_rng(0)
    for _ in range(100):
        r = sample_outcome_no_draw(0.4, 0.4, 0.2, rng)
        assert r in (0, 2)


def test_sample_outcome_distribution_converges():
    """Con p_w=0.7, p_d=0.2, p_l=0.1 el local gana ~70%."""
    rng = np.random.default_rng(42)
    wins = sum(sample_outcome(0.7, 0.2, 0.1, rng) == 0 for _ in range(10_000))
    assert 0.68 < wins / 10_000 < 0.72


def test_wilson_ci_bounds():
    low, high = wilson_ci(0.16, 10_000)
    assert 0.0 <= low < 0.16 < high <= 1.0
