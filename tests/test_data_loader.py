"""Tests de data loader."""
from src.data_loader import (
    load_results,
    load_fifa_ranking,
    load_groups_2026,
    load_bracket_2026,
)


def test_results_has_expected_columns():
    df = load_results()
    for col in ("date", "home_team", "away_team", "home_score", "away_score", "tournament"):
        assert col in df.columns


def test_results_has_reasonable_size():
    df = load_results()
    assert len(df) > 40_000


def test_fifa_ranking_loads():
    df = load_fifa_ranking()
    for col in ("date", "rank", "team", "points"):
        assert col in df.columns


def test_groups_2026_has_48_teams():
    g = load_groups_2026()
    assert len(g) == 48
    assert g["team"].nunique() == 48
    assert g["group"].nunique() == 12


def test_bracket_2026_has_correct_phases():
    b = load_bracket_2026()
    counts = b["round"].value_counts().to_dict()
    assert counts.get("R32") == 16
    assert counts.get("R16") == 8
    assert counts.get("QF") == 4
    assert counts.get("SF") == 2
