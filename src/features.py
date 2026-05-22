"""
features.py - Ingenieria de caracteristicas.
"""
from __future__ import annotations

from math import log
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# ---------- Constantes ----------
DECAY_HALFLIFE_YEARS = 4.0
DECAY_LAMBDA = log(2.0) / DECAY_HALFLIFE_YEARS

ELO_INIT = 1500.0
ELO_HOME_ADVANTAGE = 100.0

K_BY_TOURNAMENT_TYPE = {
    "WC": 60.0,
    "Continental": 50.0,
    "WC_Qualifier": 40.0,
    "Qualifier": 40.0,
    "Friendly": 30.0,
    "Other": 35.0,
}

LABEL_TO_INT = {"W": 0, "D": 1, "L": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


# ---------- Helpers de timezone ----------
def _to_naive(t):
    t = pd.Timestamp(t)
    if getattr(t, "tz", None) is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t


def _now_naive():
    return _to_naive(pd.Timestamp.utcnow())


# ---------- ELO ----------
def expected_score(ra: float, rb: float, home_adv: float = 0.0) -> float:
    """Probabilidad esperada de que A gane a B (Elo clasico)."""
    return 1.0 / (1.0 + 10.0 ** ((rb - ra - home_adv) / 400.0))


def _result_score(label):
    if label == "W":
        return 1.0, 0.0
    if label == "L":
        return 0.0, 1.0
    return 0.5, 0.5


def _goal_factor(goal_diff: int) -> float:
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def compute_elo(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.sort_values("date").reset_index(drop=True).copy()
    elo = defaultdict(lambda: ELO_INIT)
    elo_h = np.empty(len(df))
    elo_a = np.empty(len(df))
    exp_h = np.empty(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        h, a = row.home_team, row.away_team
        rh, ra = elo[h], elo[a]
        home_adv = 0.0 if row.neutral else ELO_HOME_ADVANTAGE
        e_home = expected_score(rh, ra, home_adv)
        elo_h[i] = rh
        elo_a[i] = ra
        exp_h[i] = e_home
        s_home, s_away = _result_score(row.label)
        K = K_BY_TOURNAMENT_TYPE.get(row.tournament_type, 35.0)
        K *= _goal_factor(int(row.goal_diff))
        elo[h] = rh + K * (s_home - e_home)
        elo[a] = ra + K * (s_away - (1.0 - e_home))
    df["elo_pre_home"] = elo_h
    df["elo_pre_away"] = elo_a
    df["elo_diff"] = elo_h - elo_a
    df["elo_expected_home"] = exp_h
    df.attrs["final_elo"] = pd.Series(dict(elo)).sort_values(ascending=False)
    return df


# ---------- Rolling form ----------
def rolling_form(matches: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = matches.sort_values("date").reset_index(drop=True).copy()
    history = defaultdict(lambda: deque(maxlen=window))
    n = len(df)
    h_wr, h_gf, h_ga = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    a_wr, a_gf, a_ga = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i, row in enumerate(df.itertuples(index=False)):
        h, a = row.home_team, row.away_team
        hh, ha = history[h], history[a]
        if hh:
            arr = np.array(hh)
            h_wr[i] = (arr[:, 0] == 1).mean()
            h_gf[i] = arr[:, 1].mean()
            h_ga[i] = arr[:, 2].mean()
        if ha:
            arr = np.array(ha)
            a_wr[i] = (arr[:, 0] == 1).mean()
            a_gf[i] = arr[:, 1].mean()
            a_ga[i] = arr[:, 2].mean()
        hs, as_ = int(row.home_score), int(row.away_score)
        if row.label == "W":
            home_result = 1
        elif row.label == "D":
            home_result = 0
        else:
            home_result = -1
        hh.append((home_result, hs, as_))
        ha.append((-home_result if home_result != 0 else 0, as_, hs))
    df[f"home_win_rate_{window}"] = h_wr
    df[f"home_gf_{window}"] = h_gf
    df[f"home_ga_{window}"] = h_ga
    df[f"away_win_rate_{window}"] = a_wr
    df[f"away_gf_{window}"] = a_gf
    df[f"away_ga_{window}"] = a_ga
    return df


# ---------- Head-to-Head ----------
def head_to_head(matches: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = matches.sort_values("date").reset_index(drop=True).copy()
    h2h = defaultdict(lambda: deque(maxlen=window))
    n = len(df)
    wr = np.full(n, np.nan)
    gd = np.full(n, np.nan)
    cnt = np.zeros(n, dtype=int)
    for i, row in enumerate(df.itertuples(index=False)):
        a, b = row.home_team, row.away_team
        key = frozenset((a, b))
        past = list(h2h[key])
        if past:
            wins_home = sum(1 for w, _, h_team in past if w == a)
            draws = sum(1 for w, _, _ in past if w is None)
            gdiff = [(gd_ if h_team == a else -gd_) for _, gd_, h_team in past]
            wr[i] = (wins_home + 0.5 * draws) / len(past)
            gd[i] = float(np.mean(gdiff))
            cnt[i] = len(past)
        if row.label == "W":
            winner = a
        elif row.label == "L":
            winner = b
        else:
            winner = None
        h2h[key].append((winner, int(row.goal_diff), a))
    df["h2h_home_win_rate"] = wr
    df["h2h_avg_goal_diff"] = gd
    df["h2h_count"] = cnt
    return df


# ---------- Decay temporal ----------
def decay_weight(date, ref_date=None) -> float:
    ref = _to_naive(ref_date) if ref_date is not None else _now_naive()
    d = _to_naive(date)
    age = (ref - d).total_seconds() / (365.25 * 86400)
    return float(np.exp(-DECAY_LAMBDA * max(age, 0.0)))


def decay_weights_array(dates, ref_date=None):
    ref = _to_naive(ref_date) if ref_date is not None else _now_naive()
    s = pd.to_datetime(pd.Series(dates))
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    age = (ref - s).dt.total_seconds() / (365.25 * 86400)
    age = age.clip(lower=0.0)
    return np.exp(-DECAY_LAMBDA * age).to_numpy()


# ---------- Ensamblado ----------
def build_feature_matrix(master: pd.DataFrame, window_form=10, window_h2h=5):
    df = master.copy()
    df = compute_elo(df)
    df = rolling_form(df, window=window_form)
    df = head_to_head(df, window=window_h2h)

    feature_cols = [
        "elo_pre_home", "elo_pre_away", "elo_diff", "elo_expected_home",
        f"home_win_rate_{window_form}", f"home_gf_{window_form}", f"home_ga_{window_form}",
        f"away_win_rate_{window_form}", f"away_gf_{window_form}", f"away_ga_{window_form}",
        "h2h_home_win_rate", "h2h_avg_goal_diff", "h2h_count",
    ]

    df["is_home_advantage"] = (~df["neutral"]).astype(int)
    feature_cols.append("is_home_advantage")

    tt_dum = pd.get_dummies(df["tournament_type"], prefix="tt").astype(int)
    df = pd.concat([df, tt_dum], axis=1)
    feature_cols.extend(tt_dum.columns.tolist())

    df["sample_weight"] = decay_weights_array(df["date"])

    df_ok = df.dropna(subset=[f"home_win_rate_{window_form}", f"away_win_rate_{window_form}"]).copy()
    df_ok["h2h_home_win_rate"] = df_ok["h2h_home_win_rate"].fillna(0.5)
    df_ok["h2h_avg_goal_diff"] = df_ok["h2h_avg_goal_diff"].fillna(0.0)

    X = df_ok[feature_cols].astype(float).values
    y = df_ok["label"].map(LABEL_TO_INT).values.astype(int)
    sw = df_ok["sample_weight"].values
    return df_ok, X, y, sw, feature_cols
