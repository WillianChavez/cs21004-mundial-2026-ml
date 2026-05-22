"""
simulator_fast.py - Monte Carlo del Mundial 2026 (version optimizada).

Precomputa todas las probabilidades de pares (2256 + 141 = 2397) una sola vez
y luego solo hace lookup en el loop de Monte Carlo.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.simulator import (
    FEATURE_COLS, HOST_COUNTRIES, POINTS_W, POINTS_D, POINTS_L,
    build_match_features, sample_outcome, sample_outcome_no_draw,
    R32_SLOTS, R16_PAIRS, QF_PAIRS, SF_PAIRS, wilson_ci,
)
from src.data_loader import load_groups_2026

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJ_ROOT / "data" / "processed"
MODELS_DIR = PROJ_ROOT / "models"


def precompute_probas(snapshot, model):
    """Precomputa P(W,D,L) para todos los pares ordenados."""
    teams = list(snapshot.keys())
    proba_neutral, proba_host = {}, {}

    feats_n, keys_n = [], []
    for a in teams:
        for b in teams:
            if a == b: continue
            feats_n.append(build_match_features(snapshot[a], snapshot[b], host_neutral=True))
            keys_n.append((a, b))
    X = np.stack(feats_n)
    P = model.predict_proba(X)
    for k, p in zip(keys_n, P):
        proba_neutral[k] = p

    feats_h, keys_h = [], []
    for host in HOST_COUNTRIES:
        if host not in snapshot: continue
        for opp in teams:
            if opp == host: continue
            feats_h.append(build_match_features(snapshot[host], snapshot[opp], host_neutral=False))
            keys_h.append((host, opp))
    if feats_h:
        X = np.stack(feats_h)
        P = model.predict_proba(X)
        for k, p in zip(keys_h, P):
            proba_host[k] = p
    return proba_neutral, proba_host


def simulate_group_fast(group_teams, proba_neutral, proba_host, snapshot, host_country, rng):
    stats = {t: {'P': 0, 'GD': 0, 'GF': 0, 'team': t} for t in group_teams}
    for i in range(4):
        for j in range(i+1, 4):
            a, b = group_teams[i], group_teams[j]
            is_host_a = (a == host_country)
            is_host_b = (b == host_country)
            if is_host_b and not is_host_a:
                a, b = b, a; is_host_a = True
            if is_host_a:
                p = proba_host[(a, b)]
            else:
                p = proba_neutral[(a, b)]
            outcome = sample_outcome(p[0], p[1], p[2], rng)
            lambda_a = max(snapshot[a]['gf'], 0.3)
            lambda_b = max(snapshot[b]['gf'], 0.3)
            ga = int(rng.poisson(lambda_a))
            gb = int(rng.poisson(lambda_b))
            if outcome == 0:
                if ga <= gb: ga = gb + 1
            elif outcome == 2:
                if gb <= ga: gb = ga + 1
            else:
                ga = gb = max(ga, gb)
            if outcome == 0:
                stats[a]['P'] += POINTS_W
            elif outcome == 2:
                stats[b]['P'] += POINTS_W
            else:
                stats[a]['P'] += POINTS_D; stats[b]['P'] += POINTS_D
            stats[a]['GD'] += ga - gb; stats[b]['GD'] += gb - ga
            stats[a]['GF'] += ga; stats[b]['GF'] += gb
    return sorted(stats.values(),
                  key=lambda s: (s['P'], s['GD'], s['GF'], rng.random()),
                  reverse=True)


def simulate_knockout_fast(team_a, team_b, proba_neutral, rng):
    p = proba_neutral[(team_a, team_b)]
    out = sample_outcome_no_draw(p[0], p[1], p[2], rng)
    return team_a if out == 0 else team_b


def simulate_tournament_once_fast(groups_df, snapshot, proba_neutral, proba_host, rng):
    standings_by_group = {}
    for letter in groups_df['group'].unique():
        teams = groups_df[groups_df['group'] == letter]['team'].tolist()
        host = next((t for t in teams if t in HOST_COUNTRIES), None)
        standings_by_group[letter] = simulate_group_fast(teams, proba_neutral, proba_host, snapshot, host, rng)

    slot_to_team = {}
    thirds = []
    for letter, st in standings_by_group.items():
        slot_to_team[f'1{letter}'] = st[0]['team']
        slot_to_team[f'2{letter}'] = st[1]['team']
        thirds.append({'letter': letter, **st[2]})

    sorted_thirds = sorted(thirds, key=lambda s: (s['P'], s['GD'], s['GF'], rng.random()), reverse=True)
    top8 = sorted_thirds[:8]
    third_by_letter = {t['letter']: t['team'] for t in top8}

    used_letters = set()
    match_winners = {}
    for match_num, slot_a, slot_b in R32_SLOTS:
        ta = slot_to_team.get(slot_a)
        tb = slot_to_team.get(slot_b)
        if ta is None and slot_a.startswith('3'):
            for L in slot_a[1:].split('/'):
                if L in third_by_letter and L not in used_letters:
                    ta = third_by_letter[L]; used_letters.add(L); break
        if tb is None and slot_b.startswith('3'):
            for L in slot_b[1:].split('/'):
                if L in third_by_letter and L not in used_letters:
                    tb = third_by_letter[L]; used_letters.add(L); break
        if ta is None or tb is None:
            for L in third_by_letter:
                if L not in used_letters:
                    if ta is None: ta = third_by_letter[L]; used_letters.add(L); continue
                    if tb is None: tb = third_by_letter[L]; used_letters.add(L); continue
        match_winners[match_num] = simulate_knockout_fast(ta, tb, proba_neutral, rng)

    qualifiers_r16 = set()
    for mn, a, b in R16_PAIRS:
        qualifiers_r16.add(match_winners[a]); qualifiers_r16.add(match_winners[b])
        match_winners[mn] = simulate_knockout_fast(match_winners[a], match_winners[b], proba_neutral, rng)

    qualifiers_qf = set()
    for mn, a, b in QF_PAIRS:
        qualifiers_qf.add(match_winners[a]); qualifiers_qf.add(match_winners[b])
        match_winners[mn] = simulate_knockout_fast(match_winners[a], match_winners[b], proba_neutral, rng)

    qualifiers_sf = set()
    for mn, a, b in SF_PAIRS:
        qualifiers_sf.add(match_winners[a]); qualifiers_sf.add(match_winners[b])
        match_winners[mn] = simulate_knockout_fast(match_winners[a], match_winners[b], proba_neutral, rng)

    finalists = {match_winners[101], match_winners[102]}
    champion = simulate_knockout_fast(match_winners[101], match_winners[102], proba_neutral, rng)
    runner_up = (finalists - {champion}).pop()
    r32_advanced = set(match_winners[m[0]] for m in R32_SLOTS)

    return {
        'champion': champion, 'runner_up': runner_up,
        'semi_finalists': qualifiers_sf, 'quarter_finalists': qualifiers_qf,
        'r16_qualifiers': qualifiers_r16, 'r32_advanced': r32_advanced,
    }


def simulate_tournament_fast(snapshot, model, n_sim=10000, seed=42, verbose=True):
    groups = load_groups_2026()
    teams = groups['team'].tolist()

    t0 = time.time()
    proba_neutral, proba_host = precompute_probas(snapshot, model)
    if verbose:
        print(f"  Precomputed {len(proba_neutral)+len(proba_host)} probas in {time.time()-t0:.2f}s")

    rng = np.random.default_rng(seed)
    counts = {t: {'champion': 0, 'runner_up': 0, 'sf': 0, 'qf': 0, 'r16': 0, 'r32': 0} for t in teams}

    t1 = time.time()
    for i in range(n_sim):
        if verbose and i > 0 and i % 2000 == 0:
            print(f"  iter {i}/{n_sim}  ({time.time()-t1:.1f}s)")
        result = simulate_tournament_once_fast(groups, snapshot, proba_neutral, proba_host, rng)
        counts[result['champion']]['champion'] += 1
        counts[result['runner_up']]['runner_up'] += 1
        for t in result['semi_finalists']: counts[t]['sf'] += 1
        for t in result['quarter_finalists']: counts[t]['qf'] += 1
        for t in result['r16_qualifiers']: counts[t]['r16'] += 1
        for t in result['r32_advanced']: counts[t]['r32'] += 1

    if verbose:
        print(f"  Total: {time.time()-t1:.1f}s para {n_sim} simulaciones")

    rows = [{'team': t, 'p_champion': c['champion']/n_sim,
             'p_final': (c['champion']+c['runner_up'])/n_sim, 'p_sf': c['sf']/n_sim,
             'p_qf': c['qf']/n_sim, 'p_r16': c['r16']/n_sim, 'p_r32': c['r32']/n_sim,
             'n_sim': n_sim} for t, c in counts.items()]
    df = pd.DataFrame(rows).sort_values('p_champion', ascending=False).reset_index(drop=True)
    df['ci_low'], df['ci_high'] = zip(*df.apply(lambda r: wilson_ci(r['p_champion'], r['n_sim']), axis=1))
    return df


def main():
    from src.simulator import prepare_output_path
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sim", type=int, default=10000)
    parser.add_argument("--model", default="xgboost_calibrated")
    parser.add_argument("--out", default="reports/top5_champions.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = prepare_output_path(args.out)

    snapshot_df = pd.read_csv(DATA_PROC / 'teams_2026_snapshot.csv')
    snapshot = {row['team']: {'elo': row['elo'], 'win_rate': row['win_rate'],
                              'gf': row['gf'], 'ga': row['ga']}
                for _, row in snapshot_df.iterrows()}
    model = joblib.load(MODELS_DIR / f"{args.model}.joblib")

    df = simulate_tournament_fast(snapshot, model, n_sim=args.n_sim, seed=args.seed)
    df.to_csv(out_path, index=False)
    print(f"\nOK: {out_path}")
    print("\n=== TOP-10 CAMPEONES ===")
    print(df.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
