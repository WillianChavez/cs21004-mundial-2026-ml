"""
simulator.py - Monte Carlo del Mundial 2026.
"""
from __future__ import annotations
import argparse
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings

# Los modelos .joblib se serializaron con una version concreta de scikit-learn.
# Si se cargan con otra version, sklearn emite InconsistentVersionWarning (inofensivo
# aqui: los resultados son identicos). Lo silenciamos para no ensuciar la salida.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC = PROJ_ROOT / "data" / "processed"
MODELS_DIR = PROJ_ROOT / "models"

POINTS_W, POINTS_D, POINTS_L = 3, 1, 0

# Feature columns en el orden esperado por el modelo (igual que en X_train)
FEATURE_COLS = [
    "elo_pre_home", "elo_pre_away", "elo_diff", "elo_expected_home",
    "home_win_rate_10", "home_gf_10", "home_ga_10",
    "away_win_rate_10", "away_gf_10", "away_ga_10",
    "h2h_home_win_rate", "h2h_avg_goal_diff", "h2h_count",
    "is_home_advantage",
    "tt_Continental", "tt_Friendly", "tt_Other", "tt_Qualifier", "tt_WC", "tt_WC_Qualifier",
]

HOST_COUNTRIES = {"USA", "Mexico", "Canada"}


def build_match_features(team_a: dict, team_b: dict, is_wc: bool = True,
                         host_neutral: bool = True, h2h_lookup=None) -> np.ndarray:
    """Construye el vector de features para un partido futuro.

    Args:
        team_a, team_b: dicts con {elo, win_rate, gf, ga}
        is_wc: True si es Mundial (siempre True aqui)
        host_neutral: True si la cancha es neutral (eliminatorias) o False si team_a es anfitrion en grupos
        h2h_lookup: dict opcional {(team_a_name, team_b_name): (win_rate, goal_diff, count)}
    """
    elo_a, elo_b = team_a['elo'], team_b['elo']
    home_adv = 0.0 if host_neutral else 100.0
    expected = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a - home_adv) / 400.0))
    is_home_adv = 0 if host_neutral else 1

    if h2h_lookup is not None:
        h2h_wr, h2h_gd, h2h_cnt = h2h_lookup
    else:
        h2h_wr, h2h_gd, h2h_cnt = 0.5, 0.0, 0

    feats = {
        "elo_pre_home": elo_a, "elo_pre_away": elo_b,
        "elo_diff": elo_a - elo_b, "elo_expected_home": expected,
        "home_win_rate_10": team_a['win_rate'], "home_gf_10": team_a['gf'], "home_ga_10": team_a['ga'],
        "away_win_rate_10": team_b['win_rate'], "away_gf_10": team_b['gf'], "away_ga_10": team_b['ga'],
        "h2h_home_win_rate": h2h_wr, "h2h_avg_goal_diff": h2h_gd, "h2h_count": h2h_cnt,
        "is_home_advantage": is_home_adv,
        "tt_Continental": 0, "tt_Friendly": 0, "tt_Other": 0,
        "tt_Qualifier": 0, "tt_WC": 1 if is_wc else 0, "tt_WC_Qualifier": 0,
    }
    return np.array([feats[c] for c in FEATURE_COLS])


def predict_proba_batch(model, X) -> np.ndarray:
    """Wrapper. Devuelve matriz n x 3 con P(W,D,L)."""
    return model.predict_proba(X)


# ---------- Simulacion partido ----------
def sample_outcome(p_w: float, p_d: float, p_l: float, rng) -> int:
    """0=W home, 1=D, 2=L. Renormaliza para evitar errores numericos."""
    p = np.array([p_w, p_d, p_l])
    p = np.clip(p, 1e-9, 1.0)
    p /= p.sum()
    return int(rng.choice(3, p=p))


def sample_outcome_no_draw(p_w: float, p_d: float, p_l: float, rng) -> int:
    """Para eliminatorias: redistribuye empate 50/50 entre W y L."""
    p_w2 = p_w + 0.5 * p_d
    p_l2 = p_l + 0.5 * p_d
    s = p_w2 + p_l2
    return 0 if rng.random() < p_w2 / s else 2


# ---------- Simulacion grupo ----------
def simulate_group_round_robin(group_teams, snapshot, model, host_country=None, rng=None):
    """Simula los 6 partidos de un grupo round-robin. Devuelve standings ordenado.

    Args:
        group_teams: lista de 4 nombres de equipo
        snapshot: dict {team: {elo, win_rate, gf, ga}}
        model: clasificador con predict_proba
        host_country: nombre del anfitrion (si juega aqui, no neutral)
        rng: np.random.Generator
    """
    stats = {t: {'P': 0, 'GD': 0, 'GF': 0, 'team': t} for t in group_teams}
    # Generar los 6 partidos: cada par juega 1 vez
    matches = []
    for i in range(4):
        for j in range(i+1, 4):
            matches.append((group_teams[i], group_teams[j]))

    feats_list = []
    host_list = []
    for a, b in matches:
        is_host_a = (a == host_country)
        is_host_b = (b == host_country)
        host_neutral = not (is_host_a or is_host_b)
        # Asignar 'home' al que sea anfitrion, sino al primer alfabetico (neutral)
        if is_host_b and not is_host_a:
            a, b = b, a
            host_neutral = False
        feats_list.append(build_match_features(snapshot[a], snapshot[b],
                                               is_wc=True, host_neutral=host_neutral))
        host_list.append((a, b))
    X = np.stack(feats_list)
    P = model.predict_proba(X)

    # Resultados estocásticos
    for (a, b), p in zip(host_list, P):
        outcome = sample_outcome(p[0], p[1], p[2], rng)
        # Estimar marcador para tiebreaks: usar Poisson con rate ~gf/ga
        # Simplificación: muestra 1 gol al ganador, 0 al perdedor; o 1-1 si empate.
        # Para tiebreaks usaremos goles aleatorios mas reales:
        lambda_a = max(snapshot[a]['gf'], 0.3)
        lambda_b = max(snapshot[b]['gf'], 0.3)
        ga = int(rng.poisson(lambda_a))
        gb = int(rng.poisson(lambda_b))
        if outcome == 0:  # a gana
            if ga <= gb: ga = gb + 1
        elif outcome == 2:  # b gana
            if gb <= ga: gb = ga + 1
        else:  # empate
            ga = gb = max(ga, gb)
        # Actualizar stats
        if outcome == 0:
            stats[a]['P'] += POINTS_W; stats[b]['P'] += POINTS_L
        elif outcome == 2:
            stats[a]['P'] += POINTS_L; stats[b]['P'] += POINTS_W
        else:
            stats[a]['P'] += POINTS_D; stats[b]['P'] += POINTS_D
        stats[a]['GD'] += ga - gb; stats[b]['GD'] += gb - ga
        stats[a]['GF'] += ga; stats[b]['GF'] += gb

    # Ranking: P → GD → GF → aleatorio
    standings = sorted(stats.values(),
                       key=lambda s: (s['P'], s['GD'], s['GF'], rng.random()),
                       reverse=True)
    return standings


# ---------- Bracket eliminatorio ----------
def best_third_places(thirds, rng):
    """De los 12 terceros, devuelve los 8 mejores ordenados."""
    sorted_t = sorted(thirds,
                      key=lambda s: (s['P'], s['GD'], s['GF'], rng.random()),
                      reverse=True)
    return sorted_t[:8]


def simulate_knockout_match(team_a, team_b, snapshot, model, rng):
    """Simula un partido eliminatorio (sin empate). Devuelve ganador."""
    feats = build_match_features(snapshot[team_a], snapshot[team_b],
                                 is_wc=True, host_neutral=True).reshape(1, -1)
    p = model.predict_proba(feats)[0]
    out = sample_outcome_no_draw(p[0], p[1], p[2], rng)
    return team_a if out == 0 else team_b


# El bracket exacto de R32 (segun cup_finals.txt)
# Slots referenciados: 1X = winner group X, 2X = runner-up X, 3W/X/Y/Z = third from one of W,X,Y,Z
R32_SLOTS = [
    # (match_num, slot_a, slot_b)
    (73, '2A', '2B'),
    (74, '1E', '3A/B/C/D/F'),
    (75, '1F', '2C'),
    (76, '1C', '2F'),
    (77, '1I', '3C/D/F/G/H'),
    (78, '2E', '2I'),
    (79, '1A', '3C/E/F/H/I'),
    (80, '1L', '3E/H/I/J/K'),
    (81, '1D', '3B/E/F/I/J'),
    (82, '1G', '3A/E/H/I/J'),
    (83, '2K', '2L'),
    (84, '1H', '2J'),
    (85, '1B', '3E/F/G/I/J'),
    (86, '1J', '2H'),
    (87, '1K', '3D/E/I/J/L'),
    (88, '2D', '2G'),
]

# R16: empareja ganadores de R32 segun cup_finals.txt
R16_PAIRS = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
]

QF_PAIRS = [
    (97, 89, 90),
    (98, 93, 94),
    (99, 91, 92),
    (100, 95, 96),
]

SF_PAIRS = [
    (101, 97, 98),
    (102, 99, 100),
]


def assign_third_to_match(third_options, available_thirds, used_thirds):
    """De los terceros disponibles (entre las letras dadas), toma el primero no usado.

    third_options: ej "A/B/C/D/F" -> ['A','B','C','D','F']
    available_thirds: dict {group_letter: standings_3rd_place_team_dict} de los 8 mejores
    used_thirds: set de letras ya usadas
    """
    opts = third_options.split('/')
    for letter in opts:
        if letter in available_thirds and letter not in used_thirds:
            used_thirds.add(letter)
            return available_thirds[letter]
    # Si ningun tercero de los preferidos esta entre los 8 mejores, asignar el siguiente disponible
    for letter in available_thirds:
        if letter not in used_thirds:
            used_thirds.add(letter)
            return available_thirds[letter]
    return None


def simulate_tournament_once(groups_df, snapshot, model, rng):
    """Simula UN torneo entero. Devuelve dict con resultados clave."""
    # 1. Simular 12 grupos
    standings_by_group = {}
    for group_letter in groups_df['group'].unique():
        teams = groups_df[groups_df['group'] == group_letter]['team'].tolist()
        # Detectar host
        host = None
        for t in teams:
            if t in HOST_COUNTRIES:
                host = t
                break
        standings = simulate_group_round_robin(teams, snapshot, model, host_country=host, rng=rng)
        standings_by_group[group_letter] = standings

    # 2. Construir mapa de slots
    slot_to_team = {}
    thirds = []
    for letter, standing in standings_by_group.items():
        slot_to_team[f'1{letter}'] = standing[0]['team']
        slot_to_team[f'2{letter}'] = standing[1]['team']
        thirds.append({'letter': letter, **standing[2]})

    # 3. Mejores 8 terceros
    sorted_thirds = sorted(thirds, key=lambda s: (s['P'], s['GD'], s['GF'], rng.random()), reverse=True)
    top8_thirds = sorted_thirds[:8]
    third_by_letter = {t['letter']: t['team'] for t in top8_thirds}

    # 4. Resolver slots de terceros en R32
    used_letters = set()
    match_winners = {}
    for match_num, slot_a, slot_b in R32_SLOTS:
        team_a = slot_to_team.get(slot_a) or third_by_letter.get(slot_a, None)
        team_b = slot_to_team.get(slot_b) or third_by_letter.get(slot_b, None)
        # Si slot es del tipo '3A/B/C/D/F'
        if team_a is None and slot_a.startswith('3'):
            opts = slot_a[1:].split('/')
            for L in opts:
                if L in third_by_letter and L not in used_letters:
                    team_a = third_by_letter[L]
                    used_letters.add(L)
                    break
        if team_b is None and slot_b.startswith('3'):
            opts = slot_b[1:].split('/')
            for L in opts:
                if L in third_by_letter and L not in used_letters:
                    team_b = third_by_letter[L]
                    used_letters.add(L)
                    break
        if team_a is None or team_b is None:
            # Fallback: cualquier tercero no usado
            for L in third_by_letter:
                if L not in used_letters:
                    if team_a is None: team_a = third_by_letter[L]; used_letters.add(L); continue
                    if team_b is None: team_b = third_by_letter[L]; used_letters.add(L); continue
        winner = simulate_knockout_match(team_a, team_b, snapshot, model, rng)
        match_winners[match_num] = winner

    # 5. R16, QF, SF, Final
    qualifiers_r16 = set()
    for mn, a, b in R16_PAIRS:
        w = simulate_knockout_match(match_winners[a], match_winners[b], snapshot, model, rng)
        match_winners[mn] = w
        qualifiers_r16.add(match_winners[a]); qualifiers_r16.add(match_winners[b])

    qualifiers_qf = set()
    for mn, a, b in QF_PAIRS:
        w = simulate_knockout_match(match_winners[a], match_winners[b], snapshot, model, rng)
        match_winners[mn] = w
        qualifiers_qf.add(match_winners[a]); qualifiers_qf.add(match_winners[b])

    qualifiers_sf = set()
    for mn, a, b in SF_PAIRS:
        w = simulate_knockout_match(match_winners[a], match_winners[b], snapshot, model, rng)
        match_winners[mn] = w
        qualifiers_sf.add(match_winners[a]); qualifiers_sf.add(match_winners[b])

    finalists = {match_winners[101], match_winners[102]}
    champion = simulate_knockout_match(match_winners[101], match_winners[102], snapshot, model, rng)
    runner_up = (finalists - {champion}).pop()

    # Recolectar los 32 clasificados a R16 (avanzaron del R32)
    r32_advanced = set(match_winners[m[0]] for m in R32_SLOTS)

    return {
        'champion': champion,
        'runner_up': runner_up,
        'semi_finalists': qualifiers_sf,
        'quarter_finalists': qualifiers_qf,
        'r16_qualifiers': qualifiers_r16,
        'r32_advanced': r32_advanced,
    }


def simulate_tournament(snapshot, model, n_sim=10000, seed=42):
    """Corre n_sim simulaciones. Devuelve DataFrame con stats por equipo."""
    from src.data_loader import load_groups_2026
    groups = load_groups_2026()
    teams = groups['team'].tolist()

    rng = np.random.default_rng(seed)
    counts = {t: {'champion': 0, 'runner_up': 0, 'sf': 0, 'qf': 0, 'r16': 0, 'r32': 0} for t in teams}

    for i in range(n_sim):
        if i % 1000 == 0 and i > 0:
            print(f"  iter {i}/{n_sim}")
        result = simulate_tournament_once(groups, snapshot, model, rng)
        counts[result['champion']]['champion'] += 1
        counts[result['runner_up']]['runner_up'] += 1
        for t in result['semi_finalists']: counts[t]['sf'] += 1
        for t in result['quarter_finalists']: counts[t]['qf'] += 1
        for t in result['r16_qualifiers']: counts[t]['r16'] += 1
        for t in result['r32_advanced']: counts[t]['r32'] += 1

    rows = []
    for t, c in counts.items():
        rows.append({
            'team': t,
            'p_champion': c['champion'] / n_sim,
            'p_final': (c['champion'] + c['runner_up']) / n_sim,
            'p_sf': c['sf'] / n_sim,
            'p_qf': c['qf'] / n_sim,
            'p_r16': c['r16'] / n_sim,
            'p_r32': c['r32'] / n_sim,
            'n_sim': n_sim,
        })
    df = pd.DataFrame(rows).sort_values('p_champion', ascending=False).reset_index(drop=True)
    return df


def wilson_ci(p, n, z=1.96):
    """Intervalo de confianza Wilson 95%."""
    if n == 0: return (0.0, 0.0)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def prepare_output_path(out: str) -> Path:
    """Crea la carpeta de salida si no existe y borra el archivo previo.

    Asi, si el archivo ya existe, se limpia y se regenera sin errores.
    """
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
        print(f"Archivo previo encontrado, se elimina para regenerar: {out_path}")
    return out_path


def main():
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

    print(f"Corriendo {args.n_sim} simulaciones con modelo={args.model}...")
    df = simulate_tournament(snapshot, model, n_sim=args.n_sim, seed=args.seed)
    df["ci_low"], df["ci_high"] = zip(*df.apply(lambda r: wilson_ci(r["p_champion"], r["n_sim"]), axis=1))
    df.to_csv(out_path, index=False)
    print(f"\nOK: {out_path}")
    print("\n=== TOP-10 CAMPEONES ===")
    print(df.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
