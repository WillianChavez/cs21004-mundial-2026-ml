"""
data_loader.py - Lectura y normalizacion de las fuentes en data/raw/.

Funciones publicas:
    load_results()              -> DataFrame martj42/international_results
    load_fifa_ranking()         -> DataFrame samuraitruong/fifa-ranking-data
    load_groups_2026()          -> DataFrame con los 12 grupos del Mundial 2026
    load_bracket_2026()         -> DataFrame con los 40 partidos eliminatorios
    load_baseline_kaggle()      -> DataFrame con probabilidades baseline Kaggle WC2026
    build_master_dataset(save)  -> DataFrame consolidado para modelado (W/D/L label)
    normalize_country(name)     -> str: nombre normalizado de pais
"""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJ_ROOT / "data" / "raw"
DATA_PROC = PROJ_ROOT / "data" / "processed"


# ---------- Normalizacion de nombres de pais ----------
COUNTRY_NORMALIZATION = {
    "South Korea": "Korea Republic",
    "Korea South": "Korea Republic",
    "Iran": "IR Iran",
    "Cape Verde": "Cabo Verde",
    "DR Congo": "Congo DR",
    "Ivory Coast": "Cote d'Ivoire",
    "Czech Republic": "Czechia",
    "Turkey": "Turkiye",
    "United States": "USA",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "Republic of Ireland": "Ireland",
}


def normalize_country(name):
    if not isinstance(name, str):
        return name
    return COUNTRY_NORMALIZATION.get(name.strip(), name.strip())


# ---------- Loaders basicos ----------
def load_results() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])
    df["home_team"] = df["home_team"].map(normalize_country)
    df["away_team"] = df["away_team"].map(normalize_country)
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df


def load_fifa_ranking() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "fifa_ranking_all.csv", parse_dates=["date"])
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"country": "team", "ct": "code"})
    df["team"] = df["team"].map(normalize_country)
    return df


def load_baseline_kaggle() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW / "future_match_probabilities_baseline.csv")
    df["home_team"] = df["home_team"].map(normalize_country)
    df["away_team"] = df["away_team"].map(normalize_country)
    return df


# ---------- Parsers openfootball ----------
_GROUP_LINE_RE = re.compile(r"^Group\s+([A-L])\s*\|\s*(.+?)\s*$")


def load_groups_2026() -> pd.DataFrame:
    """Parsea cup.txt: 12 grupos de 4 equipos cada uno."""
    path = DATA_RAW / "openfootball-worldcup" / "2026--usa" / "cup.txt"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _GROUP_LINE_RE.match(line.strip())
            if not m:
                continue
            group, rest = m.group(1), m.group(2)
            teams = [t.strip() for t in re.split(r"\s{2,}", rest) if t.strip()]
            assert len(teams) == 4, f"Grupo {group} mal parseado: {rest!r} -> {teams}"
            for i, t in enumerate(teams, 1):
                rows.append({"group": group, "team": normalize_country(t), "slot": i})
    return pd.DataFrame(rows)


_BRACKET_RE = re.compile(
    r"^\s*\((\d+)\)\s+[\d:]+\s+UTC\S+\s+(.+?)\s+v\s+(.+?)\s+@\s+(.+?)\s*$"
)
_FINAL_RE = re.compile(
    r"^\s+[\d:]+\s+UTC\S+\s+(\S+)\s+v\s+(\S+)\s+@\s+(.+?)\s*$"
)

ROUND_MAP = {
    "Round of 32": "R32",
    "Round of 16": "R16",
    "Quarter-final": "QF",
    "Semi-final": "SF",
    "Match for third place": "3RD",
    "Final": "FINAL",
}


def load_bracket_2026() -> pd.DataFrame:
    """Parsea cup_finals.txt: bracket 2026 con 40 partidos eliminatorios."""
    path = DATA_RAW / "openfootball-worldcup" / "2026--usa" / "cup_finals.txt"
    rows = []
    current_round = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line_r = line.rstrip()
            stripped = line_r.strip()
            if stripped.startswith("▪"):  # caracter de bullet
                tag = stripped.lstrip("▪").strip()
                current_round = ROUND_MAP.get(tag, tag)
                continue
            m = _BRACKET_RE.match(line_r)
            if m:
                rows.append({
                    "match_num": int(m.group(1)),
                    "round": current_round,
                    "slot_a": m.group(2).strip(),
                    "slot_b": m.group(3).strip(),
                    "stadium": m.group(4).strip(),
                })
                continue
            if current_round in ("3RD", "FINAL"):
                m2 = _FINAL_RE.match(line_r)
                if m2:
                    rows.append({
                        "match_num": 103 if current_round == "3RD" else 104,
                        "round": current_round,
                        "slot_a": m2.group(1).strip(),
                        "slot_b": m2.group(2).strip(),
                        "stadium": m2.group(3).strip(),
                    })
    return pd.DataFrame(rows).sort_values("match_num").reset_index(drop=True)


# ---------- Dataset maestro ----------
def _label_from_score(h, a):
    if pd.isna(h) or pd.isna(a):
        return None
    if h > a: return "W"
    if h < a: return "L"
    return "D"


def _tournament_type(t):
    t = (t or "").lower()
    if "world cup" in t and "qualif" in t:
        return "WC_Qualifier"
    if "world cup" in t:
        return "WC"
    if "friendly" in t:
        return "Friendly"
    keys = ("euro", "copa am", "africa", "asian cup", "concacaf", "afc", "uefa", "conmebol", "gold cup")
    if any(k in t for k in keys):
        return "Continental"
    if "qualif" in t:
        return "Qualifier"
    return "Other"


def build_master_dataset(save: bool = True) -> pd.DataFrame:
    df = load_results().copy()
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["label"] = [_label_from_score(h, a) for h, a in zip(df["home_score"], df["away_score"])]
    df["year"] = df["date"].dt.year
    df["decade"] = (df["year"] // 10) * 10
    df["tournament_type"] = df["tournament"].map(_tournament_type)
    df["goal_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df = df[[
        "date", "year", "decade", "home_team", "away_team",
        "home_score", "away_score", "goal_diff", "total_goals",
        "tournament", "tournament_type", "city", "country", "neutral", "label",
    ]].reset_index(drop=True)
    if save:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(DATA_PROC / "matches_master.parquet", index=False)
        except Exception:
            df.to_csv(DATA_PROC / "matches_master.csv", index=False)
    return df


if __name__ == "__main__":
    print("results:", load_results().shape)
    print("ranking:", load_fifa_ranking().shape)
    g = load_groups_2026()
    print("grupos:", g.shape, "teams unicos:", g["team"].nunique())
    print(g.head(12).to_string())
    b = load_bracket_2026()
    print("bracket:", b.shape)
    print(b.head().to_string())
    m = build_master_dataset(save=True)
    print("master:", m.shape)
    print(m["label"].value_counts(normalize=True))
