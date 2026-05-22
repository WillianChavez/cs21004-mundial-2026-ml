# El Oráculo del Balón — Predicción Mundial FIFA 2026

**Autor:** Willian Alexander Chávez Servellón
**Carnet:** CS21004
**Correo:** cs21004@ues.edu.sv
**Institución:** Universidad de El Salvador, Especialización en Inteligencia Artificial

---

## 📄 Reporte técnico (entregable principal)

El reporte técnico tipo IEEE está en la **raíz del proyecto**, con este nombre:

> **`Reporte_Mundial2026_CS21004.docx`**

Es el documento que contiene el EDA, la metodología, los experimentos, las figuras y las conclusiones. Está en la raíz (no dentro de subcarpetas) para que sea fácil de localizar.

---

Pipeline de Machine Learning para estimar la probabilidad de que cada una de las **48 selecciones** del Mundial 2026 se corone campeona, mediante:

1. Un **clasificador a nivel de partido** (Gana / Empata / Pierde) entrenado sobre el histórico de partidos internacionales 1872–2025.
2. Una **simulación Monte Carlo (≥10 000 iteraciones)** del torneo completo respetando el formato oficial 2026 (12 grupos de 4, 32 equipos a octavos, etc.).
3. **Métricas de calibración** (Log-Loss y Brier Score) y validación retrospectiva sobre el Mundial 2022.

## Estructura del repositorio

```
desafio-practico-01/
├── Reporte_Mundial2026_CS21004.docx   # >>> REPORTE TÉCNICO (entregable principal) <<<
├── data/
│   ├── raw/             # Datasets crudos descargados (no modificar)
│   ├── processed/       # Datasets construidos (features, train/test)
│   └── external/        # Reservado para descargas auxiliares
├── notebooks/
│   ├── 01_eda.ipynb         # Análisis Exploratorio
│   ├── 02_features.ipynb    # Feature engineering + decay temporal
│   ├── 03_modeling.ipynb    # Entrenamiento + tuning
│   └── 04_montecarlo.ipynb  # Simulación del torneo
├── src/
│   ├── data_loader.py       # Lectura y normalización de fuentes
│   ├── features.py          # ELO, decay, plantilla, etc.
│   ├── model.py             # XGBoost + Logistic Regression
│   ├── simulator.py         # Monte Carlo del bracket 2026
│   └── evaluation.py        # Log-Loss, Brier, calibración
├── reports/                 # Resultados en CSV (top5, métricas, importancias)
├── models/                  # Modelos serializados (.joblib)
├── figures/                 # Gráficas de EDA y resultados
├── tests/                   # Tests unitarios mínimos
├── requirements.txt
├── .gitignore
└── README.md
```

## Datasets ya descargados (en `data/raw/`)

| Archivo | Filas | Fuente | Uso |
|---|---|---|---|
| `results.csv` | 49 329 | [martj42/international_results](https://github.com/martj42/international_results) | Partidos internacionales 1872–2026 |
| `goalscorers.csv` | 47 601 | martj42 | Goleadores por partido |
| `shootouts.csv` | 677 | martj42 | Ganadores en tandas de penales |
| `fifa_ranking_all.csv` | 60 544 | [samuraitruong/fifa-ranking-data](https://github.com/samuraitruong/fifa-ranking-data) | Histórico FIFA Ranking 1992–2019 |
| `fifa_countries.csv` | 210 | samuraitruong | Catálogo de países y códigos |
| `openfootball-worldcup/` | 23 ediciones | [openfootball/worldcup](https://github.com/openfootball/worldcup) | Fixture oficial 1930–2026 (incluye grupos 2026) |
| `future_match_probabilities_baseline.csv` | 72 | [Kaggle WC2026 Baseline](https://www.kaggle.com/) | Probabilidades Elo baseline para los 72 partidos de fase de grupos 2026 |

**Pendiente / a calcular en pipeline:**

- ELO ratings de selecciones (se calcula desde `results.csv` con la fórmula de [eloratings.net](https://www.eloratings.net/)).
- Plantillas y valores de mercado (Transfermarkt — opcional).
- Variables socioeconómicas para debutantes (PIB per cápita, población, etc.).

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cómo reproducir

```bash
# 1. EDA
jupyter notebook notebooks/01_eda.ipynb

# 2. Construir features
python -m src.features

# 3. Entrenar modelo
python -m src.model

# 4. Correr Monte Carlo (10 000 simulaciones)
python -m src.simulator --n_sim 10000 --out reports/top5.csv
```

## Decisiones técnicas clave

- **Granularidad del entrenamiento**: a nivel de **partido** (clase=Gana/Empata/Pierde desde la perspectiva del equipo local) — evita el problema de desbalance "campeón vs no campeón".
- **Decay temporal**: peso = `exp(-λ · años)` con `λ = ln(2)/4` → vida media = 4 años. Un partido de hace 4 años pesa 0.5 respecto a uno reciente.
- **Modelos comparados**: Logistic Regression multinomial (baseline interpretable) y XGBoost multinomial con `RandomizedSearchCV`.
- **Métricas primarias**: Log-Loss (multi-clase) + Brier Score multi-clase. Accuracy se reporta solo como referencia.
- **Validación temporal**: split train ≤ 2018, test 2019–2024 (incluye Mundial 2022 como holdout para validar el pipeline completo).
- **Monte Carlo**: 10 000 iteraciones; en cada una se sortea cada partido con `np.random.choice(['W','D','L'], p=model_proba)`; los terceros mejores se rankean según puntos > goles > GD; bracket exacto del archivo `2026--usa/cup_finals.txt`.

## Resultados principales

Top-5 candidatos a campeón (10 000 simulaciones Monte Carlo, con intervalos Wilson 95%): España (16.3%), Argentina (15.9%), Francia (11.2%), Inglaterra (6.7%) y Brasil (5.4%). El detalle metodológico y los experimentos están en el reporte técnico `Reporte_Mundial2026_CS21004.docx` (en la raíz del proyecto).

## Limitaciones conocidas

- El ranking FIFA disponible llega hasta 2019. Para 2020–2026 lo aproximamos con ELO calculado.
- Los datos previos a 2010 no incluyen xG. Se omite esa feature o se imputa con tasa de conversión histórica.
- El modelo no observa lesiones de último momento ni cambios de DT recientes — esto se discute en la sección "Limitaciones" del reporte.

## Licencia

Uso académico (curso de Especialización). Datasets bajo sus licencias originales.
