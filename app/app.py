"""
F1 Race Intelligence — Databricks App (Plans B, G, E).

Three tabs over three separately-trained model sets:
- Strategy Simulator (Plan B): pick a driver, team, circuit, and pit-stop
  count, get a predicted stint-by-stint tyre strategy and win probability.
  Models from notebooks/phase_3_strategy.ipynb.
- Race Replay (Plans E, G): replay a finished race lap by lap as a full
  20-car grid of cards, a battle board, tyre and position charts, and key
  moments. Reads the precomputed f1_replay_lap_state / f1_battle_pairs /
  f1_battle_bands tables from notebooks/phase_3_battles.ipynb; podium and
  winner chances come from f1_live_predictions (notebooks/phase_3_live.ipynb).

Runs as a lightweight Streamlit web service, separate from any notebook.
It gets a real Spark session via Databricks Connect (serverless) rather
than depending on a notebook's native `spark` object — an App has no
notebook to inherit one from. Model artifacts come from the Unity Catalog
Volume /Volumes/workspace/default/f1_data (strategy_models/,
battle_models/) and the f1_strategy_*/f1_circuit_overtake_prior tables — workspace-relative paths like
"../models/" (the notebooks' own convenience copies) aren't reachable
from this App's separate container at all.

UI theme: mirrors F1 broadcast-graphics language (Titillium Web type,
brand red on near-black, angular/skewed panels, Pirelli compound colors)
via .streamlit/config.toml (native widget theming) plus the CSS block
below (layout, timeline, gauge, ticker, and motion this app needs that the
config-level theme can't reach). See CLAUDE.md if the fonts CDN turns
out to be blocked by the workspace's outbound-network policy — same
class of issue as the FastF1 rate-limit/network diagnosis in phase_1;
the CSS falls back to system sans-serif if the Google Fonts request
fails, so the app still functions either way.
"""

import html
import json
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from databricks.connect.session import DatabricksSession
from pyspark.sql import functions as F
from xgboost import XGBClassifier

import future as fu
import replay as rp

STRATEGY_VOLUME_DIR = "/Volumes/workspace/default/f1_data/strategy_models"
CAT_FEATURES = ["Circuit", "TeamName", "Driver"]
TYRE_LIFE_LIMITS = {"SOFT": 25, "MEDIUM": 40, "HARD": 55, "INTERMEDIATE": 30, "WET": 40}

# Pirelli / F1 broadcast-graphic compound colors.
COMPOUND_COLORS = {
    "SOFT": "#FF1E1E",
    "MEDIUM": "#FFD400",
    "HARD": "#F5F5F5",
    "INTERMEDIATE": "#43B02A",
    "WET": "#0067B1",
}
# Dark text reads better on the light SOFT/HARD... no: MEDIUM (yellow) and HARD (white) chips.
COMPOUND_TEXT_COLORS = {
    "SOFT": "#FFFFFF",
    "MEDIUM": "#15151E",
    "HARD": "#15151E",
    "INTERMEDIATE": "#FFFFFF",
    "WET": "#FFFFFF",
}


class NumericLogit:
    # logistic regression over the scaled numeric block (first n_num columns) of the shared feature vector
    def __init__(self, coef, intercept, n_num):
        self.coef_ = np.asarray(coef, dtype=float)
        self.intercept_ = float(intercept)
        self.n_num = int(n_num)
        self.feature_importances_ = np.abs(self.coef_) / np.abs(self.coef_).sum()

    def predict_proba(self, X):
        z = np.nan_to_num(np.asarray(X, dtype=float)[:, :self.n_num]) @ self.coef_ + self.intercept_
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


@st.cache_resource
def load_backend():
    """Spark session + trained models — expensive, loaded once per app instance."""
    spark = DatabricksSession.builder.serverless().getOrCreate()

    # NOT loaded via pyspark.ml.PipelineModel.load(): that goes through
    # pyspark.ml's classic JVM-backed reader (_jvm() -> needs a real
    # SparkContext), which a Databricks App's Spark Connect session
    # (databricks-connect) has no JVM gateway to reach --
    # "AttributeError: Cannot load _jvm from SparkContext". Instead this
    # loads the plain-JSON params phase_3_strategy.ipynb exports from the
    # fitted pipeline (StringIndexer label orderings + StandardScaler
    # mean/std), and _build_feature_vector() below reimplements the same
    # StringIndexer -> OneHotEncoder -> VectorAssembler -> StandardScaler
    # transform in pure NumPy -- verified to match the real fitted
    # pipeline's output in that notebook before being trusted here.
    # Read JSON file through Spark (Apps don't have direct /Volumes/ filesystem access)
    preproc_params_text = spark.read.text(f"{STRATEGY_VOLUME_DIR}/preproc_params.json", wholetext=True).first()[0]
    preproc_params = json.loads(preproc_params_text)

    # Load XGBoost models and JSON via Spark, write to temp files
    tmpdir = tempfile.mkdtemp()

    # Strategy models
    for model_file in ["m1_compound_classifier.json"]:
        model_text = spark.read.text(f"{STRATEGY_VOLUME_DIR}/{model_file}", wholetext=True).first()[0]
        with open(os.path.join(tmpdir, model_file), 'w') as f:
            f.write(model_text)
    
    m1 = XGBClassifier()
    m1.load_model(os.path.join(tmpdir, "m1_compound_classifier.json"))

    m3_text = spark.read.text(f"{STRATEGY_VOLUME_DIR}/m3_win_logit.json", wholetext=True).first()[0]
    m3 = NumericLogit(**json.loads(m3_text))

    compound_label_text = spark.read.text(f"{STRATEGY_VOLUME_DIR}/compound_label_map.json", wholetext=True).first()[0]
    compound_idx_to_name = {int(k): v for k, v in json.loads(compound_label_text).items()}

    driver_profile = spark.table("workspace.default.f1_strategy_driver_profile")
    constructor_profile = spark.table("workspace.default.f1_strategy_constructor_profile")
    circuit_profile = spark.table("workspace.default.f1_strategy_circuit_profile")
    cleaned_laps = spark.table("workspace.default.f1_cleaned_lap_dataset")

    return {
        "spark": spark,
        "preproc_params": preproc_params,
        "m1": m1,
        "m3": m3,
        "compound_idx_to_name": compound_idx_to_name,
        "driver_profile": driver_profile,
        "constructor_profile": constructor_profile,
        "circuit_profile": circuit_profile,
        "cleaned_laps": cleaned_laps,
    }


@st.cache_data(ttl=3600)
def load_options():
    """Distinct driver/team/circuit values for the form dropdowns."""
    backend = load_backend()
    drivers = sorted(r["Driver"] for r in backend["driver_profile"].select("Driver").distinct().collect())
    teams = sorted(r["TeamName"] for r in backend["constructor_profile"].select("TeamName").distinct().collect())
    circuits = sorted(r["Circuit"] for r in backend["circuit_profile"].select("Circuit").distinct().collect())
    years = sorted(
        r["Year"] for r in backend["driver_profile"].select("Year").distinct().collect()
    )
    return drivers, teams, circuits, years


@st.cache_data(ttl=3600)
def load_circuit_typical_laps():
    """Circuit -> typical race distance in laps (max laps completed by any
    driver at that circuit across training data -- see CircuitTypicalLaps
    in phase_3_strategy.ipynb's circuit_profile). Used to default the
    Strategy Simulator's Race Distance field per circuit instead of a flat
    manual guess disconnected from which track is selected.
    """
    backend = load_backend()
    rows = backend["circuit_profile"].select("Circuit", "CircuitTypicalLaps").collect()
    return {r["Circuit"]: int(r["CircuitTypicalLaps"]) for r in rows if r["CircuitTypicalLaps"] is not None}


@st.cache_data(ttl=3600)
def load_races():
    """(Year, Circuit) -> race names, since a circuit can host 2 races in one season."""
    rows = load_backend()["spark"].table("workspace.default.f1_replay_lap_state").select("Year", "Circuit", "Race").distinct().collect()
    races = {}
    for r in rows:
        races.setdefault((r["Year"], r["Circuit"]), []).append(r["Race"])
    return {k: sorted(v) for k, v in races.items()}


@st.cache_resource(ttl=3600)
def load_future_model():
    spark = load_backend()["spark"]
    t = lambda n: spark.table(f"workspace.default.{n}").toPandas()
    return fu.FutureModel(t("f1_podium_model")["Json"].iloc[0], t("f1_podium_driver_form"), t("f1_podium_team_form"), t("f1_podium_circuits"), t("f1_podium_recent_grids"))


@st.cache_resource(ttl=3600, max_entries=6)
def load_race_view(year, race):
    spark = load_backend()["spark"]
    by_race = (F.col("Year") == int(year)) & (F.col("Race") == race)
    state = spark.table("workspace.default.f1_replay_lap_state").filter(by_race).toPandas()
    pairs = spark.table("workspace.default.f1_battle_pairs").filter(by_race).toPandas()
    bands = spark.table("workspace.default.f1_battle_bands").toPandas()
    return rp.RaceView(state, pairs, bands)


def pick_race(year, circuit, key):
    # Race dropdown, only shown when the season ran 2+ races at this circuit
    options = load_races().get((int(year), circuit), [])
    if len(options) > 1:
        return st.selectbox("Race", options, key=key)
    return options[0] if options else None


def _build_feature_vector(row_dict, driver, team, circuit, preproc_params):
    """Pure-NumPy reimplementation of the fitted Spark preprocessing
    pipeline's transform (StringIndexer -> OneHotEncoder -> VectorAssembler
    -> StandardScaler) -- see phase_3_strategy.ipynb's export/verify cells
    for why this exists and how it was checked against the real pipeline.
    """
    cat_values = {"Circuit": circuit, "TeamName": team, "Driver": driver}
    vec = [float(row_dict[nf]) for nf in preproc_params["numeric_features"]]
    for c in preproc_params["cat_features"]:
        labels = preproc_params["cat_labels"][c]
        n = len(labels)
        # Width is n, not n-1: the StringIndexer behind these labels uses
        # handleInvalid="keep", which reserves an extra "unseen" bucket in
        # the category-count metadata beyond these n real labels.
        # OneHotEncoder's default dropLast=True drops THAT phantom bucket,
        # not one of the n real labels, so every real label gets its own
        # column (verified against the real fitted pipeline in
        # phase_3_strategy.ipynb's export/verify cells).
        oh = [0.0] * n
        val = cat_values[c]
        if val in labels:
            oh[labels.index(val)] = 1.0
        # Unseen categories fall through as all-zero -- that's also the
        # correct behavior: an unseen value would map to the phantom
        # bucket, which is exactly the one OneHotEncoder drops.
        vec.extend(oh)
    x = np.array(vec, dtype=float)
    mean = np.array(preproc_params["scaler_mean"])
    std = np.array(preproc_params["scaler_std"])
    out = np.where(std != 0, (x - mean) / np.where(std == 0, 1.0, std), 0.0)
    return out.reshape(1, -1)


def predict_strategy(
    driver: str,
    team: str,
    year: int,
    circuit: str,
    quali_position: int,
    total_laps: int = 57,
    num_pit_stops: int = 1,
) -> dict:
    """
    Same modeling logic as phase_4.ipynb's predict_strategy (adapted from
    phase_3_strategy.ipynb) — kept in sync manually since an App can't
    import a notebook. One deliberate divergence: preprocessing here uses
    _build_feature_vector() (pure NumPy) instead of the Spark ML
    PipelineModel phase_4.ipynb uses, since pyspark.ml.PipelineModel.load()
    doesn't work from this App's Spark-Connect-only session (see
    load_backend()'s comment). Same output contract either way.
    Returns {"strategy": [...], "win_probability": float}.
    """
    backend = load_backend()
    preproc_params = backend["preproc_params"]
    m1, m3 = backend["m1"], backend["m3"]
    compound_idx_to_name = backend["compound_idx_to_name"]
    driver_profile = backend["driver_profile"]
    constructor_profile = backend["constructor_profile"]
    circuit_profile = backend["circuit_profile"]
    cleaned_laps = backend["cleaned_laps"]

    def get_profile(sdf_local, col, val, year_local):
        row = sdf_local.filter((F.col(col) == val) & (F.col("Year") == year_local)).first()
        if row is None:
            row = sdf_local.filter(F.col(col) == val).orderBy(F.desc("Year")).first()
        return row

    def safe(row, key, default=10.0):
        try:
            return float(row[key]) if row and row[key] is not None else default
        except Exception:
            return default

    drv_row = get_profile(driver_profile, "Driver", driver, year)
    car_row = get_profile(constructor_profile, "TeamName", team, year)
    cir_row = circuit_profile.filter(F.col("Circuit") == circuit).first()

    drv_avg_pos = safe(drv_row, "DriverAvgPos", 10.0)
    drv_avg_quali = safe(drv_row, "DriverAvgQuali", quali_position)
    drv_win_rate = safe(drv_row, "DriverWinRate", 0.05)
    drv_pts_rate = safe(drv_row, "DriverPointsRate", 0.5)
    drv_pod_rate = safe(drv_row, "DriverPodiumRate", 0.15)
    drv_consist = safe(drv_row, "DriverConsistency", 4.0)
    car_avg_pos = safe(car_row, "CarAvgPos", 10.0)
    car_avg_quali = safe(car_row, "CarAvgQuali", quali_position)
    car_win_rate = safe(car_row, "CarWinRate", 0.05)
    car_pod_rate = safe(car_row, "CarPodiumRate", 0.15)
    car_pts_rate = safe(car_row, "CarPointsRate", 0.5)
    cir_avg_comp = safe(cir_row, "CircuitAvgCompoundRank", 3.0)
    cir_avg_temp = safe(cir_row, "CircuitAvgTrackTemp", 30.0)

    year_drivers = (
        cleaned_laps.filter(F.col("Year") == year)
        .select("Driver", "QualiPosition")
        .dropDuplicates()
    )
    threat_df = year_drivers.filter(F.col("QualiPosition") < quali_position)

    threat_score = 0.0
    max_threat = 0.0
    for t_row in threat_df.collect():
        t_drv = get_profile(driver_profile, "Driver", t_row["Driver"], year)
        t_wr = safe(t_drv, "DriverWinRate", 0.0)
        threat_score += t_wr
        max_threat = max(max_threat, t_wr)

    strategy = []
    current_lap = 1
    current_pos = quali_position
    total_stints = num_pit_stops + 1
    prev_compound = None
    row_dict = {}

    for stint_num in range(1, total_stints + 1):
        lap_frac = current_lap / total_laps

        row_dict = {
            "Stint": stint_num,
            "StintStartLap": current_lap,
            "StintStartFrac": lap_frac,
            "QualiPosition": quali_position,
            "TotalLaps": total_laps,
            "NumStints": total_stints,
            "AirTemp_C": 22.0,
            "TrackTemp_C": cir_avg_temp,
            "Humidity_pct": 40.0,
            "WindSpeed_kmh": 2.0,
            "DriverAvgPos": drv_avg_pos,
            "DriverAvgQuali": drv_avg_quali,
            "DriverWinRate": drv_win_rate,
            "DriverPointsRate": drv_pts_rate,
            "DriverPodiumRate": drv_pod_rate,
            "DriverConsistency": drv_consist,
            "CarAvgPos": car_avg_pos,
            "CarAvgQuali": car_avg_quali,
            "CarWinRate": car_win_rate,
            "CarPodiumRate": car_pod_rate,
            "CarPointsRate": car_pts_rate,
            "CircuitAvgCompoundRank": cir_avg_comp,
            "CircuitAvgStints": total_stints,
            "CircuitAvgTrackTemp": cir_avg_temp,
            "ThreatScoreAhead": threat_score,
            "MaxThreatAhead": max_threat,
            "FreshTyre": 1,
            "FinalPosition": current_pos,
            "IsWin": 0,
            "StintCompound": "SOFT",
            "NextPitLap": float(total_laps),
        }

        feat_arr = _build_feature_vector(row_dict, driver, team, circuit, preproc_params)

        prob_vec = m1.predict_proba(feat_arr)[0]
        comp_prefs = [(compound_idx_to_name.get(i, "SOFT"), p) for i, p in enumerate(prob_vec)]
        comp_prefs.sort(key=lambda x: x[1], reverse=True)

        valid_compounds = [c[0] for c in comp_prefs if c[0] in ["SOFT", "MEDIUM", "HARD"]]

        if stint_num == total_stints and total_stints > 1:
            used_compounds = set(s["compound"] for s in strategy)
            if len(used_compounds) == 1:
                used_c = list(used_compounds)[0]
                valid_compounds = [c for c in valid_compounds if c != used_c]

        if len(valid_compounds) > 1 and prev_compound in valid_compounds:
            valid_compounds.remove(prev_compound)

        if total_stints <= 2:
            if valid_compounds and valid_compounds[0] == "SOFT":
                hard_med = [c for c in valid_compounds if c in ["HARD", "MEDIUM"]]
                if hard_med:
                    valid_compounds = hard_med + [c for c in valid_compounds if c not in hard_med]
            if stint_num == 2 and "HARD" not in [s["compound"] for s in strategy]:
                if "HARD" in valid_compounds:
                    valid_compounds.remove("HARD")
                    valid_compounds.insert(0, "HARD")

        compound = valid_compounds[0] if valid_compounds else "SOFT"

        if stint_num == total_stints:
            next_pit = total_laps
        else:
            p_lap = int(round(current_lap + (total_laps - current_lap) / (total_stints - stint_num + 1)))  # even split of remaining laps
            max_laps_for_compound = TYRE_LIFE_LIMITS.get(compound, 30)
            p_lap = min(p_lap, current_lap + max_laps_for_compound)
            p_lap = max(p_lap, current_lap + 5)
            next_pit = p_lap

            remaining_laps = total_laps - next_pit
            remaining_stints = total_stints - stint_num
            if remaining_laps > remaining_stints * max(TYRE_LIFE_LIMITS.values()):
                next_pit = min(current_lap + max_laps_for_compound, total_laps - remaining_stints * 5)

        strategy.append({
            "stint": stint_num,
            "lap_start": current_lap,
            "lap_end": next_pit - 1 if next_pit < total_laps else total_laps,
            "compound": compound,
            "laps_on_tyre": (next_pit - 1 if next_pit < total_laps else total_laps) - current_lap + 1,
        })

        prev_compound = compound
        current_lap = next_pit

    wp_dict = dict(row_dict)
    wp_dict["StintCompound"] = strategy[0]["compound"]
    wp_arr = _build_feature_vector(wp_dict, driver, team, circuit, preproc_params)
    raw_win_prob = float(m3.predict_proba(wp_arr)[0, 1])

    win_prob = max(0.001, min(0.999, raw_win_prob))

    return {"strategy": strategy, "win_probability": win_prob, "threat_score": threat_score}


@st.cache_data(ttl=3600, max_entries=6)
def load_live(year, race):
    # walk-forward podium/winner chances per lap (phase_3_live), plus its backtest vs the running order
    spark = load_backend()["spark"]
    by_race = (F.col("Year") == int(year)) & (F.col("Race") == race)
    pred = spark.table("workspace.default.f1_live_predictions").filter(by_race).toPandas()
    return pred, spark.table("workspace.default.f1_live_backtest").toPandas()


# ---------------------------------------------------------------------------
# Theme — CSS the config.toml theme can't reach: layout, timeline/gauge
# components, and motion. Native widget colors/radius come from
# .streamlit/config.toml instead of being fought here.
# ---------------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');

:root {
    --f1-red: #E10600;
    --f1-red-bright: #FF3B30;
    --bg: #0C0C10;
    --surface: #17171C;
    --surface-alt: #1F1F26;
    --border: #2A2A32;
    --text: #F5F5F7;
    --text-dim: #9A9AA5;
}

html, body, [class*="css"] {
    font-family: 'Titillium Web', 'Segoe UI', sans-serif;
}

/* Hide default Streamlit chrome so the custom header owns the top of the page */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] > .main {
    background:
        linear-gradient(180deg, rgba(225,6,0,0.06) 0%, rgba(12,12,16,0) 220px),
        var(--bg);
}
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* Top speed-stripe accent */
.speed-stripe {
    height: 4px;
    width: 100%;
    background: repeating-linear-gradient(
        -45deg, var(--f1-red) 0 18px, #ffffff 18px 22px, var(--f1-red) 22px 40px, #15151E 40px 44px
    );
    margin: -1.5rem -1rem 1.75rem -1rem;
    border-radius: 0;
}

/* Hero header */
.f1-hero { margin-bottom: 1.75rem; }
.f1-eyebrow {
    color: var(--f1-red-bright);
    font-weight: 700;
    letter-spacing: 0.24em;
    font-size: 0.72rem;
    text-transform: uppercase;
    margin: 0 0 0.35rem 0;
}
.f1-title {
    font-weight: 900;
    font-size: 2.6rem;
    line-height: 1.05;
    letter-spacing: 0.01em;
    text-transform: uppercase;
    color: var(--text);
    margin: 0;
    position: relative;
    display: inline-block;
}
.f1-title::after {
    content: "";
    position: absolute;
    left: 2px; bottom: -10px;
    width: 68px; height: 8px;
    background: var(--f1-red);
    transform: skewX(-20deg);
}
.f1-subtitle {
    color: var(--text-dim);
    font-size: 0.95rem;
    margin: 1.1rem 0 0 0;
    max-width: 640px;
}

/* Sidebar "pit wall" panel */
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border);
}
.pitwall-label {
    color: var(--f1-red-bright);
    font-weight: 700;
    letter-spacing: 0.18em;
    font-size: 0.7rem;
    text-transform: uppercase;
    border-left: 3px solid var(--f1-red);
    padding-left: 0.5rem;
    margin: 0 0 1rem 0;
}
[data-testid="stWidgetLabel"] p {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
    color: var(--text-dim) !important;
    font-weight: 600;
}

/* Buttons: angular F1 speed-block shape */
.stButton > button, .stFormSubmitButton > button {
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    clip-path: polygon(0 0, 100% 0, 96% 100%, 4% 100%);
    transition: filter 0.15s ease, transform 0.15s ease;
    border: none !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    filter: brightness(1.18);
    transform: translateY(-1px);
}

/* Driver / result card header */
.result-card {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--f1-red);
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
    animation: slideFadeIn 0.5s ease-out both;
}
.result-driver-code {
    font-weight: 900;
    font-size: 1.6rem;
    letter-spacing: 0.04em;
    color: var(--text);
}
.result-meta { color: var(--text-dim); font-size: 0.85rem; }
.result-grid-badge {
    margin-left: auto;
    background: var(--f1-red);
    color: white;
    font-weight: 800;
    font-size: 0.95rem;
    padding: 0.35rem 0.9rem;
    clip-path: polygon(10% 0, 100% 0, 90% 100%, 0 100%);
}

/* Circular win-probability gauge */
.f1-gauge {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.1rem 1.3rem;
    height: 100%;
    animation: slideFadeIn 0.5s ease-out 0.1s both;
}
.f1-gauge svg { width: 92px; height: 92px; transform: rotate(-90deg); flex-shrink: 0; }
.gauge-track { fill: none; stroke: var(--surface-alt); stroke-width: 10; }
.gauge-fill {
    fill: none;
    stroke: var(--f1-red);
    stroke-width: 10;
    stroke-linecap: round;
    stroke-dasharray: var(--circumference);
    stroke-dashoffset: var(--circumference);
    animation: fillGauge 1.1s cubic-bezier(0.22, 1, 0.36, 1) 0.3s forwards;
}
@keyframes fillGauge { to { stroke-dashoffset: var(--offset-final); } }
.gauge-label { display: flex; flex-direction: column; }
.gauge-value {
    font-weight: 900;
    font-size: 1.9rem;
    font-variant-numeric: tabular-nums;
    color: var(--text);
    line-height: 1;
}
.gauge-value small { font-size: 1rem; color: var(--text-dim); }
.gauge-caption {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.68rem;
    color: var(--text-dim);
    margin-top: 0.4rem;
}

/* Threat chip */
.threat-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.1rem 1.3rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    animation: slideFadeIn 0.5s ease-out 0.15s both;
}
.threat-value {
    font-weight: 900;
    font-size: 1.9rem;
    color: var(--text);
    font-variant-numeric: tabular-nums;
}
.threat-caption {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.68rem;
    color: var(--text-dim);
    margin-top: 0.4rem;
}

/* Stint timeline */
.f1-timeline { margin: 1.75rem 0 0.5rem 0; animation: slideFadeIn 0.5s ease-out 0.2s both; }
.timeline-heading {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-bottom: 0.6rem;
}
.timeline-track {
    position: relative;
    height: 56px;
    background: var(--surface);
    border: 1px solid var(--border);
    overflow: hidden;
}
.stint-block {
    position: absolute;
    top: 0; bottom: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0.6rem;
    box-sizing: border-box;
    border-right: 2px solid var(--bg);
    overflow: hidden;
    white-space: nowrap;
    animation: growStint 0.7s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}
@keyframes growStint { from { width: 0%; } to { width: var(--target-width); } }
.stint-compound { font-weight: 900; font-size: 1.05rem; }
.stint-laps { font-weight: 700; font-size: 0.72rem; opacity: 0.85; }
.timeline-ticks {
    display: flex;
    justify-content: space-between;
    color: var(--text-dim);
    font-size: 0.7rem;
    margin-top: 0.4rem;
    font-variant-numeric: tabular-nums;
}

/* Stint data table */
.f1-table { width: 100%; border-collapse: collapse; margin-top: 1.5rem; animation: slideFadeIn 0.5s ease-out 0.3s both; }
.f1-table th {
    text-align: left;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.68rem;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding: 0.5rem 0.75rem;
}
.f1-table td {
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid var(--border);
    color: var(--text);
    font-size: 0.88rem;
    font-variant-numeric: tabular-nums;
}
.f1-table tr:nth-child(even) td { background: var(--surface); }
.compound-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 0.5rem;
    vertical-align: middle;
    border: 1px solid rgba(255,255,255,0.25);
}

/* Idle "lights out" placeholder */
.lights-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 2.5rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.lights-row { display: flex; justify-content: center; gap: 0.9rem; margin-bottom: 1.25rem; }
.light {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #3A1010;
    box-shadow: inset 0 0 4px rgba(0,0,0,0.6);
    animation: lightsSequence 3.2s ease-in-out infinite;
}
.light:nth-child(1) { animation-delay: 0s; }
.light:nth-child(2) { animation-delay: 0.35s; }
.light:nth-child(3) { animation-delay: 0.7s; }
.light:nth-child(4) { animation-delay: 1.05s; }
.light:nth-child(5) { animation-delay: 1.4s; }
@keyframes lightsSequence {
    0%, 100% { background: #3A1010; box-shadow: inset 0 0 4px rgba(0,0,0,0.6); }
    12%, 55% { background: var(--f1-red); box-shadow: 0 0 14px 2px rgba(225,6,0,0.75); }
    70%, 90% { background: #3A1010; box-shadow: inset 0 0 4px rgba(0,0,0,0.6); }
}
.lights-caption {
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.75rem;
}

@keyframes slideFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Predicted-battles ticker */
.battle-ticker-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--f1-red-bright);
    overflow: hidden;
    white-space: nowrap;
    padding: 0.7rem 0;
    margin-bottom: 1.5rem;
}
.battle-ticker-track {
    display: inline-block;
    padding-left: 100%;
    animation: tickerScroll 30s linear infinite;
}
@keyframes tickerScroll {
    from { transform: translateX(0); }
    to { transform: translateX(-100%); }
}
.battle-ticker-item {
    display: inline-block;
    font-weight: 700;
    letter-spacing: 0.04em;
    font-size: 0.85rem;
    color: var(--text);
    padding: 0 2rem;
    text-transform: uppercase;
}
.battle-ticker-item .prob { color: var(--f1-red-bright); font-variant-numeric: tabular-nums; }
.battle-ticker-item .sep { color: var(--text-dim); margin: 0 0.6rem; }

.f1-footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 0.78rem;
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""


def render_gauge(win_probability: float) -> str:
    pct = max(0.0, min(1.0, win_probability)) * 100
    r = 54
    circumference = 2 * 3.14159265358979 * r
    offset_final = circumference * (1 - pct / 100)
    return f"""
    <div class="f1-gauge">
      <svg viewBox="0 0 120 120">
        <circle class="gauge-track" cx="60" cy="60" r="{r}" />
        <circle class="gauge-fill" cx="60" cy="60" r="{r}"
                style="--circumference:{circumference:.2f}px; --offset-final:{offset_final:.2f}px;" />
      </svg>
      <div class="gauge-label">
        <span class="gauge-value">{pct:.1f}<small>%</small></span>
        <span class="gauge-caption">Win Probability</span>
      </div>
    </div>
    """


def render_threat_chip(threat_score: float) -> str:
    return f"""
    <div class="threat-chip">
        <span class="threat-value">{threat_score:.2f}</span>
        <span class="threat-caption">Threat Ahead &middot; Summed Win Rate On Grid</span>
    </div>
    """


def render_timeline(strategy: list, total_laps: int) -> str:
    segments = []
    for i, s in enumerate(strategy):
        width_pct = s["laps_on_tyre"] / total_laps * 100
        left_pct = (s["lap_start"] - 1) / total_laps * 100
        color = COMPOUND_COLORS.get(s["compound"], "#9b59b6")
        text_color = COMPOUND_TEXT_COLORS.get(s["compound"], "#fff")
        segments.append(f"""
        <div class="stint-block"
             style="left:{left_pct:.3f}%; --target-width:{width_pct:.3f}%;
                    background:{color}; color:{text_color}; animation-delay:{i * 0.15:.2f}s;">
          <span class="stint-compound">{s['compound'][0]}</span>
          <span class="stint-laps">{s['laps_on_tyre']}L</span>
        </div>
        """)
    ticks = "".join(
        f"<span>Lap {int(total_laps * frac)}</span>" for frac in (0, 0.25, 0.5, 0.75, 1.0)
    )
    return f"""
    <div class="f1-timeline">
      <div class="timeline-heading">Stint-by-Stint Tyre Strategy</div>
      <div class="timeline-track">{''.join(segments)}</div>
      <div class="timeline-ticks">{ticks}</div>
    </div>
    """


def render_stint_table(strategy: list) -> str:
    rows = []
    for s in strategy:
        color = COMPOUND_COLORS.get(s["compound"], "#9b59b6")
        rows.append(f"""
        <tr>
          <td>{s['stint']}</td>
          <td>{s['lap_start']}&ndash;{s['lap_end']}</td>
          <td><span class="compound-dot" style="background:{color};"></span>{s['compound']}</td>
          <td>{s['laps_on_tyre']} laps</td>
        </tr>
        """)
    return f"""
    <table class="f1-table">
      <thead><tr><th>Stint</th><th>Laps</th><th>Compound</th><th>Tyre Life</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="F1 Race Intelligence", page_icon="🏎️", layout="wide")
st.markdown(THEME_CSS, unsafe_allow_html=True)
st.markdown('<div class="speed-stripe"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="f1-hero">
      <p class="f1-eyebrow">Pit Wall &middot; Prediction Suite</p>
      <h1 class="f1-title">Race Intelligence</h1>
      <p class="f1-subtitle">
        Strategy simulation and a full-grid race replay with battles, tyres and key moments —
        built on phase_3_strategy.ipynb, phase_3_ranking.ipynb, and phase_3_battles.ipynb.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Connecting to Spark and loading models..."):
    drivers, teams, circuits, years = load_options()
    circuit_typical_laps = load_circuit_typical_laps()

DEFAULT_RACE_LAPS = 55

with st.sidebar:
    st.markdown('<p class="pitwall-label">Race Setup</p>', unsafe_allow_html=True)
    # Circuit lives outside the form, deliberately: widgets inside st.form()
    # don't trigger a rerun until submit, so Race Distance below couldn't
    # react to the circuit choice if this stayed in there with everything
    # else. This is the one field that needs to be "live".
    circuit = st.selectbox("Circuit", circuits, key="circuit_select")
    with st.form("strategy_form"):
        driver = st.selectbox("Driver", drivers, index=drivers.index("HAM") if "HAM" in drivers else 0)
        team = st.selectbox("Team", teams)
        year = st.selectbox("Season (driver/car form)", years, index=len(years) - 1)
        quali_position = st.number_input("Starting grid position", min_value=1, max_value=20, value=1)
        # key includes circuit so switching circuits resets the field to
        # that circuit's own default/prior value, rather than Streamlit
        # preserving whatever number was left over from a different track
        # (value= only sets a widget's default the first time it's ever
        # rendered under a given key -- it's ignored on later reruns, so a
        # fixed key wouldn't actually update when circuit changes).
        total_laps = st.number_input(
            "Race distance (laps)",
            min_value=20, max_value=90,
            value=circuit_typical_laps.get(circuit, DEFAULT_RACE_LAPS),
            key=f"total_laps_{circuit}",
            help="Defaults to this circuit's typical race distance from training data -- override if needed.",
        )
        num_pit_stops = st.slider("Number of pit stops", min_value=1, max_value=3, value=1)
        submitted = st.form_submit_button("Predict Strategy", type="primary", use_container_width=True)

tab_strategy, tab_replay, tab_future = st.tabs(["Strategy Simulator", "Race Replay", "Future Race"])

with tab_strategy:
    if not submitted:
        st.markdown(
            """
            <div class="lights-panel">
              <div class="lights-row">
                <span class="light"></span><span class="light"></span><span class="light"></span>
                <span class="light"></span><span class="light"></span>
              </div>
              <div class="lights-caption">Set the grid in the sidebar, then lights out and away we go</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Running the strategy models..."):
            result = predict_strategy(
                driver=driver,
                team=team,
                year=int(year),
                circuit=circuit,
                quali_position=int(quali_position),
                total_laps=int(total_laps),
                num_pit_stops=int(num_pit_stops),
            )

        st.markdown(
            f"""
            <div class="result-card">
              <span class="result-driver-code">{html.escape(driver)}</span>
              <span class="result-meta">{html.escape(team)} &middot; {html.escape(circuit)} &middot; {int(year)}</span>
              <span class="result-grid-badge">P{int(quali_position)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        gauge_col, threat_col = st.columns(2)
        with gauge_col:
            st.markdown(render_gauge(result["win_probability"]), unsafe_allow_html=True)
        with threat_col:
            st.markdown(render_threat_chip(result["threat_score"]), unsafe_allow_html=True)

        st.markdown(render_timeline(result["strategy"], int(total_laps)), unsafe_allow_html=True)
        st.markdown(render_stint_table(result["strategy"]), unsafe_allow_html=True)

        st.markdown(
            """
            <div class="f1-footer">
              Win probability is the win-probability model's output, driven by grid position, driver and car form, and the win rates of the cars starting ahead. It does not model race-day events.
              </div>
            """,
            unsafe_allow_html=True,
        )

TOUR_STEPS = """
**A lap** is one trip around the track. **P1** is first place; the list below is the running order.
**A gap** is how many seconds a car is behind the one in front. When a gap is under 1 second, the chaser is in position to attack.
Use the lap slider to move through the race. Cards show only what had happened by that lap, so nothing is spoiled.
"""

with tab_replay:
    st.markdown(rp.CSS, unsafe_allow_html=True)
    first_visit = not st.session_state.get("tour_seen")
    st.session_state["tour_seen"] = True
    with st.expander("New to F1? 60-second tour", expanded=first_visit):
        st.markdown(TOUR_STEPS)

    st.markdown(
        '<div class="rp-banner">This replays a real past race (2018 onward). Gaps, tyres and positions are real. '
        "Pass, podium and win chances are model estimates; tick &lsquo;Show model estimates&rsquo; to see them.</div>",
        unsafe_allow_html=True,
    )
    r_col1, r_col2, r_col3 = st.columns([1, 2, 2])
    replay_index = load_races()
    replay_years = sorted({y for y, _ in replay_index})
    with r_col1:
        replay_year = st.selectbox("Season", replay_years, index=len(replay_years) - 1, key="replay_year")
    with r_col2:
        replay_circuit = st.selectbox("Circuit", sorted({c for y, c in replay_index if y == replay_year}), key="replay_circuit")
    with r_col3:
        replay_pick = pick_race(replay_year, replay_circuit, "replay_race")

    if replay_pick is None:
        st.markdown('<div class="lights-panel"><div class="lights-caption">No lap data found for that race</div></div>', unsafe_allow_html=True)
    else:
        view = load_race_view(int(replay_year), replay_pick)
        lap_key = f"rp_lap_{int(replay_year)}_{replay_pick}"
        st.session_state.setdefault(lap_key, 0)

        def _step(delta, key=lap_key, mx=view.total):
            st.session_state[key] = max(0, min(mx, st.session_state[key] + delta))

        n_col1, n_col2, n_col3 = st.columns([1, 1, 8])
        n_col1.button("◀ Prev", on_click=_step, args=(-1,), use_container_width=True)
        n_col2.button("Next ▶", on_click=_step, args=(1,), use_container_width=True)
        lap = n_col3.slider("Lap", 0, view.total, key=lap_key, help="Lap 0 is the starting grid.")

        names = {d: f"{d} · {view.meta[d]['name']}" for d in view.drivers}
        follow = st.selectbox("Follow a driver", [None] + view.drivers, format_func=lambda d: "Whole field" if d is None else names[d], key="rp_follow")
        o_col1, o_col2, o_col3 = st.columns(3)
        spoilers = o_col1.toggle("Hide final result", value=True, key="rp_spoil", help="On: nothing after the selected lap is shown, including the podium.")
        est_on = o_col2.toggle("Show model estimates", value=False, key="rp_est", disabled=int(replay_year) < 2021, help="Adds pass-chance bands and each car's podium and win chance. Off by default so you see only real data first. Available from 2021, where the models were tested on unseen seasons.")
        fewer = o_col3.toggle("Fewer markers", value=False, key="rp_fewer", help="Hides pit stops and under-1-second markers for cars you are not following.")

        st.markdown(f'<div class="rp-lap">{"Starting grid" if lap == 0 else f"Lap {lap} of {view.total}"}</div>', unsafe_allow_html=True)
        if lap > 0 and bool(view.sc.loc[lap].any()):
            st.markdown('<div class="rp-sc">Safety car or virtual safety car this lap</div>', unsafe_allow_html=True)

        est_map = {}
        if est_on and lap > 0 and not view.pairs.empty:
            lp = view.pairs[view.pairs["LapNumber"] == lap]
            est_map = dict(zip(lp["Trailing"], lp["Band"]))
            if view.bands:
                legend = " / ".join(f"{b} {view.bands[b]['lo'] * 100:.1f}-{view.bands[b]['hi'] * 100:.1f}% (observed 1 in {rp.one_in(view.bands[b]['rate'])})" for b in ("Low", "Medium", "High") if b in view.bands)
                st.caption(f"Pass-chance bands, next 5 laps if the fight stays on track: {legend}.")
        live_pred, live_bt = load_live(int(replay_year), replay_pick) if est_on else (pd.DataFrame(), pd.DataFrame())
        live_map = {}
        if est_on and lap > 0 and not live_pred.empty:
            lp = live_pred[live_pred["LapNumber"] == lap]
            live_map = {r.Driver: (r.PodiumP, r.WinP) for r in lp.itertuples()}

        g_tab, b_tab, p_tab, t_tab, m_tab = st.tabs(["Grid", "Battles", "Positions", "Tyres", "Key moments"])
        with g_tab:
            cards = view.cards(lap)
            st.markdown(rp.render_grid(cards, lap, follow, est_map, live_map) + rp.render_legend(view), unsafe_allow_html=True)
            st.caption("Tap a card for the car ahead's tyres and the last few gaps.")
        with b_tab:
            more = st.toggle("Show more battles", value=False, key="rp_more")
            board = view.board(lap, follow, est_on, n=8 if more else 3)
            st.markdown(rp.render_board(board, est_on, lap), unsafe_allow_html=True)
        with p_tab:
            chart = rp.position_chart(view, lap, follow)
            if chart is not None:
                st.altair_chart(chart, use_container_width=True)
        with t_tab:
            st.markdown(rp.render_stints(view, lap), unsafe_allow_html=True)
        with m_tab:
            st.markdown(rp.render_moments(view.moments_upto(lap, follow, fewer), view), unsafe_allow_html=True)

        if est_on:
            with st.expander("Podium and win chance over the race", expanded=True):
                if live_pred.empty:
                    st.caption("No podium or win estimates for this race.")
                else:
                    seen = live_pred[live_pred["LapNumber"] <= max(lap, 1)]
                    pod = seen.pivot(index="LapNumber", columns="Driver", values="PodiumP") * 100
                    win = seen.pivot(index="LapNumber", columns="Driver", values="WinP") * 100
                    if spoilers:
                        top = pod.iloc[-1].sort_values(ascending=False).head(5).index.tolist()
                        st.caption("Top 5 by current podium chance (final result hidden).")
                    else:
                        top = view.order(view.total)[0][:3]
                        st.caption(f"Eventual podium: {', '.join(top)}")
                    c1, c2 = st.columns(2)
                    c1.markdown("**Podium chance, %**")
                    c1.line_chart(pod[[d for d in top if d in pod.columns]], height=240)
                    c2.markdown("**Win chance, %**")
                    c2.line_chart(win[[d for d in top if d in win.columns]].fillna(0), height=240)
                    st.caption(rp.live_note(live_bt))
        with st.expander("Glossary"):
            st.markdown("\n".join(f"- **{k}**: {v}" for k, v in rp.GLOSSARY.items()))

    st.markdown(
        """
        <div class="f1-footer">
          Gaps, positions and tyres are measured at the end of each lap (FastF1 lap data to 2023, OpenF1 from 2024), not live telemetry, so &ldquo;under 1 s&rdquo; is a lap-end snapshot and says nothing about DRS.
          Pass-chance bands come from phase_3_battles.ipynb (validated 2021-2023); podium and win chances from phase_3_live.ipynb (walk-forward, each season predicted by a model trained only on earlier seasons).
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_future:
    st.markdown(fu.CSS, unsafe_allow_html=True)
    fmodel = load_future_model()
    st.markdown(
        f'<div class="rp-banner">Predicts a race that has not happened yet from the starting grid and recent form (results up to {html.escape(fmodel.m["asof"])}, '
        f'{html.escape(fmodel.m["latest_race"])}). Each number is the chance of finishing in the top 3; the field adds up to 300%. '
        "Weather, penalties and first-lap incidents are not modelled.</div>",
        unsafe_allow_html=True,
    )
    circ_names = fmodel.circuits.set_index("Location")["CircuitName"].sort_values()
    locs = circ_names.index.tolist()
    f_col1, f_col2 = st.columns([1, 1])
    circuit_loc = f_col1.selectbox("Circuit", locs, index=locs.index("yas_marina") if "yas_marina" in locs else 0, format_func=lambda k: circ_names[k], key="fut_circuit")
    recent = fmodel.recent_grids[["Round", "Race"]].drop_duplicates().sort_values("Round", ascending=False)
    prefill_opts = ["Estimated from current form"] + [f"Actual grid: {r.Race}" for r in recent.itertuples()]
    prefill = f_col2.selectbox("Starting grid", prefill_opts, key="fut_prefill", help="Start from an estimate, or copy a real recent grid and edit it. After Saturday qualifying, enter the real grid.")

    if prefill == prefill_opts[0]:
        order = fmodel.default_order()
        base = pd.DataFrame({"Driver": order, "Grid": range(1, len(order) + 1)})
    else:
        gr = fmodel.recent_grids[fmodel.recent_grids["Race"] == prefill.replace("Actual grid: ", "")]
        gr = gr[gr["Driver"].isin(fmodel.current_drivers.index)].sort_values("GridPosition")
        base = pd.DataFrame({"Driver": gr["Driver"].tolist(), "Grid": range(1, len(gr) + 1)})
        missing = [d for d in fmodel.current_drivers.index if d not in set(base["Driver"])]
        base = pd.concat([base, pd.DataFrame({"Driver": missing, "Grid": range(len(base) + 1, len(base) + len(missing) + 1)})], ignore_index=True)
    base["Team"] = base["Driver"].map(fmodel.current_drivers["TeamKey"])
    base["Quali gap to pole (%)"] = np.nan
    team_opts = sorted(fmodel.teams.index)

    edited = st.data_editor(
        base[["Driver", "Team", "Grid", "Quali gap to pole (%)"]],
        key=f"fut_editor_{circuit_loc}_{prefill}", hide_index=True, use_container_width=True, num_rows="fixed",
        column_config={
            "Driver": st.column_config.TextColumn(disabled=True),
            "Team": st.column_config.SelectboxColumn(options=team_opts, help="Change this to ask 'what if this driver were at another team'."),
            "Grid": st.column_config.NumberColumn(min_value=1, max_value=len(base), step=1),
            "Quali gap to pole (%)": st.column_config.NumberColumn(min_value=0.0, max_value=10.0, help="Optional. Leave blank to use the typical gap for that grid slot."),
        },
    )
    focus_opts = [None] + edited["Driver"].tolist()
    fut_names = fmodel.drivers["FullName"].to_dict()
    focus = st.selectbox("Highlight a driver", focus_opts, format_func=lambda d: "Whole field" if d is None else f"{d} · {fut_names.get(d, '')}", key="fut_focus")

    if edited["Grid"].isna().any() or edited["Grid"].duplicated().any():
        st.warning("Each driver needs a different grid position. Fix the duplicates in the table to see predictions.")
    else:
        entries = [
            {"driver": r["Driver"], "team": r["Team"], "grid": int(r["Grid"]), "gap": r["Quali gap to pole (%)"]}
            for r in edited.to_dict("records")
        ]
        mob = fmodel.circuits.set_index("Location")["CircuitMobility"].get(circuit_loc)
        results = fmodel.predict(entries, None if pd.isna(mob) else float(mob))
        sel = next((r for r in results if r["driver"] == focus), None)
        if sel:
            rank = sorted(results, key=lambda x: -x["p"]).index(sel) + 1
            st.markdown(
                f'<div class="rp-board"><b>{html.escape(fut_names.get(focus, focus))}</b> ({html.escape(sel["team"])}) starting P{sel["grid"]}: '
                f'about <b>{sel["p"] * 100:.0f}%</b> to finish on the podium (model range {sel["lo"] * 100:.0f}&ndash;{sel["hi"] * 100:.0f}%), '
                f'ranked {rank} of {len(results)} in the field. Grid-slot-only baseline: {sel["baseline"] * 100:.0f}%.</div>',
                unsafe_allow_html=True,
            )
        total = sum(r["p"] for r in results)
        st.markdown(fu.render_results(results, fu.TEAM_COLORS, fut_names, focus), unsafe_allow_html=True)
        pooled = fmodel.m["pooled"]
        st.caption(
            f"Total across the field: {total * 100:.0f}% (three podium places). Backtest 2022 to {fmodel.m['latest_year']}, pooled: "
            f"model Brier {pooled['model']['brier']:.4f} vs grid-slot baseline {pooled['slot']['brier']:.4f} (lower is better); AUC {pooled['model']['auc']:.3f} vs {pooled['slot']['auc']:.3f}."
            + ("" if fmodel.m["ship_model"] else " The model did not clearly beat the baseline, so treat these as indicative.")
        )
    with st.expander("How to read this"):
        st.markdown(
            "- **Number:** chance of a top-3 finish. **Bar:** how much the estimate moves if the model is refit on slightly different past races (not a forecast of race-day luck); the white tick is the central estimate.\n"
            "- **Car pace** comes from recent races, weighted toward the current season. After a rule change (2022, 2026) older seasons count for very little.\n"
            "- **Drivers or teams with few races** (a rookie, a new team) get a cautious prior and a wider range.\n"
            "- **Not modelled:** weather, grid penalties applied after qualifying, first-lap collisions, strategy gambles."
        )
