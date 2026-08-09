"""
F1 Race Intelligence — Databricks App (Plans B, G, E).

Three tabs over three separately-trained model sets:
- Strategy Simulator (Plan B): pick a driver, team, circuit, and pit-stop
  count, get a predicted stint-by-stint tyre strategy and win probability.
  Models from notebooks/phase_3_strategy_experimental.ipynb.
- Predicted Battles (Plan G): scan a race/lap for drivers running close
  together and get an overtake probability per pair. Model from
  notebooks/phase_3_battles.ipynb.
- Post-Race Replay (Plan E): replay a finished race lap-by-lap through
  live_model and chart each driver's predicted podium probability over
  time. Model from notebooks/phase_3.ipynb.

Runs as a lightweight Streamlit web service, separate from any notebook.
It gets a real Spark session via Databricks Connect (serverless) rather
than depending on a notebook's native `spark` object — an App has no
notebook to inherit one from. Model artifacts come from the Unity Catalog
Volume /Volumes/workspace/default/f1_data (strategy_models/,
battle_models/), the f1_strategy_*/f1_circuit_overtake_prior tables, and
the UC model registry for live_model — workspace-relative paths like
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

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import streamlit as st
from databricks.connect.session import DatabricksSession
from mlflow import MlflowClient
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from xgboost import XGBClassifier, XGBRegressor

STRATEGY_VOLUME_DIR = "/Volumes/workspace/default/f1_data/strategy_models"
BATTLE_VOLUME_DIR = "/Volumes/workspace/default/f1_data/battle_models"
CAT_FEATURES = ["Circuit", "TeamName", "Driver"]
TYRE_LIFE_LIMITS = {"SOFT": 25, "MEDIUM": 40, "HARD": 55, "INTERMEDIATE": 30, "WET": 40}

BATTLE_FEATURES = [
    "Gap", "GapClosingRate", "TyreLifeDelta",
    "TrailingCompound", "LeadingCompound",
    "SpeedST_Delta", "CircuitOvertakePrior",
]
BATTLE_CATEGORICAL = ["TrailingCompound", "LeadingCompound"]

FEATURES_LIVE = [
    "LapNumber", "RacePhase", "Position", "GapToAhead", "DeltaToLeader",
    "Compound", "TyreLife", "Stint", "TrackStatus", "tyre_degradation_rate",
    "AirTemp_delta", "TrackTemp_delta", "Humidity_delta", "WindSpeed_delta",
]
CATEGORICAL_LIVE = ["Compound", "RacePhase"]

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


@st.cache_resource
def load_backend():
    """Spark session + trained models — expensive, loaded once per app instance."""
    spark = DatabricksSession.builder.serverless().getOrCreate()

    preproc_model = PipelineModel.load(f"{STRATEGY_VOLUME_DIR}/preproc_pipeline")

    m1 = XGBClassifier()
    m1.load_model(f"{STRATEGY_VOLUME_DIR}/m1_compound_classifier.json")

    m2 = XGBRegressor()
    m2.load_model(f"{STRATEGY_VOLUME_DIR}/m2_pitlap_regressor.json")

    m3 = XGBClassifier()
    m3.load_model(f"{STRATEGY_VOLUME_DIR}/m3_win_classifier.json")

    with open(f"{STRATEGY_VOLUME_DIR}/compound_label_map.json") as f:
        compound_idx_to_name = {int(k): v for k, v in json.load(f).items()}

    driver_profile = spark.table("workspace.default.f1_strategy_driver_profile")
    constructor_profile = spark.table("workspace.default.f1_strategy_constructor_profile")
    circuit_profile = spark.table("workspace.default.f1_strategy_circuit_profile")
    cleaned_laps = spark.table("workspace.default.f1_cleaned_lap_dataset")

    battle_model = XGBClassifier()
    battle_model.load_model(f"{BATTLE_VOLUME_DIR}/m_overtake_classifier.json")
    circuit_overtake_prior = spark.table("workspace.default.f1_circuit_overtake_prior")

    # live_model.json is workspace-relative to notebooks/, unreachable from
    # this App's own container — loaded from the UC model registry instead.
    mlflow.set_registry_uri("databricks-uc")
    _live_versions = MlflowClient().search_model_versions("name='workspace.default.f1_live_model'")
    _live_latest = max(int(v.version) for v in _live_versions)
    live_model = mlflow.xgboost.load_model(f"models:/workspace.default.f1_live_model/{_live_latest}")
    ml_lap_dataset = spark.table("workspace.default.f1_ml_lap_dataset")

    return {
        "spark": spark,
        "preproc_model": preproc_model,
        "m1": m1,
        "m2": m2,
        "m3": m3,
        "compound_idx_to_name": compound_idx_to_name,
        "driver_profile": driver_profile,
        "constructor_profile": constructor_profile,
        "circuit_profile": circuit_profile,
        "cleaned_laps": cleaned_laps,
        "battle_model": battle_model,
        "circuit_overtake_prior": circuit_overtake_prior,
        "live_model": live_model,
        "ml_lap_dataset": ml_lap_dataset,
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
    phase_3_strategy_experimental.ipynb) — kept in sync manually since an
    App can't import a notebook. Returns {"strategy": [...], "win_probability": float}.
    """
    backend = load_backend()
    spark = backend["spark"]
    preproc_model = backend["preproc_model"]
    m1, m2, m3 = backend["m1"], backend["m2"], backend["m3"]
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
            "StintLength": max(1, total_laps - current_lap),
            "StintStartFrac": lap_frac,
            "StintAvgLapTime": 95.0,
            "StintBestLap": 92.0,
            "QualiPosition": quali_position,
            "TotalLaps": total_laps,
            "NumStints": total_stints,
            "AvgGapToAhead": 1.5,
            "AvgDeltaToLeader": max(0, (current_pos - 1) * 1.2),
            "PositionAtStintEnd": current_pos,
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

        row_spark = spark.createDataFrame([row_dict])
        for cat in CAT_FEATURES:
            row_spark = row_spark.withColumn(
                cat, F.lit(driver if cat == "Driver" else (team if cat == "TeamName" else circuit))
            )

        pipe_cols = (
            ["CompoundLabel", "features_raw", "features"]
            + [c + "_idx" for c in CAT_FEATURES]
            + [c + "_ohe" for c in CAT_FEATURES]
        )
        for c in pipe_cols:
            if c in row_spark.columns:
                row_spark = row_spark.drop(c)

        row_proc = preproc_model.transform(row_spark)
        feat_arr = np.array([row_proc.select("features").first()[0].toArray()])

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
            p_lap = int(round(float(m2.predict(feat_arr)[0])))
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
    wp_row = spark.createDataFrame([wp_dict])
    for cat in CAT_FEATURES:
        wp_row = wp_row.withColumn(
            cat, F.lit(driver if cat == "Driver" else (team if cat == "TeamName" else circuit))
        )
    pipe_cols = (
        ["CompoundLabel", "features_raw", "features"]
        + [c + "_idx" for c in CAT_FEATURES]
        + [c + "_ohe" for c in CAT_FEATURES]
    )
    for c in pipe_cols:
        if c in wp_row.columns:
            wp_row = wp_row.drop(c)
    wp_proc = preproc_model.transform(wp_row)
    wp_arr = np.array([wp_proc.select("features").first()[0].toArray()])
    raw_win_prob = float(m3.predict_proba(wp_arr)[0, 1])

    grid_baseline = (
        0.40 if quali_position == 1 else
        0.25 if quali_position == 2 else
        0.15 if quali_position == 3 else
        max(0.01, 0.10 - quali_position * 0.01)
    )
    driver_factor = drv_win_rate * 2.0
    threat_penalty = max(0, threat_score - drv_win_rate) * 0.5
    calibrated_prob = (raw_win_prob * 3.0) + grid_baseline + (driver_factor * 0.5) - threat_penalty
    win_prob = max(0.001, min(0.999, calibrated_prob))

    return {"strategy": strategy, "win_probability": win_prob, "threat_score": threat_score}


def predict_overtake_probability(gap, gap_closing_rate, tyre_life_delta, trailing_compound, leading_compound, speedst_delta, circuit_overtake_prior):
    backend = load_backend()
    row = pd.DataFrame([{
        "Gap": gap,
        "GapClosingRate": gap_closing_rate,
        "TyreLifeDelta": tyre_life_delta,
        "TrailingCompound": trailing_compound,
        "LeadingCompound": leading_compound,
        "SpeedST_Delta": speedst_delta,
        "CircuitOvertakePrior": circuit_overtake_prior,
    }])
    for c in BATTLE_CATEGORICAL:
        row[c] = row[c].astype("category")
    return float(backend["battle_model"].predict_proba(row[BATTLE_FEATURES])[0, 1])


def predict_battles(year, circuit, lap_number, gap_threshold=3.0):
    backend = load_backend()
    spark = backend["spark"]

    race_laps = backend["cleaned_laps"].filter(
        (F.col("Year") == year) & (F.col("Circuit") == circuit)
    )
    closing_window = Window.partitionBy("Driver").orderBy("LapNumber").rowsBetween(-2, 0)

    pos_df = race_laps.select(
        "LapNumber", "Driver", "Position", "GapToAhead", "TyreLife", "Compound", "SpeedST"
    ).dropDuplicates(["Driver", "LapNumber"])

    pos_df = (
        pos_df
        .withColumn("_gc_n", F.count("GapToAhead").over(closing_window))
        .withColumn("_gc_sum_x", F.sum("LapNumber").over(closing_window))
        .withColumn("_gc_sum_y", F.sum("GapToAhead").over(closing_window))
        .withColumn("_gc_sum_xy", F.sum(F.col("LapNumber") * F.col("GapToAhead")).over(closing_window))
        .withColumn("_gc_sum_xx", F.sum(F.col("LapNumber") * F.col("LapNumber")).over(closing_window))
    )
    _gc_denom = F.col("_gc_n") * F.col("_gc_sum_xx") - F.col("_gc_sum_x") * F.col("_gc_sum_x")
    pos_df = pos_df.withColumn(
        "gap_closing_rate",
        F.when(
            (F.col("_gc_n") >= 2) & (_gc_denom != 0),
            (F.col("_gc_n") * F.col("_gc_sum_xy") - F.col("_gc_sum_x") * F.col("_gc_sum_y")) / _gc_denom
        ).otherwise(F.lit(None).cast("double"))
    ).drop("_gc_n", "_gc_sum_x", "_gc_sum_y", "_gc_sum_xy", "_gc_sum_xx")

    lap_state = pos_df.filter(F.col("LapNumber") == lap_number)
    trailing = lap_state.alias("t")
    leading = lap_state.alias("l")

    pairs = (
        trailing.join(leading, F.col("t.Position") == F.col("l.Position") + 1)
        .filter((F.col("t.GapToAhead") > 0) & (F.col("t.GapToAhead") <= gap_threshold))
        .select(
            F.col("t.Driver").alias("TrailingDriver"),
            F.col("l.Driver").alias("LeadingDriver"),
            F.col("t.GapToAhead").alias("Gap"),
            F.col("t.gap_closing_rate").alias("GapClosingRate"),
            (F.col("t.TyreLife") - F.col("l.TyreLife")).alias("TyreLifeDelta"),
            F.col("t.Compound").alias("TrailingCompound"),
            F.col("l.Compound").alias("LeadingCompound"),
            (F.col("t.SpeedST") - F.col("l.SpeedST")).alias("SpeedST_Delta"),
        )
        .toPandas()
    )

    prior_row = backend["circuit_overtake_prior"].filter(F.col("Circuit") == circuit).first()
    circuit_prior = float(prior_row["CircuitOvertakePrior"]) if prior_row and prior_row["CircuitOvertakePrior"] is not None else 0.15

    battles = []
    for _, row in pairs.iterrows():
        prob = predict_overtake_probability(
            gap=row["Gap"],
            gap_closing_rate=row["GapClosingRate"],
            tyre_life_delta=row["TyreLifeDelta"],
            trailing_compound=row["TrailingCompound"],
            leading_compound=row["LeadingCompound"],
            speedst_delta=row["SpeedST_Delta"],
            circuit_overtake_prior=circuit_prior,
        )
        battles.append({
            "trailing_driver": row["TrailingDriver"],
            "leading_driver": row["LeadingDriver"],
            "gap": float(row["Gap"]),
            "overtake_probability": prob,
        })

    return sorted(battles, key=lambda b: -b["overtake_probability"])


def replay_race(year, circuit):
    backend = load_backend()
    race_df = (
        backend["ml_lap_dataset"]
        .filter((F.col("Year") == year) & (F.col("Circuit") == circuit))
        .select(FEATURES_LIVE + ["Driver", "FinalPosition"])
        .toPandas()
    )
    for c in CATEGORICAL_LIVE:
        race_df[c] = race_df[c].astype("category")

    rows = []
    for lap_number, lap_group in race_df.groupby("LapNumber"):
        scores = backend["live_model"].predict(lap_group[FEATURES_LIVE])
        exp_scores = np.exp(scores - scores.max())
        probs = exp_scores / exp_scores.sum()
        for driver, final_pos, prob in zip(lap_group["Driver"], lap_group["FinalPosition"], probs):
            rows.append({
                "LapNumber": lap_number,
                "Driver": driver,
                "FinalPosition": final_pos,
                "Probability": prob,
            })

    return pd.DataFrame(rows)


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


def render_battle_ticker(battles: list) -> str:
    if not battles:
        return '<div class="battle-ticker-wrap"><div class="battle-ticker-track"><span class="battle-ticker-item">No battles within the gap threshold on this lap</span></div></div>'
    items = "".join(
        f"""<span class="battle-ticker-item">
              {html.escape(b['trailing_driver'])} CLOSING ON {html.escape(b['leading_driver'])}
              <span class="sep">&middot;</span>
              <span class="prob">{b['overtake_probability'] * 100:.0f}%</span> OVERTAKE CHANCE
              <span class="sep">&middot;</span> GAP {b['gap']:.1f}s
            </span>"""
        for b in battles
    )
    return f'<div class="battle-ticker-wrap"><div class="battle-ticker-track">{items}{items}</div></div>'


def render_battle_table(battles: list) -> str:
    rows = []
    for b in battles:
        rows.append(f"""
        <tr>
          <td>{html.escape(b['trailing_driver'])} &rarr; {html.escape(b['leading_driver'])}</td>
          <td>{b['gap']:.2f}s</td>
          <td>{b['overtake_probability'] * 100:.1f}%</td>
        </tr>
        """)
    return f"""
    <table class="f1-table">
      <thead><tr><th>Battle</th><th>Gap</th><th>Overtake Chance</th></tr></thead>
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
        Strategy simulation, predicted battles, and post-race probability replay — from the
        models trained in phase_3.ipynb, phase_3_strategy_experimental.ipynb, and phase_3_battles.ipynb.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Connecting to Spark and loading models..."):
    drivers, teams, circuits, years = load_options()

with st.sidebar:
    st.markdown('<p class="pitwall-label">Race Setup</p>', unsafe_allow_html=True)
    with st.form("strategy_form"):
        driver = st.selectbox("Driver", drivers, index=drivers.index("HAM") if "HAM" in drivers else 0)
        team = st.selectbox("Team", teams)
        circuit = st.selectbox("Circuit", circuits)
        year = st.selectbox("Season (driver/car form)", years, index=len(years) - 1)
        quali_position = st.number_input("Starting grid position", min_value=1, max_value=20, value=1)
        total_laps = st.number_input("Race distance (laps)", min_value=20, max_value=90, value=55)
        num_pit_stops = st.slider("Number of pit stops", min_value=1, max_value=3, value=1)
        submitted = st.form_submit_button("Predict Strategy", type="primary", use_container_width=True)

tab_strategy, tab_battles, tab_replay = st.tabs(["Strategy Simulator", "Predicted Battles", "Post-Race Replay"])

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
              Win probability is a heuristic blend of the model's raw prediction with grid-position
              baselines and driver form — see predict_strategy in phase_4.ipynb for the exact calibration.
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_battles:
    st.markdown('<p class="pitwall-label">Scan For Close Battles</p>', unsafe_allow_html=True)
    b_col1, b_col2, b_col3, b_col4 = st.columns([2, 2, 1, 1])
    with b_col1:
        battle_year = st.selectbox("Season", years, index=len(years) - 1, key="battle_year")
    with b_col2:
        battle_circuit = st.selectbox("Circuit", circuits, key="battle_circuit")
    with b_col3:
        battle_lap = st.number_input("Lap", min_value=1, max_value=90, value=10, key="battle_lap")
    with b_col4:
        st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
        battle_go = st.button("Scan Battles", type="primary", use_container_width=True)

    if battle_go:
        with st.spinner("Scanning for close battles..."):
            battles = predict_battles(int(battle_year), battle_circuit, int(battle_lap))
        st.markdown(render_battle_ticker(battles), unsafe_allow_html=True)
        if battles:
            st.markdown(render_battle_table(battles), unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="lights-panel">
              <div class="lights-caption">Pick a season, circuit, and lap, then scan for close battles</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="f1-footer">
          Overtake chance comes from battle_model in phase_3_battles.ipynb — gap, gap-closing
          rate, tyre-life/compound delta, speed-trap delta, and a circuit overtake-difficulty prior.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab_replay:
    st.markdown('<p class="pitwall-label">Replay A Finished Race</p>', unsafe_allow_html=True)
    r_col1, r_col2, r_col3 = st.columns([2, 2, 1])
    with r_col1:
        replay_year = st.selectbox("Season", years, index=len(years) - 1, key="replay_year")
    with r_col2:
        replay_circuit = st.selectbox("Circuit", circuits, key="replay_circuit")
    with r_col3:
        st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
        replay_go = st.button("Run Replay", type="primary", use_container_width=True)

    if replay_go:
        with st.spinner("Running live_model over every lap..."):
            replay_df = replay_race(int(replay_year), replay_circuit)

        if replay_df.empty:
            st.markdown(
                '<div class="lights-panel"><div class="lights-caption">No lap data found for that race</div></div>',
                unsafe_allow_html=True,
            )
        else:
            podium_finishers = (
                replay_df.drop_duplicates("Driver")
                .sort_values("FinalPosition")
                .head(3)["Driver"].tolist()
            )
            pivot = replay_df.pivot(index="LapNumber", columns="Driver", values="Probability")

            st.markdown(
                f'<p class="result-meta">Eventual podium: {", ".join(html.escape(d) for d in podium_finishers)}</p>',
                unsafe_allow_html=True,
            )
            podium_cols = [d for d in podium_finishers if d in pivot.columns]
            st.line_chart(pivot[podium_cols] if podium_cols else pivot)
            with st.expander("Full field"):
                st.line_chart(pivot)
    else:
        st.markdown(
            """
            <div class="lights-panel">
              <div class="lights-caption">Pick a finished race to replay its lap-by-lap podium probability</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="f1-footer">
          Predicted podium probability per lap comes from softmax over live_model's raw
          scores within that lap's field — see replay_race in phase_4.ipynb.
        </div>
        """,
        unsafe_allow_html=True,
    )
