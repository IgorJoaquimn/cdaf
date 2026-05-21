#!/usr/bin/env python3
"""
calculate_xt_momentum.py
========================
Computes Expected Threat (xT) and Momentum metrics from Gradient event data.

Outputs (saved to data/processed/):
  - xt_grid.npy          : 16×12 xT grid (calibrated on all 306 games)
  - events_xt.parquet    : per-event DataFrame for the 63 mapped games, with xT column
  - momentum.parquet     : per-game per-minute Momentum (rolling 5 min)

Usage:
  uv run python src/metrics/calculate_xt_momentum.py

References:
  - Singh, K. (2019). Introducing Expected Threat (xT).
    https://karun.in/blog/expected-threat.html
"""

import json
import os
import glob

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVENTS_DIR = os.path.join(ROOT, "data", "events", "2024-2025")
VIDEO_MAPPING_PATH = os.path.join(ROOT, "data", "processed", "video_mapping.json")
OUT_DIR = os.path.join(ROOT, "data", "processed")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
X_BINS = 16    # columns (pitch length direction, 0-105 m)
Y_BINS = 12    # rows    (pitch width direction, 0-68 m)
XT_ITERATIONS = 8
MOMENTUM_WINDOW = 5  # minutes

PE_TYPE_LABELS = {
    "SH": "Chute", "PA": "Passe", "BC": "Condução",
    "CR": "Cruzamento", "CH": "Duelo", "CL": "Corte",
    "RE": "Recuperação", "IT": "Interceptação", "TC": "Controle",
}


# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------
def normalize_attack(
    x: float, y: float, home_team: bool, period: int
) -> tuple[float, float]:
    """
    Convert raw Gradient XY coordinates to attacking-direction coordinates
    where x_atk=105 is the opponent's goal.

    Coordinate system (Gradient):
      - Centre of pitch = (0, 0)
      - X ∈ [-53.5, 53.5]  (length axis)
      - Y ∈ [-34,   34  ]  (width  axis)
      - Home team attacks towards +X in Period 1, −X in Period 2.
    """
    x_raw = x + 53.5   # → [0, 107]
    y_raw = y + 34.0   # → [0,  68]

    # Home P1 or Away P2: attack towards increasing X
    if (home_team is True and period == 1) or (home_team is False and period == 2):
        return x_raw, y_raw
    else:
        return 105.0 - x_raw, 68.0 - y_raw


# ---------------------------------------------------------------------------
# Grid utilities
# ---------------------------------------------------------------------------
x_edges = np.linspace(0, 105, X_BINS + 1)
y_edges = np.linspace(0, 68, Y_BINS + 1)


def bin_coords(
    x: "np.ndarray | float", y: "np.ndarray | float"
) -> tuple["np.ndarray | int", "np.ndarray | int"]:
    """Discretise (x, y) coordinates into xT grid indices."""
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, X_BINS - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, Y_BINS - 1)
    return xi, yi


# ---------------------------------------------------------------------------
# Step 1 – Parse a single JSON event file
# ---------------------------------------------------------------------------
def parse_events(path: str, video_id: str | None = None) -> tuple[list, list]:
    """
    Parse one Gradient JSON file.

    Returns
    -------
    event_rows    : list of dicts (possession events with coordinates)
    discrete_rows : list of dicts (goals, cards, substitutions)
    """
    game_id = os.path.splitext(os.path.basename(path))[0]

    with open(path) as f:
        data = json.load(f)

    event_rows: list[dict] = []
    discrete_rows: list[dict] = []

    for e in data:
        ge = e.get("GAME_EVENTS", {})
        pe = e.get("POSSESSION_EVENTS", {})
        fouls = e.get("FOULS", {})
        ball_list = e.get("BALL") or []
        ball = ball_list[0] if ball_list else {}

        period = ge.get("PERIOD")
        clock = ge.get("START_GAME_CLOCK")   # seconds from kick-off of that half
        home_team = ge.get("HOME_TEAM")
        pe_type = ge_type = None
        pe_type = pe.get("POSSESSION_EVENT_TYPE")
        ge_type = ge.get("GAME_EVENT_TYPE")
        x, y = ball.get("X"), ball.get("Y")

        # Match minute (Period 1 capped at 45, Period 2 starts at 46 and capped at 90 to prevent overlap)
        match_minute: int | None = None
        if period and clock is not None:
            if period == 1:
                match_minute = min(int(clock // 60), 45)
            elif period == 2:
                match_minute = min(int((clock - 2700) // 60) + 46, 90)

        # --- Possession events (for xT / Momentum) ---
        if pe_type and period and clock is not None and x is not None and home_team is not None:
            x_atk, y_atk = normalize_attack(x, y, home_team, period)
            event_rows.append(
                {
                    "video_id": video_id,
                    "game_id": game_id,
                    "period": period,
                    "match_minute": match_minute,
                    "clock_s": clock,
                    "home_team": home_team,
                    "pe_type": pe_type,
                    "pe_type_label": PE_TYPE_LABELS.get(pe_type, pe_type),
                    "shot_outcome": pe.get("SHOT_OUTCOME_TYPE"),
                    "pass_outcome": pe.get("PASS_OUTCOME_TYPE"),
                    "x_raw": x,
                    "y_raw": y,
                    "x_atk": x_atk,
                    "y_atk": y_atk,
                }
            )

        # --- Discrete events (goals, cards, subs) ---
        if ge_type == "G" or pe.get("SHOT_OUTCOME_TYPE") == "G":
            discrete_rows.append(
                {
                    "video_id": video_id,
                    "game_id": game_id,
                    "match_minute": match_minute,
                    "home_team": home_team,
                    "event_type": "Gol",
                }
            )
        if fouls.get("FINAL_FOUL_OUTCOME_TYPE") == "Y":
            discrete_rows.append(
                {
                    "video_id": video_id,
                    "game_id": game_id,
                    "match_minute": match_minute,
                    "home_team": home_team,
                    "event_type": "Cartão",
                }
            )
        if ge_type == "SUB":
            discrete_rows.append(
                {
                    "video_id": video_id,
                    "game_id": game_id,
                    "match_minute": match_minute,
                    "home_team": home_team,
                    "event_type": "Substituição",
                }
            )

    return event_rows, discrete_rows


# ---------------------------------------------------------------------------
# Step 2 – Build xT grid from ALL 306 season games
# ---------------------------------------------------------------------------
def build_xt_grid() -> np.ndarray:
    """
    Compute an empirical xT grid using all available Gradient JSON files.

    Algorithm (Karun Singh, iterative):
      xT(x,y) = shoot_rate(x,y) * goal_rate(x,y)
                + (1 - shoot_rate(x,y)) * xT_prev(x,y)

    Returns
    -------
    xt_grid : np.ndarray of shape (X_BINS, Y_BINS)
    """
    total_count = np.zeros((X_BINS, Y_BINS))
    shot_count = np.zeros((X_BINS, Y_BINS))
    goal_count = np.zeros((X_BINS, Y_BINS))

    all_files = sorted(glob.glob(os.path.join(EVENTS_DIR, "*.json")))
    print(f"  Calibrating xT grid on {len(all_files)} games …")

    for path in all_files:
        rows, _ = parse_events(path)
        for r in rows:
            xi, yi = bin_coords(r["x_atk"], r["y_atk"])
            total_count[xi, yi] += 1
            if r["pe_type"] == "SH":
                shot_count[xi, yi] += 1
            if r.get("shot_outcome") == "G":
                goal_count[xi, yi] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        shoot_rate = np.where(total_count > 0, shot_count / total_count, 0.0)
        goal_rate = np.where(shot_count > 0, goal_count / shot_count, 0.0)

    xt_grid = np.zeros((X_BINS, Y_BINS))
    for _ in range(XT_ITERATIONS):
        xt_grid = shoot_rate * goal_rate + (1.0 - shoot_rate) * xt_grid

    return xt_grid


# ---------------------------------------------------------------------------
# Step 3 – Assign xT to each event and compute per-minute Momentum
# ---------------------------------------------------------------------------
def assign_xt(df: pd.DataFrame, xt_grid: np.ndarray) -> pd.DataFrame:
    """Vectorised xT assignment."""
    xi, yi = bin_coords(df["x_atk"].values, df["y_atk"].values)
    df = df.copy()
    df["xt"] = xt_grid[xi, yi]
    return df


def compute_momentum(df_ev: pd.DataFrame, window: int = MOMENTUM_WINDOW) -> pd.DataFrame:
    """
    Compute rolling-window Momentum per game per minute.

    Momentum(t) = sum(xT_home, t-W..t) - sum(xT_away, t-W..t)

    Parameters
    ----------
    df_ev  : DataFrame with columns [game_id, match_minute, home_team, xt]
    window : rolling window size in minutes

    Returns
    -------
    DataFrame with columns [game_id, match_minute, xt_home, xt_away, momentum]
    """
    xt_min = (
        df_ev.groupby(["game_id", "match_minute", "home_team"])["xt"]
        .sum()
        .reset_index()
    )
    xt_home = (
        xt_min[xt_min["home_team"] == True][["game_id", "match_minute", "xt"]]
        .rename(columns={"xt": "xt_home"})
    )
    xt_away = (
        xt_min[xt_min["home_team"] == False][["game_id", "match_minute", "xt"]]
        .rename(columns={"xt": "xt_away"})
    )
    merged = xt_home.merge(xt_away, on=["game_id", "match_minute"], how="outer").fillna(0.0)

    results = []
    for gid, gdf in merged.groupby("game_id"):
        gdf = gdf.sort_values("match_minute").set_index("match_minute")
        gdf["momentum"] = (
            gdf["xt_home"].rolling(window, min_periods=1).sum()
            - gdf["xt_away"].rolling(window, min_periods=1).sum()
        )
        gdf["game_id"] = gid
        results.append(gdf.reset_index())

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load video mapping
    with open(VIDEO_MAPPING_PATH) as f:
        video_mapping: dict[str, str] = json.load(f)
    print(f"Loaded mapping: {len(video_mapping)} videos → game IDs")

    # 2. Build xT grid (all 306 games)
    xt_grid = build_xt_grid()
    xt_path = os.path.join(OUT_DIR, "xt_grid.npy")
    np.save(xt_path, xt_grid)
    print(f"  xT grid saved → {xt_path}  (max xT: {xt_grid.max():.4f})")

    # 3. Parse the 63 mapped games
    print(f"Parsing {len(video_mapping)} mapped games …")
    all_event_rows: list[dict] = []
    all_discrete_rows: list[dict] = []

    for video_id, game_id in video_mapping.items():
        path = os.path.join(EVENTS_DIR, f"{game_id}.json")
        ev, disc = parse_events(path, video_id=video_id)
        all_event_rows.extend(ev)
        all_discrete_rows.extend(disc)

    df_ev = pd.DataFrame(all_event_rows)
    df_disc = pd.DataFrame(all_discrete_rows)
    print(f"  Possession events : {len(df_ev):,}")
    print(f"  Discrete events   : {len(df_disc):,}")

    # 4. Assign xT
    df_ev_valid = assign_xt(df_ev.dropna(subset=["x_atk", "y_atk"]), xt_grid)

    ev_path = os.path.join(OUT_DIR, "events_xt.parquet")
    df_ev_valid.to_parquet(ev_path, index=False)
    print(f"  Events+xT saved   → {ev_path}")

    disc_path = os.path.join(OUT_DIR, "discrete_events.parquet")
    df_disc.to_parquet(disc_path, index=False)
    print(f"  Discrete events   → {disc_path}")

    # 5. Compute Momentum
    df_mom = compute_momentum(df_ev_valid)
    mom_path = os.path.join(OUT_DIR, "momentum.parquet")
    df_mom.to_parquet(mom_path, index=False)
    print(f"  Momentum saved    → {mom_path}")
    print("Done.")


if __name__ == "__main__":
    main()
