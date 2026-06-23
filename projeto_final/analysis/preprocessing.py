"""Loads match_minute_metrics.csv and builds an evenly spaced 0-90 panel per match."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COUNT_COLS = ["pos_comments", "neg_comments", "neu_comments", "total_comments"]
EVENT_COLS = ["goals", "cards", "subs"]
SMOOTH_COLS = ["polarity", "wsi", "xt_home", "xt_away", "momentum"]


def load_match_minute_panel() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "match_minute_metrics.csv")

    full_index = pd.MultiIndex.from_product(
        [sorted(df["game_id"].unique()), range(0, 91)],
        names=["game_id", "match_minute"],
    )
    panel = df.set_index(["game_id", "match_minute"]).reindex(full_index).reset_index()

    panel[COUNT_COLS + EVENT_COLS] = panel[COUNT_COLS + EVENT_COLS].fillna(0.0)

    # A handful of matches dropped minutes with zero chat traffic upstream,
    # which also erased that minute's on-field metrics even though play
    # continued. Interpolating restores the even spacing CCF/Granger need
    # without inventing sentiment spikes.
    panel[SMOOTH_COLS] = panel.groupby("game_id", group_keys=False)[SMOOTH_COLS].apply(
        lambda g: g.interpolate(limit_direction="both")
    )

    panel["volume_total"] = panel[["pos_comments", "neg_comments", "neu_comments"]].sum(axis=1)
    panel["xt_diff"] = panel["xt_home"] - panel["xt_away"]

    return panel


def games_with_chat_data(panel: pd.DataFrame) -> pd.DataFrame:
    """Drops matches with zero chat volume end-to-end (no video_id was ever
    linked, so wsi/polarity/volume are all-zero placeholders, not signal)."""
    has_chat = panel.groupby("game_id")["volume_total"].transform("sum") > 0
    return panel[has_chat].reset_index(drop=True)
