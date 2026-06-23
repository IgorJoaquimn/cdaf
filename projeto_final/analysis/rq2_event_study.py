"""RQ2: estudo de evento (time-to-event) ao redor de gols e cartoes.

Roteiro (exps_cdaf.md): para cada gol/cartao (minuto >= 5, para nao truncar a
janela), extrai uma janela de -5 a +5 minutos do volume normalizado de
mensagens e do WSI absoluto, agrega media + IC95% por instante relativo, e
compara a fase pre-evento (t-2, t-1) contra a fase basal da partida (minutos
fora de qualquer janela de evento) com teste t pareado e Wilcoxon.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from preprocessing import games_with_chat_data, load_match_minute_panel

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

WINDOW = list(range(-5, 6))
EVENT_TYPES = {"goals": "Gols", "cards": "Cartoes"}


def add_volume_norm(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    game_mean_volume = panel.groupby("game_id")["volume_total"].transform("mean")
    panel["volume_norm"] = panel["volume_total"] / game_mean_volume
    return panel


def build_event_matrix(panel: pd.DataFrame, event_col: str, window=WINDOW):
    min_minute = -min(window)
    max_minute = 90 - max(window)
    events = panel[
        (panel[event_col] >= 1)
        & (panel["match_minute"] >= min_minute)
        & (panel["match_minute"] <= max_minute)
    ]
    indexed = panel.set_index(["game_id", "match_minute"])
    rows_vol, rows_wsi, meta = [], [], []
    for _, ev in events.iterrows():
        gid, minute = ev["game_id"], ev["match_minute"]
        window_vol = [indexed.loc[(gid, minute + k), "volume_norm"] for k in window]
        window_wsi = [indexed.loc[(gid, minute + k), "wsi"] for k in window]
        rows_vol.append(window_vol)
        rows_wsi.append(window_wsi)
        meta.append((gid, minute))
    vol_df = pd.DataFrame(rows_vol, columns=window)
    wsi_df = pd.DataFrame(rows_wsi, columns=window)
    meta_df = pd.DataFrame(meta, columns=["game_id", "event_minute"])
    return vol_df, wsi_df, meta_df


def mean_ci(df: pd.DataFrame, confidence: float = 0.95):
    n = df.shape[0]
    mean = df.mean()
    sem = df.sem()
    tcrit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, mean - tcrit * sem, mean + tcrit * sem


def plot_event_study(vol_df: pd.DataFrame, wsi_df: pd.DataFrame, label: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    panels = [
        (axes[0], vol_df, f"Volume normalizado de mensagens -- {label} (n={len(vol_df)})", "Volume / media da partida"),
        (axes[1], wsi_df, f"WSI absoluto -- {label} (n={len(wsi_df)})", "WSI"),
    ]
    for ax, df, title, ylabel in panels:
        mean, lo, hi = mean_ci(df)
        x = [int(c) for c in df.columns]
        ax.plot(x, mean, marker="o", color="#b03a2e")
        ax.fill_between(x, lo, hi, alpha=0.25, color="#b03a2e")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Minutos relativos ao evento (t=0)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(list(df.columns))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def baseline_excluding_events(panel: pd.DataFrame, event_col: str, window=WINDOW) -> pd.DataFrame:
    ev_rows = panel[panel[event_col] >= 1]
    excluded = {
        (gid, minute + k) for gid, minute in zip(ev_rows["game_id"], ev_rows["match_minute"]) for k in window
    }
    keys = list(zip(panel["game_id"], panel["match_minute"]))
    mask = ~pd.Series(keys, index=panel.index).isin(excluded)
    baseline = (
        panel[mask]
        .groupby("game_id")[["volume_norm", "wsi"]]
        .mean()
        .rename(columns={"volume_norm": "baseline_volume", "wsi": "baseline_wsi"})
    )
    return baseline


def paired_tests(vol_df: pd.DataFrame, wsi_df: pd.DataFrame, meta_df: pd.DataFrame, baseline: pd.DataFrame, window=WINDOW) -> list:
    # "Tardia" (t-2, t-1) e igual ao roteiro original. "Precoce" usa os 3
    # minutos mais distantes da janela testada, para checar se o efeito de
    # antecipacao aparece mais cedo quando a janela pre-evento e alongada.
    late_cols = [-2, -1]
    early_cols = [window[0], window[0] + 1, window[0] + 2]
    merged = meta_df.join(baseline, on="game_id")

    phases = [("tardia (t-2,t-1)", late_cols)]
    if max(early_cols) < min(late_cols) - 1:
        phases.append((f"precoce (t{early_cols[0]}..t{early_cols[-1]})", early_cols))

    results = []
    for phase_label, cols in phases:
        pre_vol = vol_df[cols].mean(axis=1)
        pre_wsi = wsi_df[cols].mean(axis=1)
        for metric, pre, base_col in [("volume", pre_vol, "baseline_volume"), ("wsi", pre_wsi, "baseline_wsi")]:
            base = merged[base_col]
            diff = pre - base
            t_stat, t_p = stats.ttest_rel(pre, base)
            if (diff != 0).any():
                w_stat, w_p = stats.wilcoxon(diff)
            else:
                w_stat, w_p = np.nan, np.nan
            results.append(
                dict(
                    phase=phase_label,
                    metric=metric,
                    n=len(pre),
                    mean_pre_evento=pre.mean(),
                    mean_basal=base.mean(),
                    t_stat=t_stat,
                    t_p=t_p,
                    wilcoxon_stat=w_stat,
                    wilcoxon_p=w_p,
                )
            )
    return results


def main() -> None:
    panel = load_match_minute_panel()
    panel = games_with_chat_data(panel)
    print(f"Partidas com sinal real de chat: {panel['game_id'].nunique()} de 63")
    panel = add_volume_norm(panel)

    summary_rows = []
    for event_col, label in EVENT_TYPES.items():
        vol_df, wsi_df, meta_df = build_event_matrix(panel, event_col)
        print(f"\n=== {label}: {len(vol_df)} eventos validos (minuto 5-85) ===")
        if vol_df.empty:
            continue

        plot_event_study(vol_df, wsi_df, label, OUT_DIR / "figures" / f"rq2_event_study_{event_col}.png")

        baseline = baseline_excluding_events(panel, event_col)
        tests = paired_tests(vol_df, wsi_df, meta_df, baseline)
        for res in tests:
            print(f"-- {res['phase']} / {res['metric']}: {res}")
            summary_rows.append(dict(event_type=label, **res))

        vol_out = vol_df.copy()
        vol_out["game_id"] = meta_df["game_id"]
        vol_out["event_minute"] = meta_df["event_minute"]
        vol_out.to_csv(OUT_DIR / "tables" / f"rq2_volume_matrix_{event_col}.csv", index=False)

        wsi_out = wsi_df.copy()
        wsi_out["game_id"] = meta_df["game_id"]
        wsi_out["event_minute"] = meta_df["event_minute"]
        wsi_out.to_csv(OUT_DIR / "tables" / f"rq2_wsi_matrix_{event_col}.csv", index=False)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "tables" / "rq2_event_study_stats.csv", index=False)


if __name__ == "__main__":
    main()
