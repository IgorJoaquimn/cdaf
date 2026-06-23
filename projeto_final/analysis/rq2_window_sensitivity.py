"""RQ2 - checagem de robustez do tamanho da janela pre-evento.

Mantemos o lado pos-evento fixo em t+5 (ja bem suportado pelo pico reativo
visto na janela padrao) e alongamos só o lado pre-evento (5, 8, 10 minutos),
testando em cada tamanho duas fases: "tardia" (t-2, t-1, igual ao roteiro
original) e "precoce" (os 3 minutos mais distantes da janela). Reportamos
todos os tamanhos testados -- a ideia nao e escolher o que "deu significativo"
e sim ver se o efeito de antecipacao se sustenta ao alongar o horizonte.
"""
from pathlib import Path

import pandas as pd

from preprocessing import games_with_chat_data, load_match_minute_panel
from rq2_event_study import (
    EVENT_TYPES,
    add_volume_norm,
    baseline_excluding_events,
    build_event_matrix,
    paired_tests,
    plot_event_study,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

PRE_WINDOW_SWEEP = [5, 8, 10]
POST_WINDOW = 5


def main() -> None:
    panel = games_with_chat_data(load_match_minute_panel())
    panel = add_volume_norm(panel)

    summary_rows = []
    for event_col, label in EVENT_TYPES.items():
        for pre in PRE_WINDOW_SWEEP:
            window = list(range(-pre, POST_WINDOW + 1))
            vol_df, wsi_df, meta_df = build_event_matrix(panel, event_col, window=window)
            print(f"\n=== {label}, janela pre-evento=-{pre}: {len(vol_df)} eventos validos ===")
            if vol_df.empty:
                continue

            plot_event_study(
                vol_df, wsi_df, f"{label} (pre=-{pre})",
                OUT_DIR / "figures" / f"rq2_window_sensitivity_{event_col}_pre{pre}.png",
            )

            baseline = baseline_excluding_events(panel, event_col, window=window)
            tests = paired_tests(vol_df, wsi_df, meta_df, baseline, window=window)
            for res in tests:
                print(f"-- {res['phase']} / {res['metric']}: n={res['n']} t_p={res['t_p']:.4f} wilcoxon_p={res['wilcoxon_p']:.4f}")
                summary_rows.append(dict(event_type=label, pre_window=pre, **res))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "tables" / "rq2_window_sensitivity_stats.csv", index=False)
    print("\n=== Resumo: n de eventos elegiveis por tamanho de janela ===")
    print(summary_df[["event_type", "pre_window", "phase", "metric", "n"]].drop_duplicates().to_string(index=False))


if __name__ == "__main__":
    main()
