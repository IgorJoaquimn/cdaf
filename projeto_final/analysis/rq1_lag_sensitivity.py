"""RQ1 - checagem de robustez do tamanho da janela de lag da CCF.

Em vez de fixar k em -5..+5 (exps_cdaf.md) e comparar com 1-2 alternativas
escolhidas a dedo, varremos varios tamanhos de janela (maxlag) e reportamos
TODOS os resultados: se o pico da CCF agregada continua proximo de +1..+3 e o
"lag de maior |CCF| por partida" para de se acumular exatamente na borda
conforme a janela cresce, o efeito e real. Se o pico/borda persegue o novo
extremo a cada janela maior, o que vimos em +-5 era ruido de borda.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import games_with_chat_data, load_match_minute_panel
from rq1_temporal_correlation import (
    FIELD_VARS,
    SENTIMENT_VARS,
    best_lag_per_match,
    run_ccf,
    smooth,
    summarize_ccf,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

MAXLAG_SWEEP = [3, 5, 8, 10]


def diagnostics_for_maxlag(smoothed: pd.DataFrame, maxlag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lags = range(-maxlag, maxlag + 1)
    ccf_df = run_ccf(smoothed, lags=lags)
    summary = summarize_ccf(ccf_df).assign(maxlag=maxlag)
    best = best_lag_per_match(ccf_df).assign(maxlag=maxlag)
    return summary, best


def plot_full_curve(all_summary: pd.DataFrame, sent_var: str, field_var: str, path: Path) -> None:
    # CCF(k) para um k fixo nao depende de quantos outros lags foram testados
    # junto -- por isso plotamos uma unica curva (na maior janela) e marcamos
    # onde cada janela menor teria cortado, em vez de sobrepor curvas
    # identicas.
    biggest = max(MAXLAG_SWEEP)
    sub = all_summary[
        (all_summary.sentiment == sent_var) & (all_summary.field == field_var) & (all_summary.maxlag == biggest)
    ].sort_values("lag")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sub["lag"], sub["mean_ccf"], marker="o", color="#3b6fa0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    for maxlag in MAXLAG_SWEEP[:-1]:
        ax.axvspan(maxlag + 0.1, biggest, alpha=0.0)
        ax.axvline(maxlag, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axvline(-maxlag, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.annotate(f"corte\nmaxlag={maxlag}", (maxlag, ax.get_ylim()[1] * 0.9), fontsize=7, ha="center")
    ax.set_xlabel("Lag k (min)")
    ax.set_ylabel("Correlacao cruzada media")
    ax.set_title(f"Curva completa de CCF (janela maxima testada): {sent_var.upper()} x {FIELD_VARS[field_var]}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_edge_diagnostic(diag_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for sent_var in SENTIMENT_VARS:
        for field_var in FIELD_VARS:
            sub = diag_df[(diag_df.sentiment == sent_var) & (diag_df.field == field_var)].sort_values("maxlag")
            ax.plot(sub["maxlag"], sub["pct_best_lag_na_borda"], marker="o", label=f"{sent_var} x {field_var}")
    chance = [2 / (2 * m + 1) for m in MAXLAG_SWEEP]
    ax.plot(MAXLAG_SWEEP, chance, linestyle="--", color="black", label="chance (ruido uniforme)")
    ax.set_xlabel("Tamanho da janela testada (maxlag)")
    ax.set_ylabel("% de partidas com lag otimo exatamente na borda")
    ax.set_title("Atracao pela borda: lag de maior |CCF| por partida vs. tamanho da janela")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    panel = games_with_chat_data(load_match_minute_panel())
    smoothed = smooth(panel)

    all_summaries, all_bests = [], []
    for maxlag in MAXLAG_SWEEP:
        summary, best = diagnostics_for_maxlag(smoothed, maxlag)
        all_summaries.append(summary)
        all_bests.append(best)
    all_summary = pd.concat(all_summaries, ignore_index=True)
    all_best = pd.concat(all_bests, ignore_index=True)

    all_summary.to_csv(OUT_DIR / "tables" / "rq1_lag_sensitivity_ccf.csv", index=False)
    all_best.to_csv(OUT_DIR / "tables" / "rq1_lag_sensitivity_best_lag.csv", index=False)

    diag_rows = []
    for maxlag in MAXLAG_SWEEP:
        for sent_var in SENTIMENT_VARS:
            for field_var in FIELD_VARS:
                sub_summary = all_summary[
                    (all_summary.maxlag == maxlag) & (all_summary.sentiment == sent_var) & (all_summary.field == field_var)
                ]
                peak_row = sub_summary.loc[sub_summary["mean_ccf"].idxmax()]
                sub_best = all_best[
                    (all_best.maxlag == maxlag) & (all_best.sentiment == sent_var) & (all_best.field == field_var)
                ]
                pct_edge = (sub_best["lag"].abs() == maxlag).mean()
                diag_rows.append(
                    dict(
                        sentiment=sent_var,
                        field=field_var,
                        maxlag=maxlag,
                        peak_lag=int(peak_row["lag"]),
                        peak_mean_ccf=peak_row["mean_ccf"],
                        pct_best_lag_na_borda=pct_edge,
                        n_partidas=int(sub_best.shape[0]),
                    )
                )

    for sent_var in SENTIMENT_VARS:
        for field_var in FIELD_VARS:
            plot_full_curve(
                all_summary, sent_var, field_var,
                OUT_DIR / "figures" / f"rq1_lag_sensitivity_{sent_var}_{field_var}.png",
            )

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(OUT_DIR / "tables" / "rq1_lag_sensitivity_diagnostics.csv", index=False)
    plot_edge_diagnostic(diag_df, OUT_DIR / "figures" / "rq1_lag_sensitivity_edge_attraction.png")
    print("=== Sensibilidade ao tamanho da janela de lag (CCF) ===")
    print(diag_df.to_string(index=False))


if __name__ == "__main__":
    main()
