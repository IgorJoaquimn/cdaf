"""RQ1: correlacao temporal entre sentimento do chat e desempenho em campo.

Roteiro (exps_cdaf.md): suaviza as series com media movel de 3 min, calcula a
Funcao de Correlacao Cruzada (CCF) por partida para lags de -5 a +5 min, agrega
o lag de maior |correlacao| entre as 63 partidas e roda o Teste de Causalidade
de Granger (maxlag=3) entre Campo (momentum / xT) e Sentimento (wsi / polarity).
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, pearsonr
from statsmodels.tsa.stattools import grangercausalitytests

from preprocessing import games_with_chat_data, load_match_minute_panel

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

LAGS = range(-5, 6)
SENTIMENT_VARS = ["wsi", "polarity"]
FIELD_VARS = {"momentum": "Momentum", "xt_diff": "xT (casa - fora)"}
ROLL_WINDOW = 3
GRANGER_MAXLAG = 3


def smooth(panel: pd.DataFrame) -> pd.DataFrame:
    cols = SENTIMENT_VARS + list(FIELD_VARS)
    smoothed = panel.groupby("game_id", group_keys=False)[cols].apply(
        lambda g: g.rolling(ROLL_WINDOW, center=True, min_periods=1).mean()
    )
    return pd.concat([panel[["game_id", "match_minute"]], smoothed], axis=1)


def ccf_for_match(sentiment: pd.Series, field: pd.Series, lags=LAGS) -> dict:
    # CCF(k) = Corr(Sentimento_t, Campo_{t-k}). pandas .shift(k) places the
    # value originally at t-k into row t, which lines the series up exactly
    # as the formula requires (k>0 -> campo leads sentimento by k minutos).
    out = {}
    for k in lags:
        shifted = field.shift(k)
        mask = sentiment.notna() & shifted.notna()
        if mask.sum() < 10 or sentiment[mask].std() == 0 or shifted[mask].std() == 0:
            out[k] = np.nan
            continue
        r, _ = pearsonr(sentiment[mask], shifted[mask])
        out[k] = r
    return out


def run_ccf(panel: pd.DataFrame, lags=LAGS) -> pd.DataFrame:
    rows = []
    for game_id, g in panel.groupby("game_id"):
        g = g.sort_values("match_minute")
        for sent_var in SENTIMENT_VARS:
            for field_var in FIELD_VARS:
                for k, r in ccf_for_match(g[sent_var], g[field_var], lags=lags).items():
                    rows.append(dict(game_id=game_id, sentiment=sent_var, field=field_var, lag=k, ccf=r))
    return pd.DataFrame(rows)


def summarize_ccf(ccf_df: pd.DataFrame) -> pd.DataFrame:
    return (
        ccf_df.groupby(["sentiment", "field", "lag"])["ccf"]
        .agg(mean_ccf="mean", std_ccf="std", n="count")
        .reset_index()
    )


def best_lag_per_match(ccf_df: pd.DataFrame) -> pd.DataFrame:
    valid = ccf_df.dropna(subset=["ccf"])
    idx = valid.groupby(["game_id", "sentiment", "field"])["ccf"].apply(lambda s: s.abs().idxmax())
    return valid.loc[idx.values].reset_index(drop=True)


def plot_ccf(summary: pd.DataFrame, sent_var: str, field_var: str, path: Path) -> None:
    sub = summary[(summary.sentiment == sent_var) & (summary.field == field_var)].sort_values("lag")
    n_matches = int(sub["n"].max())
    sem = sub["std_ccf"] / np.sqrt(sub["n"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(sub["lag"], sub["mean_ccf"], yerr=sem, capsize=4, color="#3b6fa0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Lag k (min) -- CCF(k) = Corr(Sentimento_t, Campo_(t-k))")
    ax.set_ylabel(f"Correlacao cruzada media ({n_matches} partidas com chat)")
    ax.set_title(f"CCF media: {sent_var.upper()} x {FIELD_VARS[field_var]}")
    ax.set_xticks(sorted(sub["lag"].unique()))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_granger(panel: pd.DataFrame, maxlag: int = GRANGER_MAXLAG) -> pd.DataFrame:
    rows = []
    for game_id, g in panel.groupby("game_id"):
        g = g.sort_values("match_minute")
        for sent_var in SENTIMENT_VARS:
            for field_var in FIELD_VARS:
                for direction, cols in [
                    (f"{field_var}->{sent_var}", [sent_var, field_var]),
                    (f"{sent_var}->{field_var}", [field_var, sent_var]),
                ]:
                    data = g[cols].dropna()
                    if len(data) < maxlag + 10 or (data.std() == 0).any():
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            res = grangercausalitytests(data.values, maxlag=maxlag, verbose=False)
                    except Exception:
                        continue
                    pval = res[maxlag][0]["ssr_ftest"][1]
                    rows.append(
                        dict(game_id=game_id, sentiment=sent_var, field=field_var, direction=direction, p_value=pval)
                    )
    return pd.DataFrame(rows)


def summarize_granger(granger_df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        _, combined_p = combine_pvalues(g["p_value"].clip(1e-12, 1), method="fisher")
        return pd.Series(
            {
                "pct_partidas_sig_5pct": (g["p_value"] < 0.05).mean(),
                "fisher_combined_p": combined_p,
                "n_partidas": len(g),
            }
        )

    return (
        granger_df.groupby(["sentiment", "field", "direction"])[["p_value"]]
        .apply(agg)
        .reset_index()
    )


def main() -> None:
    panel = load_match_minute_panel()
    panel = games_with_chat_data(panel)
    print(f"Partidas com sinal real de chat: {panel['game_id'].nunique()} de 63")
    smoothed = smooth(panel)

    ccf_df = run_ccf(smoothed)
    summary = summarize_ccf(ccf_df)
    summary.to_csv(OUT_DIR / "tables" / "rq1_ccf_summary.csv", index=False)

    best = best_lag_per_match(ccf_df)
    best.to_csv(OUT_DIR / "tables" / "rq1_best_lag_per_match.csv", index=False)
    print("=== Lag de maior |CCF| por partida (media +/- desvio) ===")
    print(best.groupby(["sentiment", "field"])["lag"].agg(["mean", "std"]))
    print("\n=== Coeficiente de CCF no lag otimo (media +/- desvio) ===")
    print(best.groupby(["sentiment", "field"])["ccf"].agg(["mean", "std"]))

    for sent_var in SENTIMENT_VARS:
        for field_var in FIELD_VARS:
            plot_ccf(summary, sent_var, field_var, OUT_DIR / "figures" / f"rq1_ccf_{sent_var}_{field_var}.png")

    granger_df = run_granger(smoothed)
    granger_df.to_csv(OUT_DIR / "tables" / "rq1_granger_raw.csv", index=False)
    granger_summary = summarize_granger(granger_df)
    granger_summary.to_csv(OUT_DIR / "tables" / "rq1_granger_summary.csv", index=False)
    print("\n=== Causalidade de Granger (maxlag=3), agregada por partida ===")
    print(granger_summary.to_string(index=False))


if __name__ == "__main__":
    main()
