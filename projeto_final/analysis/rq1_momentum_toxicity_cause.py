"""Analyzes whether the change of momentum causes a change in toxicity or message volume.
Applies Z-score normalization per game and first-differencing for stationarity.
Performs CCF (cross-correlation) and Granger causality tests.
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
GRANGER_MAXLAG = 3

def compute_toxicity_and_normalize(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    # Toxicity score: negative comments / total comments
    df["toxicity_ratio"] = df["neg_comments"] / (df["volume_total"] + 1e-5)
    
    # Standardize (Z-score) the variables within each game
    cols_to_scale = ["momentum", "toxicity_ratio", "volume_total"]
    for col in cols_to_scale:
        df[col] = df.groupby("game_id")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    
    # Calculate first differences (changes)
    for col in cols_to_scale:
        df[f"{col}_diff"] = df.groupby("game_id")[col].diff()
        
    return df

def ccf_for_match(s1: pd.Series, s2: pd.Series, lags=LAGS) -> dict:
    out = {}
    for k in lags:
        shifted = s2.shift(k)
        mask = s1.notna() & shifted.notna()
        if mask.sum() < 10 or s1[mask].std() == 0 or shifted[mask].std() == 0:
            out[k] = np.nan
            continue
        r, _ = pearsonr(s1[mask], shifted[mask])
        out[k] = r
    return out

def run_ccf_diff(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [
        ("toxicity_ratio_diff", "momentum_diff", "Toxicity Change", "Momentum Change"),
        ("volume_total_diff", "momentum_diff", "Volume Change", "Momentum Change"),
    ]
    for game_id, g in panel.groupby("game_id"):
        g = g.sort_values("match_minute")
        for s_var, f_var, s_label, f_label in pairs:
            ccf_vals = ccf_for_match(g[s_var], g[f_var], lags=LAGS)
            for k, r in ccf_vals.items():
                rows.append(dict(
                    game_id=game_id,
                    sentiment_var=s_var,
                    field_var=f_var,
                    sentiment_label=s_label,
                    field_label=f_label,
                    lag=k,
                    ccf=r
                ))
    return pd.DataFrame(rows)

def plot_ccf(summary: pd.DataFrame, s_var: str, f_var: str, s_label: str, f_label: str, path: Path) -> None:
    sub = summary[(summary.sentiment_var == s_var) & (summary.field_var == f_var)].sort_values("lag")
    n_matches = int(sub["n"].max())
    sem = sub["std_ccf"] / np.sqrt(sub["n"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(sub["lag"], sub["mean_ccf"], yerr=sem, capsize=4, color="#a93226" if "Toxicity" in s_label else "#28b463")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Lag k (min) -- CCF(k) = Corr(Sentimento_t, Campo_(t-k))")
    ax.set_ylabel(f"Correlacao cruzada media ({n_matches} partidas)")
    ax.set_title(f"CCF de Mudancas: {s_label} x {f_label}")
    ax.set_xticks(sorted(sub["lag"].unique()))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def run_granger_diff(panel: pd.DataFrame, maxlag: int = GRANGER_MAXLAG) -> pd.DataFrame:
    rows = []
    pairs = [
        ("toxicity_ratio_diff", "momentum_diff", "Toxicity Change", "Momentum Change"),
        ("volume_total_diff", "momentum_diff", "Volume Change", "Momentum Change"),
    ]
    for game_id, g in panel.groupby("game_id"):
        g = g.sort_values("match_minute")
        for s_var, f_var, s_label, f_label in pairs:
            # Test in both directions:
            # 1. Momentum Change -> Sentiment Change (Cols: [s_var, f_var])
            # 2. Sentiment Change -> Momentum Change (Cols: [f_var, s_var])
            for direction, cols in [
                (f"{f_label}->{s_label}", [s_var, f_var]),
                (f"{s_label}->{f_label}", [f_var, s_var]),
            ]:
                data = g[cols].dropna()
                if len(data) < maxlag + 10 or (data.std() == 0).any():
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        res = grangercausalitytests(data.values, maxlag=maxlag, verbose=False)
                    pval = res[maxlag][0]["ssr_ftest"][1]
                    rows.append(dict(
                        game_id=game_id,
                        sentiment_var=s_var,
                        field_var=f_var,
                        direction=direction,
                        p_value=pval
                    ))
                except Exception:
                    continue
    return pd.DataFrame(rows)

def summarize_granger(granger_df: pd.DataFrame) -> pd.DataFrame:
    def agg(g):
        _, combined_p = combine_pvalues(g["p_value"].clip(1e-12, 1), method="fisher")
        return pd.Series({
            "pct_partidas_sig_5pct": (g["p_value"] < 0.05).mean(),
            "fisher_combined_p": combined_p,
            "n_partidas": len(g)
        })
    return granger_df.groupby(["sentiment_var", "field_var", "direction"]).apply(agg).reset_index()

def main():
    panel = load_match_minute_panel()
    panel = games_with_chat_data(panel)
    print(f"Partidas com chat: {panel['game_id'].nunique()}")
    
    df_normalized = compute_toxicity_and_normalize(panel)
    
    # Run CCF on differences
    ccf_df = run_ccf_diff(df_normalized)
    ccf_summary = ccf_df.groupby(["sentiment_var", "field_var", "sentiment_label", "field_label", "lag"])["ccf"].agg(
        mean_ccf="mean", std_ccf="std", n="count"
    ).reset_index()
    ccf_summary.to_csv(OUT_DIR / "tables" / "rq1_diff_ccf_summary.csv", index=False)
    
    # Plot CCF on differences
    plot_ccf(
        ccf_summary, "toxicity_ratio_diff", "momentum_diff", "Toxicity Change", "Momentum Change",
        OUT_DIR / "figures" / "rq1_ccf_diff_momentum_toxicity.png"
    )
    plot_ccf(
        ccf_summary, "volume_total_diff", "momentum_diff", "Volume Change", "Momentum Change",
        OUT_DIR / "figures" / "rq1_ccf_diff_momentum_volume.png"
    )
    
    # Run Granger Causality on differences
    granger_df = run_granger_diff(df_normalized)
    granger_df.to_csv(OUT_DIR / "tables" / "rq1_diff_granger_raw.csv", index=False)
    granger_summary = summarize_granger(granger_df)
    granger_summary.to_csv(OUT_DIR / "tables" / "rq1_diff_granger_summary.csv", index=False)
    
    print("\n=== Granger Causality on First Differences (maxlag=3) ===")
    print(granger_summary.to_string(index=False))

if __name__ == "__main__":
    main()
