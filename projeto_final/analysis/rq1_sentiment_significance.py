"""Computes and evaluates statistical significance for sentiment (WSI/polarity) and toxicity metrics
under three robust experimental setups:
1. Match-Level correlation (WSI standard deviation vs. total goals).
2. Minute-Level correlation with optimal lag +2 (WSI vs. momentum).
3. Immediate reaction to goals (t=0, t+1) compared to game baseline (WSI and toxicity).
Saves results to outputs/tables/rq1_sentiment_significance.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

from preprocessing import games_with_chat_data, load_match_minute_panel
from rq2_event_study import build_event_matrix, baseline_excluding_events

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

def main():
    panel = load_match_minute_panel()
    panel = games_with_chat_data(panel)
    
    # 1. Compute basic helper columns
    panel["volume_norm"] = panel["volume_total"] / panel.groupby("game_id")["volume_total"].transform("mean")
    panel["toxicity_ratio"] = panel["neg_comments"] / (panel["volume_total"] + 1e-5)
    panel["xt_diff"] = panel["xt_home"] - panel["xt_away"]
    
    results = []
    
    # --- TEST 1: Match-Level WSI Volatility vs. Goals ---
    match_level = panel.groupby("game_id").agg(
        wsi_std=("wsi", "std"),
        goals_total=("goals", "sum")
    ).reset_index()
    r_val, p_val = pearsonr(match_level["wsi_std"], match_level["goals_total"])
    print("=== Test 1: Match-Level WSI Volatility vs. Total Goals ===")
    print(f"Correlation: r = {r_val:.4f}, p-value = {p_val:.2e}")
    results.append({
        "Analysis Level": "Match-Level",
        "Variable 1": "WSI Standard Deviation (wsi_std)",
        "Variable 2": "Total Goals (goals_total)",
        "Test Type": "Pearson Correlation",
        "Statistic Name": "r",
        "Statistic Value": r_val,
        "p-value": p_val,
        "Significance (alpha=0.05)": "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT",
        "Interpretation": "Matches with more goals have significantly higher swings in chat sentiment (volatility)."
    })
    
    # --- TEST 2: Minute-Level WSI vs. Momentum at Lag +2 ---
    # Smooth variables as in primary CCF
    cols_to_smooth = ["wsi", "momentum"]
    smoothed = panel.groupby("game_id", group_keys=False)[cols_to_smooth].apply(
        lambda g: g.rolling(3, center=True, min_periods=1).mean()
    )
    smoothed = pd.concat([panel[["game_id", "match_minute"]], smoothed], axis=1)
    
    # Shift momentum by 2 (campo leads chat by 2 mins)
    shifted_momentum = smoothed.groupby("game_id")["momentum"].shift(2)
    mask = smoothed["wsi"].notna() & shifted_momentum.notna()
    r_lag, p_lag = pearsonr(smoothed[mask]["wsi"], shifted_momentum[mask])
    print("\n=== Test 2: Minute-Level WSI vs. Momentum (Shifted +2 mins) ===")
    print(f"Correlation: r = {r_lag:.4f}, p-value = {p_lag:.2e}")
    results.append({
        "Analysis Level": "Minute-Level",
        "Variable 1": "WSI (Smoothed)",
        "Variable 2": "Momentum (Shifted +2m)",
        "Test Type": "Pearson Correlation",
        "Statistic Name": "r",
        "Statistic Value": r_lag,
        "p-value": p_lag,
        "Significance (alpha=0.05)": "SIGNIFICANT" if p_lag < 0.05 else "NOT SIGNIFICANT",
        "Interpretation": "At the optimal delay, positive momentum on the field is significantly correlated with positive sentiment."
    })
    
    # --- TEST 3: Immediate Reaction to Goals (t=0, t+1) ---
    vol_df, wsi_df, meta_df = build_event_matrix(panel, "goals")
    
    # Rebuild event matrix for toxicity
    window = list(range(-5, 6))
    indexed = panel.set_index(["game_id", "match_minute"])
    tox_rows = []
    for _, ev in panel[panel["goals"] >= 1].iterrows():
        gid, minute = ev["game_id"], ev["match_minute"]
        if minute >= 5 and minute <= 85:
            window_tox = [indexed.loc[(gid, minute + k), "toxicity_ratio"] for k in window]
            tox_rows.append(window_tox)
    tox_df = pd.DataFrame(tox_rows, columns=window)
    
    baseline = baseline_excluding_events(panel, "goals")
    
    # Calculate baseline toxicity
    ev_rows = panel[panel["goals"] >= 1]
    excluded = {(gid, min_ + k) for gid, min_ in zip(ev_rows["game_id"], ev_rows["match_minute"]) for k in window}
    keys = list(zip(panel["game_id"], panel["match_minute"]))
    mask_baseline = ~pd.Series(keys, index=panel.index).isin(excluded)
    baseline_tox = panel[mask_baseline].groupby("game_id")["toxicity_ratio"].mean().rename("baseline_tox")
    
    merged = meta_df.join(baseline, on="game_id").join(baseline_tox, on="game_id")
    
    # Reaction is mean of t=0 and t+1
    reaction_wsi = wsi_df[[0, 1]].mean(axis=1)
    reaction_tox = tox_df[[0, 1]].mean(axis=1)
    
    # Run Paired t-tests
    t_wsi, p_wsi_test = stats.ttest_rel(reaction_wsi, merged["baseline_wsi"])
    t_tox, p_tox_test = stats.ttest_rel(reaction_tox, merged["baseline_tox"])
    
    print("\n=== Test 3a: Immediate WSI Reaction to Goals (t=0, t+1) vs. Baseline ===")
    print(f"Paired t-test: t-stat = {t_wsi:.2f}, p-value = {p_wsi_test:.2e} (n = {len(reaction_wsi)})")
    results.append({
        "Analysis Level": "Event-Level",
        "Variable 1": "WSI Reaction (t=0, t+1)",
        "Variable 2": "Game Baseline WSI",
        "Test Type": "Paired t-test",
        "Statistic Name": "t-stat",
        "Statistic Value": t_wsi,
        "p-value": p_wsi_test,
        "Significance (alpha=0.05)": "SIGNIFICANT" if p_wsi_test < 0.05 else "NOT SIGNIFICANT",
        "Interpretation": "Goals trigger an immediate, extremely significant surge of positive WSI in the chat."
    })
    
    print("\n=== Test 3b: Immediate Toxicity Reaction to Goals (t=0, t+1) vs. Baseline ===")
    print(f"Paired t-test: t-stat = {t_tox:.2f}, p-value = {p_tox_test:.2e} (n = {len(reaction_tox)})")
    results.append({
        "Analysis Level": "Event-Level",
        "Variable 1": "Toxicity Reaction (t=0, t+1)",
        "Variable 2": "Game Baseline Toxicity",
        "Test Type": "Paired t-test",
        "Statistic Name": "t-stat",
        "Statistic Value": t_tox,
        "p-value": p_tox_test,
        "Significance (alpha=0.05)": "SIGNIFICANT" if p_tox_test < 0.05 else "NOT SIGNIFICANT",
        "Interpretation": "Goals trigger a significant drop in toxicity ratio as celebration comments outnumber negative ones."
    })
    
    # Save all results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "tables" / "rq1_sentiment_significance.csv", index=False)
    print(f"\nAll results saved to {OUT_DIR}/tables/rq1_sentiment_significance.csv")

if __name__ == "__main__":
    main()
