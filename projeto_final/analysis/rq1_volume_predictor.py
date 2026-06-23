"""Builds and evaluates a predictive model for next-minute message volume rate.
Compares a baseline autoregressive model against an expanded model including on-field metrics.
Uses GroupKFold cross-validation grouped by game_id.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from preprocessing import games_with_chat_data, load_match_minute_panel

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = panel.copy()
    # Sort to ensure correct shifting
    df = df.sort_values(["game_id", "match_minute"]).reset_index(drop=True)
    
    # Message Volume Rate (comments per second)
    df["volume_rate"] = df["volume_total"] / 60.0
    
    # Total Expected Threat
    df["xt_total"] = df["xt_home"] + df["xt_away"]
    
    # Total Momentum (rolling 5 minutes sum of total threat)
    df["momentum_total"] = df.groupby("game_id", group_keys=False)["xt_total"].apply(
        lambda g: g.rolling(5, min_periods=1).sum()
    )
    
    # Momentum difference
    df["momentum_diff"] = df.groupby("game_id")["momentum"].diff()
    
    # Target: volume_rate at t+1
    df["target"] = df.groupby("game_id")["volume_rate"].shift(-1)
    
    # Lag features (current and previous volume rate)
    df["volume_lag0"] = df["volume_rate"]
    df["volume_lag1"] = df.groupby("game_id")["volume_rate"].shift(1)
    
    # Features from field
    df["xt_diff_t"] = df["xt_diff"]
    df["xt_home_t"] = df["xt_home"]
    df["xt_away_t"] = df["xt_away"]
    df["momentum_t"] = df["momentum"]
    df["momentum_diff_t"] = df["momentum_diff"]
    df["xt_total_t"] = df["xt_total"]
    df["momentum_total_t"] = df["momentum_total"]
    
    # Drop rows with NaNs in target or features
    baseline_cols = ["volume_lag0", "volume_lag1"]
    field_cols = [
        "xt_diff_t", "xt_home_t", "xt_away_t", 
        "momentum_t", "momentum_diff_t", 
        "xt_total_t", "momentum_total_t"
    ]
    all_cols = baseline_cols + field_cols
    
    clean_df = df.dropna(subset=["target"] + all_cols).reset_index(drop=True)
    return clean_df, baseline_cols, all_cols

def evaluate_model(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    gkf = GroupKFold(n_splits=5)
    X = df[feature_cols].values
    y = df["target"].values
    groups = df["game_id"].values
    
    maes, mses, r2s = [], [], []
    
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        
        # Predict and evaluate
        preds = model.predict(X_test_scaled)
        maes.append(mean_absolute_error(y_test, preds))
        mses.append(mean_squared_error(y_test, preds))
        r2s.append(r2_score(y_test, preds))
        
    return {
        "MAE": np.mean(maes),
        "MSE": np.mean(mses),
        "R2": np.mean(r2s)
    }

def train_and_get_coefs(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].values
    y = df["target"].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
    
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    
    return coef_df

def main():
    panel = load_match_minute_panel()
    panel = games_with_chat_data(panel)
    
    df_clean, baseline_cols, expanded_cols = build_features(panel)
    print(f"Número de observações limpas: {len(df_clean)}")
    
    # Evaluate Baseline Model
    baseline_metrics = evaluate_model(df_clean, baseline_cols)
    print("\n=== Baseline Autoregressive Model ===")
    print(baseline_metrics)
    
    # Evaluate Expanded Model
    expanded_metrics = evaluate_model(df_clean, expanded_cols)
    print("\n=== Expanded Model (Autoregressive + Field) ===")
    print(expanded_metrics)
    
    # Save metrics comparison
    metrics_df = pd.DataFrame([
        {"Model": "Baseline (Autoregressive)", **baseline_metrics},
        {"Model": "Expanded (Autoregressive + Field)", **expanded_metrics}
    ])
    metrics_df.to_csv(OUT_DIR / "tables" / "rq1_volume_predictor_metrics.csv", index=False)
    
    # Fit final model to extract feature importances
    coef_df = train_and_get_coefs(df_clean, expanded_cols)
    coef_df.to_csv(OUT_DIR / "tables" / "rq1_volume_predictor_coefficients.csv", index=False)
    print("\n=== Standardized Coefficients of the Expanded Model ===")
    print(coef_df)
    
    # Plot feature importances
    plt.figure(figsize=(8, 5))
    colors = ['#2e4053' if 'lag' in f else '#28b463' for f in coef_df["Feature"]]
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Coeficiente Padronizado (Importância)")
    plt.title("Importância das Features na Predição de Taxa de Volume (t+1)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "rq1_volume_predictor_feature_importance.png", dpi=150)
    plt.close()
    print("\nImportância das features plotada com sucesso.")

if __name__ == "__main__":
    main()
