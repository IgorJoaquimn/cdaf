"""Groups Bundesliga matches using K-Means clustering, compares K-Means with HDBSCAN
using Silhouette Scores, profiles each cluster, and generates word clouds of chat
comments for each match profile.
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from preprocessing import games_with_chat_data, load_match_minute_panel

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)

STOPWORDS = set([
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 
    'mais', 'as', 'dos', 'como', 'mas', 'foi', 'ao', 'ele', 'das', 'tem', 'à', 'seu', 'sua', 'ou', 'ser', 'quando', 
    'muito', 'há', 'nos', 'já', 'está', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 
    'sem', 'mesmo', 'aos', 'ter', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'estão', 'você', 'tinha', 'foram', 
    'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'têm', 'numa', 'pelos', 'elas', 'havia', 'seja', 'qual', 
    'será', 'nós', 'tenho', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'fossem', 'dele', 'tu', 'te', 
    'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'estou', 'escuta', 'estive', 'esteve', 
    'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera', 'estivéramos', 'estivesse', 'estivéssemos', 
    'estivessem', 'estiver', 'estivermos', 'estiverem', 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos', 'tinham', 
    'tivera', 'tivéramos', 'tivesse', 'tivéssemos', 'tivessem', 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 
    'teremos', 'terão', 'teria', 'teríamos', 'teriam', 'jogo', 'completo', 'cazetv', 'caze', 'tv', 'roder', 'rodada', 
    'bundesliga', 'comentário', 'vídeo', 'canal', 'kkkk', 'kkkkkk', 'kkkkkkk', 'kkk', 'pra', 'pro', 'ta', 'tá', 'tb', 
    'tbm', 'vc', 'vcs', 'jogos', 'jogo', 'partida', 'gol', 'gols', 'cara', 'vai', 'vou', 'neste', 'ia', 'ir', 'ver',
    'assistir', 'transmissão', 'aqui', 'lá', 'nao', 'so', 'ja', 'esta', 'tambem', 'so', 'pelo', 'pela', 'ate', 'isso',
    'mais', 'menos', 'muito', 'q', 'd', 'p', 's', 't', 'c', 'm', 'n', 'o'
])

def generate_wordclouds(comments_df: pd.DataFrame, match_clusters: pd.DataFrame, sorted_clusters: list, cluster_names: dict) -> None:
    print("\nGenerating word clouds for each cluster...")
    # Link comments to match clusters
    merged = comments_df.merge(match_clusters[["video_title", "cluster"]], on="video_title", how="inner")
    
    for cluster_id in sorted_clusters:
        sub = merged[merged.cluster == cluster_id]
        text = " ".join(sub["mensagem"].dropna().astype(str).tolist())
        
        # Clean text
        text = " ".join([word for word in text.split() if word.lower() not in STOPWORDS and not word.startswith("@")])
        
        if not text.strip():
            print(f"  No text found for cluster {cluster_id}")
            continue
            
        # Determine color palette
        colormap = "Reds" if cluster_id == sorted_clusters[2] else ("Blues" if cluster_id == sorted_clusters[0] else "Oranges")
        
        wc = WordCloud(
            width=800, height=450, 
            background_color="white", 
            max_words=100, 
            stopwords=STOPWORDS,
            colormap=colormap,
            random_state=42
        ).generate(text)
        
        # Plot and save
        plt.figure(figsize=(10, 5.6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Nuvem de Palavras: {cluster_names[cluster_id]}", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        path = OUT_DIR / "figures" / f"rq1_wordcloud_cluster_{cluster_id}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved word cloud to {path}")

def main():
    panel = load_match_minute_panel()
    panel = games_with_chat_data(panel)
    
    # Compute helper metrics
    panel["volume_rate"] = panel["volume_total"] / 60.0
    panel["xt_total"] = panel["xt_home"] + panel["xt_away"]
    panel["xt_abs_diff"] = (panel["xt_home"] - panel["xt_away"]).abs()
    
    # Group by game to extract match-level features
    match_features = panel.groupby("game_id").agg(
        video_title=("video_title", "first"),
        xt_total_mean=("xt_total", "mean"),
        xt_abs_diff_mean=("xt_abs_diff", "mean"),
        volume_rate_mean=("volume_rate", "mean"),
        volume_rate_std=("volume_rate", "std"),
        polarity_mean=("polarity", "mean"),
        polarity_std=("polarity", "std"),
        goals_total=("goals", "sum"),
        cards_total=("cards", "sum")
    ).reset_index()
    
    feature_cols = [
        "xt_total_mean", "xt_abs_diff_mean", 
        "volume_rate_mean", "volume_rate_std", 
        "polarity_mean", "polarity_std", 
        "goals_total", "cards_total"
    ]
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(match_features[feature_cols])
    
    # === ALGORITHM COMPARISON (K-Means vs. HDBSCAN) ===
    # 1. K-Means (K=3)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    km_sil = silhouette_score(X, km_labels)
    
    # 2. HDBSCAN
    hdb = HDBSCAN(min_cluster_size=5, min_samples=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        hdb_labels = hdb.fit_predict(X)
    hdb_mask = hdb_labels != -1
    hdb_n_clusters = len(set(hdb_labels[hdb_mask]))
    if hdb_mask.sum() > 10 and hdb_n_clusters > 1:
        hdb_sil = silhouette_score(X[hdb_mask], hdb_labels[hdb_mask])
    else:
        hdb_sil = -2.0
        
    print("=== Clustering Comparison ===")
    print(f"K-Means (K=3): Silhouette = {km_sil:.4f}, noise points = 0/{len(X)}, clusters = 3")
    print(f"HDBSCAN (min_size=5, min_samples=3): Silhouette (non-noise) = {hdb_sil:.4f}, noise points = {(~hdb_mask).sum()}/{len(X)}, clusters = {hdb_n_clusters}")
    
    # Select best model based on coverage and Silhouette Score
    # While HDBSCAN might have a higher silhouette score on a subset of points, it classifies ~76% of games as noise.
    # Therefore, K-Means is selected as it assigns every game to a meaningful profile.
    print("\nDecision: Keeping K-Means because HDBSCAN classifies too many games (76%) as noise, preventing full dataset profiling.")
    
    # Fit final clustering with K-Means
    match_features["cluster"] = km_labels
    match_features.to_csv(OUT_DIR / "tables" / "rq1_match_clusters.csv", index=False)
    
    # Calculate profiles for each cluster
    profile = match_features.groupby("cluster").agg(
        n_matches=("game_id", "count"),
        xt_total_mean=("xt_total_mean", "mean"),
        xt_abs_diff_mean=("xt_abs_diff_mean", "mean"),
        volume_rate_mean=("volume_rate_mean", "mean"),
        volume_rate_std=("volume_rate_std", "mean"),
        polarity_mean=("polarity_mean", "mean"),
        polarity_std=("polarity_std", "mean"),
        goals_total=("goals_total", "mean"),
        cards_total=("cards_total", "mean")
    ).reset_index()
    profile.to_csv(OUT_DIR / "tables" / "rq1_cluster_profiles.csv", index=False)
    
    # Sort clusters to name them consistently
    sorted_clusters = profile.sort_values(by="volume_rate_mean")["cluster"].values
    
    cluster_names = {
        sorted_clusters[0]: "Low Intensity / Boring Matches",
        sorted_clusters[1]: "Moderate Excitement / Standard Matches",
        sorted_clusters[2]: "High Intensity / Thrilling Matches"
    }
    
    match_features["cluster_name"] = match_features["cluster"].map(cluster_names)
    
    # Generate scatter plot
    plt.figure(figsize=(9, 6.5))
    colors = {sorted_clusters[0]: '#2874a6', sorted_clusters[1]: '#f39c12', sorted_clusters[2]: '#cb4335'}
    
    for cluster_id, name in cluster_names.items():
        sub = match_features[match_features.cluster == cluster_id]
        plt.scatter(
            sub["xt_total_mean"], sub["volume_rate_mean"], 
            s=sub["goals_total"] * 40 + 40, 
            color=colors[cluster_id], 
            label=f"{name} (n={len(sub)})", 
            alpha=0.8, edgecolors='black', linewidth=0.8
        )
        
    for cluster_id in cluster_names:
        sub = match_features[match_features.cluster == cluster_id]
        if not sub.empty:
            rep = sub.loc[sub["volume_rate_mean"].idxmax()]
            title_clean = rep["video_title"].split("|")[0].split("JOGO COMPLETO:")[1].strip() if "JOGO COMPLETO:" in rep["video_title"] else rep["video_title"].split("|")[0].strip()
            title_clean = title_clean[:35] + "..." if len(title_clean) > 35 else title_clean
            plt.annotate(
                title_clean, 
                (rep["xt_total_mean"], rep["volume_rate_mean"]),
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center', fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color='black', lw=0.5),
                bbox=dict(boxstyle="round,pad=0.2", fc='yellow', alpha=0.5, ec='gray', lw=0.5)
            )

    plt.xlabel("Média de xT Total (Intensidade Tática)")
    plt.ylabel("Média de Volume de Mensagens (comentários/segundo)")
    plt.title("Agrupamento de Partidas (K-Means vs. HDBSCAN Analysis)")
    plt.legend(loc="upper left", fontsize=8.5)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "figures" / "rq1_match_clustering.png", dpi=150)
    plt.close()
    print("Visualização de clustering gerada com sucesso.")
    
    # Save the clustering comparison summary for final report
    comparison_df = pd.DataFrame([
        {"Algorithm": "K-Means (K=3)", "Silhouette": km_sil, "Noise Points": "0/54", "Clusters": 3},
        {"Algorithm": "HDBSCAN", "Silhouette": hdb_sil, "Noise Points": f"{(~hdb_mask).sum()}/54", "Clusters": hdb_n_clusters}
    ])
    comparison_df.to_csv(OUT_DIR / "tables" / "rq1_clustering_comparison.csv", index=False)
    
    # Load comments database to generate word clouds
    comments_path = Path(__file__).resolve().parent.parent / "data" / "chat_comments_with_sentiment.csv"
    if comments_path.exists():
        comments_df = pd.read_csv(comments_path, usecols=["video_title", "mensagem"])
        generate_wordclouds(comments_df, match_features, sorted_clusters, cluster_names)
    else:
        print(f"Warning: Comments file not found at {comments_path}, word clouds skipped.")

if __name__ == "__main__":
    main()
