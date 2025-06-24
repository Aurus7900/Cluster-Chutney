# after_visuals.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap

# Save directory
VISUALS_DIR = "assets/visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

def plot_pca_clusters(df, cluster_col='Cluster'):
    numeric = df.select_dtypes(include=['int64', 'float64']).drop(columns=['User ID'], errors='ignore')
    if cluster_col not in df.columns or numeric.empty:
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)

    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=components[:, 0],
        y=components[:, 1],
        hue=df[cluster_col],
        palette=sns.color_palette("hls", df[cluster_col].nunique()),
        alpha=0.8
    )
    plt.title("PCA Plot of User Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    filepath = os.path.join(VISUALS_DIR, "pca_clusters.png")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def cluster_summary_table(df, cluster_col='Cluster'):
    df_encoded = pd.get_dummies(df, drop_first=False)
    if cluster_col not in df_encoded.columns:
        df_encoded[cluster_col] = df[cluster_col]
    summary = df_encoded.groupby(cluster_col).mean().round(2)
    return summary.reset_index()

def plot_cluster_centroids_heatmap(kmeans_model, feature_names, chunk_size=20):
    centroids = kmeans_model.cluster_centers_
    num_clusters, num_features = centroids.shape

    if len(feature_names) != num_features:
        raise ValueError("Mismatch between model features and provided feature names.")

    centroids_df = pd.DataFrame(centroids, columns=feature_names)

    custom_cmap = LinearSegmentedColormap.from_list("custom_div", ['#E23A46', '#F1FAEE'])

    n_chunks = math.ceil(num_features / chunk_size)

    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, num_features)
        chunk = centroids_df.iloc[:, start:end]

        plt.figure(figsize=(1.2 * chunk.shape[1], 6))
        sns.heatmap(
            chunk,
            annot=True,
            fmt=".2f",
            cmap=custom_cmap,
            center=0,
            linewidths=0.3,
            linecolor='gray',
            cbar_kws={"shrink": 0.7},
            annot_kws={"size": 8}
        )
        plt.title(f"KMeans Centroids (Features {start + 1}–{end})")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, f"centroid_chunk_{i+1}.png"))
        plt.close()
