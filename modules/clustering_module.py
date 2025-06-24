#CLUSTERING_MODULE.PY

# clustering_module.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def preprocess_for_clustering(df):
    df_model = df.select_dtypes(include=['int64', 'float64']).copy()
    df_model = df_model.dropna()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_model)
    return scaled, df_model.columns.tolist()

def run_kmeans(df, k):
    scaled, feature_names = preprocess_for_clustering(df)
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled)
    joblib.dump(kmeans, "models/cluster_model.pkl")
    return scaled, kmeans, df

def elbow_method(df):
    scaled, _ = preprocess_for_clustering(df)
    inertia = []
    for k in range(2, 21):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(scaled)
        inertia.append(km.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 21), inertia, marker='o', linestyle='--', color='teal')
    plt.title("Elbow Method")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("assets/visuals/elbow.png")
    plt.close()

def find_optimal_k(df):
    scaled, _ = preprocess_for_clustering(df)
    best_k = 2
    best_score = -1
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k