# Iris Dataset: Advanced Clustering & Silhouette Optimization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load Iris data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

print("Number of features:", X.shape[1])
print("Number of rows:", X.shape[0])

# Preprocess - scale data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Store results
results = {}

# --- KMeans ---
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    results[f'KMeans_k={k}'] = score

# --- DBSCAN ---
for eps in np.arange(0.3, 1.0, 0.1):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    if len(set(labels)) > 1 and -1 not in set(labels):  # exclude noise only
        score = silhouette_score(X_scaled, labels)
        results[f'DBSCAN_eps={eps:.1f}'] = score

# --- Agglomerative Clustering ---
for k in range(2, 10):
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    results[f'Agglomerative_k={k}'] = score

# --- Gaussian Mixture Model ---
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    results[f'GMM_k={k}'] = score

# Display all silhouette scores
print("\nSilhouette Scores:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
for method, score in sorted_results:
    print(f"{method}: {score:.4f}")

# Best Result
best_method, best_score = sorted_results[0]
print(f"\n✅ Best Clustering Method: {best_method}")
print(f"✅ Best Silhouette Score: {best_score:.4f}")

# Visualization for best method
if 'KMeans' in best_method:
    k = int(best_method.split('=')[1])
    model = KMeans(n_clusters=k, random_state=42)
elif 'DBSCAN' in best_method:
    eps = float(best_method.split('=')[1])
    model = DBSCAN(eps=eps, min_samples=5)
elif 'Agglomerative' in best_method:
    k = int(best_method.split('=')[1])
    model = AgglomerativeClustering(n_clusters=k)
elif 'GMM' in best_method:
    k = int(best_method.split('=')[1])
    model = GaussianMixture(n_components=k, random_state=42)

labels = model.fit_predict(X_scaled)

# Plot PCA visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='Set1', s=60)
plt.title(f"PCA Plot for {best_method}")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

silhouette_scores = []  # To store silhouette scores

for k in range(2, 11):  # Silhouette score needs at least 2 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

# Plot Silhouette Scores for different values of k
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', color='g', linestyle='-', markersize=8)
plt.title('Silhouette Scores for Different K')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Best K from Elbow Method
elbow_best_k = np.argmin(np.diff(np.diff(inertia))) + 2  # Estimate the best k based on elbow
print(f"Optimal K based on Elbow Method: {elbow_best_k}")

# Best K from Silhouette Score
best_silhouette_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal K based on Silhouette Score: {best_silhouette_k}")