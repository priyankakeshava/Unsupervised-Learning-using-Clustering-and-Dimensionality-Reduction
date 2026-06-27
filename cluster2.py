import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("=== Tool: Final Clustering Program ===")

# ================================
# 🔥 USER INPUT
# ================================
USER_K = 3        # 👈 number of clusters
USE_PCS = 3       # 👈 options: "ALL", 2, 3

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv(r"D:\Priyanka Keshava\Programming\Python\FullAnalaysis\PCA_Output\PCA_SCORES.csv")

print(f"📊 Rows: {len(df)}")

# ================================
# 2. SELECT PCA COMPONENTS
# ================================
if USE_PCS == "ALL":
    X = df.iloc[:, 1:].values
elif USE_PCS == 2:
    X = df.iloc[:, 1:3].values
elif USE_PCS == 3:
    X = df.iloc[:, 1:4].values
else:
    raise ValueError("USE_PCS must be 'ALL', 2, or 3")

print(f"📊 Using {X.shape[1]} PCs")
print(f"🎯 Clusters (K) = {USER_K}")

# ================================
# 3. APPLY K-MEANS
# ================================
kmeans = KMeans(n_clusters=USER_K, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

# ================================
# 4. METRICS
# ================================
inertia = kmeans.inertia_

if USER_K > 1:
    silhouette = silhouette_score(X, labels)
else:
    silhouette = None

print("-" * 50)
print(f"📊 Inertia: {inertia:,.2f}")

if silhouette is not None:
    print(f"📊 Silhouette Score: {silhouette:.4f}")
else:
    print("📊 Silhouette: Not defined for K=1")

print("-" * 50)

# ================================
# 5. ADD CLUSTERS TO DATA
# ================================
df['Cluster'] = labels

# ================================
# 6. SAVE OUTPUT
# ================================
output_path = r"D:\Priyanka Keshava\Programming\Python\FullAnalaysis\Final_Clusters.csv"
df.to_csv(output_path, index=False)

print(f"💾 Saved to: {output_path}")

# ================================
# 7. VISUALIZATION
# ================================

# CASE 1: 2 PCs OR ALL (only first 2 shown)
if X.shape[1] == 2 or USE_PCS == "ALL":
    plt.figure(figsize=(7, 5))

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=25)

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='X')

    plt.title(f"K-Means Clusters (PC1 vs PC2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    plt.show()


# CASE 2: 3 PCs → show all combinations
elif X.shape[1] == 3:
    plt.figure(figsize=(15, 4))

    centers = kmeans.cluster_centers_

    # 🔹 PC1 vs PC2
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=25)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='X')
    plt.title("PC1 vs PC2")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)

    # 🔹 PC1 vs PC3
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 2], c=labels, cmap='tab10', s=25)
    plt.scatter(centers[:, 0], centers[:, 2], c='black', s=100, marker='X')
    plt.title("PC1 vs PC3")
    plt.xlabel("PC1")
    plt.ylabel("PC3")
    plt.grid(True)

    # 🔹 PC2 vs PC3
    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='tab10', s=25)
    plt.scatter(centers[:, 1], centers[:, 2], c='black', s=100, marker='X')
    plt.title("PC2 vs PC3")
    plt.xlabel("PC2")
    plt.ylabel("PC3")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
