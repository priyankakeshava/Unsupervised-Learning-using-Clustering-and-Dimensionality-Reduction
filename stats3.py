import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("=== Tool: Final K Selection + PCA Comparison (Full Report + User K) ===")

# 🔥 USER INPUT
USER_K = int(input("Enter the K:"))  # 👈 change this anytime

# 1. Load data
df = pd.read_csv(r"D:\Priyanka Keshava\Programming\Python\FullAnalaysis\PCA_Output\PCA_SCORES.csv")

print(f"⏳ Analyzing {len(df)} rows...")

# ================================
# 🔥 FUNCTION
# ================================
def run_analysis(X, title):

    K_range_full = range(1, 11)
    K_range_sil = range(2, 11)

    inertia = []
    silhouette_scores = []

    # 1. Inertia
    for k in K_range_full:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia.append(km.inertia_)

    # 2. Silhouette
    for k in K_range_sil:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    # 3. Elbow detection
    def find_elbow_k_distance(k_values, inertia_values):
        k_array = np.array(k_values)
        i_array = np.array(inertia_values)

        point1 = np.array([k_array[0], i_array[0]])
        point2 = np.array([k_array[-1], i_array[-1]])

        distances = []
        for i in range(len(k_array)):
            point = np.array([k_array[i], i_array[i]])
            dist = np.abs(np.cross(point2 - point1, point1 - point)) / np.linalg.norm(point2 - point1)
            distances.append(dist)

        return k_array[np.argmax(distances)]

    elbow_k = find_elbow_k_distance(list(K_range_full), inertia)
    sil_k = K_range_sil[np.argmax(silhouette_scores)]

    # 4. Final auto decision
    if abs(elbow_k - sil_k) <= 1:
        final_k = sil_k
    else:
        final_k = max(elbow_k, sil_k)

    # ================================
    # 🔥 USER-DEFINED K
    # ================================
    km_user = KMeans(n_clusters=USER_K, random_state=42, n_init=10)
    labels_user = km_user.fit_predict(X)

    user_inertia = km_user.inertia_

    if USER_K > 1:
        user_silhouette = silhouette_score(X, labels_user)
    else:
        user_silhouette = None

    # ================================
    # 🔥 FULL REPORT
    # ================================
    print("\n" + "="*60)
    print(f"📊 FULL REPORT: {title}")
    print("="*60)

    print("\n📊 INERTIA (K=1 → 10):")
    for k, val in zip(K_range_full, inertia):
        print(f"K={k} | Inertia={val:,.2f}")

    print("\n📊 SILHOUETTE (K=2 → 10):")
    for k, val in zip(K_range_sil, silhouette_scores):
        print(f"K={k} | Silhouette={val:.4f}")

    print("\n🎯 AUTO SELECTION:")
    print(f"Elbow K: {elbow_k}")
    print(f"Silhouette K: {sil_k}")
    print(f"FINAL K: {final_k}")

    print("\n👤 USER-DEFINED K:")
    print(f"K = {USER_K}")
    print(f"Inertia = {user_inertia:,.2f}")

    if user_silhouette is not None:
        print(f"Silhouette = {user_silhouette:.4f}")
    else:
        print("Silhouette = Not defined for K=1")

    print("="*60)

    # ================================
    # 📊 PLOTS
    # ================================
    plt.figure(figsize=(14, 4))

    # Elbow
    plt.subplot(1, 3, 1)
    plt.plot(K_range_full, inertia, 'bo-')
    plt.axvline(final_k, linestyle='--', label='Auto K')
    plt.axvline(USER_K, linestyle=':', label='User K')
    plt.title(f"Elbow ({title})")
    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid(True)

    # Silhouette
    plt.subplot(1, 3, 2)
    plt.plot(K_range_sil, silhouette_scores, 'go-')
    plt.axvline(final_k, linestyle='--', label='Auto K')
    plt.axvline(USER_K, linestyle=':', label='User K')
    plt.title(f"Silhouette ({title})")
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    # Combined
    inertia_norm = (inertia - np.min(inertia)) / (np.max(inertia) - np.min(inertia))
    silhouette_norm = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores))

    plt.subplot(1, 3, 3)
    plt.plot(K_range_full, 1 - inertia_norm, marker='o', label='Inertia')
    plt.plot(K_range_sil, silhouette_norm, marker='o', label='Silhouette')
    plt.axvline(final_k, linestyle='--', label='Auto K')
    plt.axvline(USER_K, linestyle=':', label='User K')
    plt.title(f"Combined ({title})")
    plt.xlabel("K")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return final_k, max(silhouette_scores)


# ================================
# 🔥 RUN ANALYSIS
# ================================

# ALL PCs
X_all = df.iloc[:, 1:].values
k_all, sil_all = run_analysis(X_all, "ALL PCs")

# FIRST 2 PCs
X_2 = df.iloc[:, 1:3].values
k_2, sil_2 = run_analysis(X_2, "First 2 PCs")

# FIRST 3 PCs
X_3 = df.iloc[:, 1:4].values
k_3, sil_3 = run_analysis(X_3, "First 3 PCs")

# FIRST 4 PCs
X_4 = df.iloc[:, 1:5].values
k_4, sil_4 = run_analysis(X_4, "First 4 PCs")

# ================================
# 🏁 FINAL COMPARISON
# ================================
print("\n" + "="*60)
print("🏁 FINAL COMPARISON")
print("="*60)

print(f"ALL PCs  → K={k_all}, Silhouette={sil_all:.4f}")
print(f"2 PCs    → K={k_2}, Silhouette={sil_2:.4f}")
print(f"3 PCs    → K={k_3}, Silhouette={sil_3:.4f}")
print(f"4 PCs    → K={k_4}, Silhouette={sil_4:.4f}")
# Best configuration
best = max(
    [("ALL PCs", k_all, sil_all),
     ("2 PCs", k_2, sil_2),
     ("3 PCs", k_3, sil_3),
     ("4 PCs", k_4, sil_4)],
    key=lambda x: x[2]
)

print("\n🎯 BEST CONFIGURATION:")
print(f"{best[0]} → K={best[1]} (Silhouette={best[2]:.4f})")
