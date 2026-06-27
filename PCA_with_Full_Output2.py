import pandas as pd
import numpy as np
import os

print("=== Integrated PCA Program ===")

# -------------------------------
# 1. User Inputs
# -------------------------------
file_name = input("Enter CSV file name (e.g., data.csv): ").strip()
col_range = input("Enter DATA column range (e.g., E:S): ").upper().strip()
name_col_letter = input("Enter ROW NAMES column (e.g., A): ").upper().strip()
start_row = int(input("Enter starting row of data (e.g., 5): "))

def col_to_index(col):
    index = 0
    for c in col:
        index = index * 26 + (ord(c) - ord('A') + 1)
    return index - 1

# Convert Excel letters to indices
start_col, end_col = col_range.split(":")
data_start_idx = col_to_index(start_col)
data_end_idx = col_to_index(end_col) + 1
name_idx = col_to_index(name_col_letter)

# -------------------------------
# 2. Data Loading & Cleaning
# -------------------------------
df = pd.read_csv(file_name, skiprows=start_row - 1)

df.columns = df.columns.astype(str).str.strip()

names_series = df.iloc[:, name_idx]
X_df = df.iloc[:, data_start_idx:data_end_idx]

combined = pd.concat([names_series, X_df], axis=1)
combined.iloc[:, 1:] = combined.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
combined = combined.dropna()

clean_names = combined.iloc[:, 0].reset_index(drop=True)
X_clean_df = combined.iloc[:, 1:]

X = X_clean_df.values
feature_names = X_clean_df.columns

# -------------------------------
# 3. PCA Calculations
# -------------------------------
X_centered = X - np.mean(X, axis=0)

cov_matrix = np.cov(X_centered, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

PC_scores = X_centered.dot(eigenvectors)

explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance_ratio)

# -------------------------------
# 4. Select PCs (90% variance)
# -------------------------------
threshold = 0.90
num_pcs = np.argmax(cumulative_variance >= threshold) + 1

# Reduce everything
eigenvalues = eigenvalues[:num_pcs]
eigenvectors = eigenvectors[:, :num_pcs]
PC_scores = PC_scores[:, :num_pcs]
explained_variance_ratio = explained_variance_ratio[:num_pcs]
cumulative_variance = cumulative_variance[:num_pcs]

# -------------------------------
# 5. Formatting Outputs
# -------------------------------
pc_cols = [f"PC{i+1}" for i in range(num_pcs)]

pc_df = pd.DataFrame(PC_scores, columns=pc_cols)
pc_df.insert(0, "Row_Label", clean_names)

stats_df = pd.DataFrame({
    "Principal_Component": pc_cols,
    "Eigenvalue": eigenvalues,
    "Variance_Ratio": explained_variance_ratio,
    "Cumulative_Variance": cumulative_variance
})

eigvec_df = pd.DataFrame(eigenvectors, index=feature_names, columns=pc_cols)

# -------------------------------
# 6. Save Outputs
# -------------------------------
os.makedirs("PCA_Output", exist_ok=True)

stats_file = "PCA_Output/PCA_EIGENVALUES.csv"
eigvec_file = "PCA_Output/PCA_EIGENVECTORS.csv"
scores_file = "PCA_Output/PCA_SCORES.csv"

stats_df.to_csv(stats_file, index=False)
eigvec_df.to_csv(eigvec_file)
pc_df.to_csv(scores_file, index=False)

# -------------------------------
# 7. Summary
# -------------------------------
print("\n" + "="*40)
print("✅ PCA ANALYSIS COMPLETE")
print(f"✔ PCs retained (90% variance): {num_pcs}")

print("\n📁 Files Generated:")
print(f"1. Eigenvalues → {stats_file}")
print(f"2. Eigenvectors → {eigvec_file}")
print(f"3. Scores → {scores_file}")

print("="*40)

print("\n🔍 Column Names Used:")
print(feature_names.tolist())
