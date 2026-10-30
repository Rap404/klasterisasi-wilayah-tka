"""
03_EVALUASI_OPTIMAL_K.PY
========================
Tahap 3 Pipeline: Evaluasi dan Validasi Jumlah Klaster Optimal (k)
Menggunakan 4 Metrik Standar: Elbow (WCSS), Silhouette Score,
Davies-Bouldin Index (DBI), dan Calinski-Harabasz Score.

Input:
  - dataset/processed/fitur_kmeans_smp_standardisasi.csv
Output:
  - outputs/tables/evaluasi_optimal_k.csv
  - outputs/figures/evaluasi_optimal_k.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 1. Konfigurasi Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "processed" / "fitur_kmeans_smp_standardisasi.csv"
OUTPUT_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUT_FIGURES = BASE_DIR / "outputs" / "figures"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[03] EVALUASI PENENTUAN JUMLAH KLASTER OPTIMAL (k)")
print("=" * 60)

# 2. Muat Data
df = pd.read_csv(DATA_PATH)
fitur = [
    "tka_rerata_mtk",
    "rasio_murid_guru",
    "rasio_murid_sekolah",
    "pct_penduduk_miskin"
]
X = df[fitur].values

# 3. Evaluasi k = 2 s/d 8
k_range = range(2, 9)
hasil_evaluasi = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    inertia = kmeans.inertia_
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    
    hasil_evaluasi.append({
        "k": k,
        "inertia_wcss": round(inertia, 4),
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(dbi, 4),
        "calinski_harabasz": round(ch, 4)
    })

df_eval = pd.DataFrame(hasil_evaluasi)
eval_csv_path = OUTPUT_TABLES / "evaluasi_optimal_k.csv"
df_eval.to_csv(eval_csv_path, index=False)
print(f"[OK] Tabel metrik evaluasi k tersimpan di:\n     -> {eval_csv_path}")

print("\nRingkasan Metrik Pengujian k:")
print(df_eval.to_string(index=False))

# 4. Visualisasi Multi-Metrik Evaluasi
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=200)

# Panel 1: Elbow Method (WCSS)
axes[0].plot(df_eval["k"], df_eval["inertia_wcss"], marker="o", color="#2563eb", linewidth=2, markersize=7)
axes[0].axvline(x=4, color="#dc2626", linestyle="--", alpha=0.8, label="k = 4 (Optimal)")
axes[0].set_title("Elbow Method (Inertia / WCSS)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Jumlah Klaster (k)", fontsize=10)
axes[0].set_ylabel("Within-Cluster Sum of Squares", fontsize=10)
axes[0].grid(True, linestyle=":", alpha=0.6)
axes[0].legend(fontsize=9)

# Panel 2: Silhouette Score (Makin tinggi makin baik)
axes[1].plot(df_eval["k"], df_eval["silhouette_score"], marker="s", color="#16a34a", linewidth=2, markersize=7)
axes[1].axvline(x=4, color="#dc2626", linestyle="--", alpha=0.8, label="k = 4 (Optimal)")
axes[1].set_title("Silhouette Coefficient", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Jumlah Klaster (k)", fontsize=10)
axes[1].set_ylabel("Silhouette Score", fontsize=10)
axes[1].grid(True, linestyle=":", alpha=0.6)
axes[1].legend(fontsize=9)

# Panel 3: Davies-Bouldin Index (Makin rendah makin baik)
axes[2].plot(df_eval["k"], df_eval["davies_bouldin_index"], marker="^", color="#d97706", linewidth=2, markersize=7)
axes[2].axvline(x=4, color="#dc2626", linestyle="--", alpha=0.8, label="k = 4 (Optimal)")
axes[2].set_title("Davies-Bouldin Index (DBI)", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Jumlah Klaster (k)", fontsize=10)
axes[2].set_ylabel("Davies-Bouldin Index", fontsize=10)
axes[2].grid(True, linestyle=":", alpha=0.6)
axes[2].legend(fontsize=9)

plt.suptitle("Evaluasi Penentuan Jumlah Klaster Optimal K-Means (SMP Jawa Barat)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()

eval_img_path = OUTPUT_FIGURES / "evaluasi_optimal_k.png"
fig.savefig(eval_img_path, bbox_inches="tight")
plt.close(fig)

print(f"[OK] Grafik evaluasi k tersimpan di:\n     -> {eval_img_path}")
print("Selesai Tahap 03.\n")
