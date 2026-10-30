"""
04_KMEANS_CLUSTERING.PY
=======================
Tahap 4 Pipeline: Pemodelan K-Means Clustering (k=4),
Penetapan Label Profil Wilayah, dan Pembuatan Visualisasi Diagnostik Lengkap.

Input:
  - dataset/processed/fitur_kmeans_smp_standardisasi.csv
  - dataset/processed/jabar_smp_merged.csv
Output:
  - dataset/processed/hasil_klaster_kmeans_smp.csv
  - outputs/tables/profil_klaster_smp.csv
  - outputs/figures/visualisasi_klaster_kmeans.png
  - outputs/figures/klaster_scatter_tka_vs_kemiskinan.png
  - outputs/figures/klaster_scatter_guru_vs_sekolah.png
  - outputs/figures/klaster_fingerprint_zscore.png
  - outputs/figures/klaster_distribusi_wilayah.png
"""

from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 1. Konfigurasi Path & Parameter
K_UTAMA = 4
K_PEMBANDING = 3
RANDOM_STATE = 42
N_INIT = 10

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_STD_PATH = BASE_DIR / "dataset" / "processed" / "fitur_kmeans_smp_standardisasi.csv"
DATA_RAW_PATH = BASE_DIR / "dataset" / "processed" / "jabar_smp_merged.csv"
OUTPUT_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUT_FIGURES = BASE_DIR / "outputs" / "figures"
DATA_PROCESSED = BASE_DIR / "dataset" / "processed"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[04] PEMODELAN K-MEANS CLUSTERING (k=4) & PROFILING WILAYAH")
print("=" * 60)

# 2. Muat Data
df_std = pd.read_csv(DATA_STD_PATH)
df_raw = pd.read_csv(DATA_RAW_PATH)

# Perbaiki kode_wilayah jika perlu
if "kode_bps" in df_raw.columns:
    kb = df_raw["kode_bps"].astype(str).str.replace(r"\.0$", "", regex=True)
    df_raw["kode_wilayah"] = kb.str[:2] + "." + kb.str[2:]

fitur = [
    "tka_rerata_mtk",
    "rasio_murid_guru",
    "rasio_murid_sekolah",
    "pct_penduduk_miskin"
]
X = df_std[fitur].values

# 3. Fitting K-Means (k=4 utama & k=3 pembanding)
kmeans_utama = KMeans(n_clusters=K_UTAMA, random_state=RANDOM_STATE, n_init=N_INIT)
labels_utama = kmeans_utama.fit_predict(X)

kmeans_pemb = KMeans(n_clusters=K_PEMBANDING, random_state=RANDOM_STATE, n_init=N_INIT)
labels_pemb = kmeans_pemb.fit_predict(X)

sil_k4 = silhouette_score(X, labels_utama)
dbi_k4 = davies_bouldin_score(X, labels_utama)
print(f"Metrik Kualitas Klaster (k=4): Silhouette = {sil_k4:.4f}, Davies-Bouldin = {dbi_k4:.4f}")

# 4. Pemetaan Klaster Terurut Berdasarkan Rata-rata TKA (1 = Tertinggi s/d 4 = Terendah)
temp_df = pd.DataFrame({"raw_cluster": labels_utama, "tka": df_raw["tka_rerata_mtk"]})
urutan = (
    temp_df.groupby("raw_cluster")["tka"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
mapping_klaster = {cluster_lama: cluster_baru for cluster_baru, cluster_lama in enumerate(urutan, start=1)}
labels_k4_terurut = np.array([mapping_klaster[c] for c in labels_utama])

# Nama profil deskriptif
NAMA_KLASTER_K4 = {
    1: "Klaster 1: Perkotaan, Capaian Tinggi",
    2: "Klaster 2: Beban Kelas Tinggi, Capaian Menengah",
    3: "Klaster 3: Kemiskinan Tinggi, Capaian Menengah",
    4: "Klaster 4: Capaian Rendah, Skala Sekolah Terbatas"
}

# Gabungkan ke dataframe raw
df_hasil = df_raw.copy()
df_hasil["klaster_k4"] = labels_k4_terurut
df_hasil["nama_klaster_k4"] = df_hasil["klaster_k4"].map(NAMA_KLASTER_K4)
df_hasil["klaster_k3"] = labels_pemb + 1

# Simpan dataset hasil klasterisasi
hasil_csv_path = DATA_PROCESSED / "hasil_klaster_kmeans_smp.csv"
df_hasil.to_csv(hasil_csv_path, index=False)
print(f"[OK] Dataset hasil klasterisasi disimpan ke:\n     -> {hasil_csv_path}")

# 5. Tabel Profil Rata-Rata Asli Tiap Klaster
profil_asli = df_hasil.groupby(["klaster_k4", "nama_klaster_k4"])[fitur + ["tka_rerata_bind"]].mean().round(2)
profil_asli["jumlah_wilayah"] = df_hasil.groupby("klaster_k4").size()
profil_csv_path = OUTPUT_TABLES / "profil_klaster_smp.csv"
profil_asli.to_csv(profil_csv_path)
print(f"[OK] Profil karakteristik klaster disimpan ke:\n     -> {profil_csv_path}")

print("\nProfil Rerata Karakteristik Klaster (Skala Asli):")
print(profil_asli.to_string())

# 6. Visualisasi Diagnostik Klaster
warna_map = {
    1: "#2563eb",  # Biru
    2: "#0ea5e9",  # Cyan
    3: "#f59e0b",  # Amber / Oranye
    4: "#dc2626"   # Merah
}
colors_k4 = [warna_map[c] for c in df_hasil["klaster_k4"]]

# Visualisasi 1: Scatter Plot TKA vs Kemiskinan
fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
for k_val, group in df_hasil.groupby("klaster_k4"):
    ax.scatter(group["pct_penduduk_miskin"], group["tka_rerata_mtk"],
               c=warna_map[k_val], label=NAMA_KLASTER_K4[k_val], s=90, edgecolors="black", alpha=0.85)
    for _, row in group.iterrows():
        ax.annotate(row["nama_pendek"], (row["pct_penduduk_miskin"], row["tka_rerata_mtk"]),
                    fontsize=7.5, xytext=(4, 4), textcoords="offset points")

ax.set_title("Distribusi Capaian TKA Matematika vs Kemiskinan per Wilayah", fontsize=11, fontweight="bold")
ax.set_xlabel("Persentase Penduduk Miskin (%)", fontsize=10)
ax.set_ylabel("Nilai Rerata TKA Matematika", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=8.5, loc="lower left")
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES / "klaster_scatter_tka_vs_kemiskinan.png")
plt.close(fig)

# Visualisasi 2: Scatter Plot Rasio Murid/Guru vs Murid/Sekolah
fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
for k_val, group in df_hasil.groupby("klaster_k4"):
    ax.scatter(group["rasio_murid_sekolah"], group["rasio_murid_guru"],
               c=warna_map[k_val], label=NAMA_KLASTER_K4[k_val], s=90, edgecolors="black", alpha=0.85)
    for _, row in group.iterrows():
        ax.annotate(row["nama_pendek"], (row["rasio_murid_sekolah"], row["rasio_murid_guru"]),
                    fontsize=7.5, xytext=(4, 4), textcoords="offset points")

ax.set_title("Beban Guru (Rasio Murid/Guru) vs Skala Sekolah (Rasio Murid/Sekolah)", fontsize=11, fontweight="bold")
ax.set_xlabel("Rasio Murid per Sekolah (Skala Sarana Fisik)", fontsize=10)
ax.set_ylabel("Rasio Murid per Guru (Beban Pengajaran)", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(fontsize=8.5)
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES / "klaster_scatter_guru_vs_sekolah.png")
plt.close(fig)

# Visualisasi 3: Fingerprint Z-Score Tiap Klaster
df_std_labeled = df_std.copy()
df_std_labeled["klaster_k4"] = labels_k4_terurut
z_means = df_std_labeled.groupby("klaster_k4")[fitur].mean()

fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
x_pos = np.arange(len(fitur))
lebar = 0.18
labels_fitur = ["TKA MTK", "Rasio M/G", "Rasio M/S", "Kemiskinan (%)"]

for i, k_val in enumerate(range(1, 5)):
    ax.bar(x_pos + (i - 1.5) * lebar, z_means.loc[k_val], lebar,
           label=NAMA_KLASTER_K4[k_val], color=warna_map[k_val], edgecolor="black", alpha=0.85)

ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_fitur, fontsize=10, fontweight="bold")
ax.set_ylabel("Z-Score Rerata (Penyimpangan dari Standar Jabar)", fontsize=10)
ax.set_title("Fingerprint Karakteristik 4 Pilar Pendidikan per Klaster (Z-Score)", fontsize=11, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES / "klaster_fingerprint_zscore.png")
plt.close(fig)

# Visualisasi 4: Distribusi Jumlah Wilayah
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
dist_counts = df_hasil["klaster_k4"].value_counts().sort_index()
bars = ax.bar([f"Klaster {k}" for k in dist_counts.index], dist_counts.values,
              color=[warna_map[k] for k in dist_counts.index], edgecolor="black", alpha=0.85)
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{int(yval)} Kab/Kota",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_title("Distribusi Jumlah Kabupaten/Kota per Klaster", fontsize=11, fontweight="bold")
ax.set_ylabel("Jumlah Wilayah", fontsize=10)
ax.set_ylim(0, max(dist_counts.values) + 2)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES / "klaster_distribusi_wilayah.png")
plt.close(fig)

# Visualisasi 5: 4-Panel Komposit Visualisasi Klaster
fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=200)

# Panel A
for k_val, group in df_hasil.groupby("klaster_k4"):
    axes[0, 0].scatter(group["pct_penduduk_miskin"], group["tka_rerata_mtk"],
                      c=warna_map[k_val], s=70, edgecolors="black", label=f"K{k_val}")
axes[0, 0].set_title("A. TKA vs Kemiskinan", fontsize=10, fontweight="bold")
axes[0, 0].set_xlabel("Persentase Miskin (%)", fontsize=9)
axes[0, 0].set_ylabel("TKA MTK", fontsize=9)
axes[0, 0].grid(True, linestyle=":", alpha=0.6)
axes[0, 0].legend(fontsize=8)

# Panel B
for k_val, group in df_hasil.groupby("klaster_k4"):
    axes[0, 1].scatter(group["rasio_murid_sekolah"], group["rasio_murid_guru"],
                      c=warna_map[k_val], s=70, edgecolors="black", label=f"K{k_val}")
axes[0, 1].set_title("B. Beban Guru vs Skala Sekolah", fontsize=10, fontweight="bold")
axes[0, 1].set_xlabel("Rasio Murid/Sekolah", fontsize=9)
axes[0, 1].set_ylabel("Rasio Murid/Guru", fontsize=9)
axes[0, 1].grid(True, linestyle=":", alpha=0.6)
axes[0, 1].legend(fontsize=8)

# Panel C
for i, k_val in enumerate(range(1, 5)):
    axes[1, 0].bar(x_pos + (i - 1.5) * lebar, z_means.loc[k_val], lebar,
                  color=warna_map[k_val], edgecolor="black", label=f"K{k_val}")
axes[1, 0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(labels_fitur, fontsize=9)
axes[1, 0].set_title("C. Fingerprint Z-Score 4 Pilar", fontsize=10, fontweight="bold")
axes[1, 0].set_ylabel("Rerata Z-Score", fontsize=9)
axes[1, 0].grid(True, axis="y", linestyle=":", alpha=0.6)
axes[1, 0].legend(fontsize=8)

# Panel D
axes[1, 1].bar([f"K{k}" for k in dist_counts.index], dist_counts.values,
              color=[warna_map[k] for k in dist_counts.index], edgecolor="black")
axes[1, 1].set_title("D. Distribusi Wilayah", fontsize=10, fontweight="bold")
axes[1, 1].set_ylabel("Kab/Kota", fontsize=9)
axes[1, 1].grid(True, axis="y", linestyle=":", alpha=0.6)

plt.suptitle("Dashboard Diagnostik Klasterisasi K-Means Pendidikan SMP Jawa Barat (k=4)",
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES / "visualisasi_klaster_kmeans.png")
plt.close(fig)

print(f"[OK] Visualisasi diagnostik klaster tersimpan di folder:\n     -> {OUTPUT_FIGURES}")
print("Selesai Tahap 04.\n")
