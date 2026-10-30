"""
01_EDA_DAN_KORELASI.PY
======================
Tahap 1 Pipeline: Analisis Eksploratif Data (EDA), Statistik Deskriptif,
dan Analisis Multikolinearitas (Korelasi Pearson).

Input:
  - dataset/processed/jabar_smp_merged.csv
Output:
  - outputs/tables/statistik_deskriptif.csv
  - outputs/tables/matriks_korelasi.csv
  - outputs/tables/pasangan_korelasi_tinggi.csv
  - outputs/figures/matriks_korelasi.png
  - outputs/figures/heatmap_korelasi.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Konfigurasi Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "processed" / "jabar_smp_merged.csv"
OUTPUT_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUT_FIGURES = BASE_DIR / "outputs" / "figures"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[01] ANALISIS EKSPLORATIF (EDA) & KORELASI PEARSON")
print("=" * 60)

# 2. Muat Data
df = pd.read_csv(DATA_PATH)
print(f"Data berhasil dimuat: {df.shape[0]} wilayah, {df.shape[1]} kolom.")

# 3. Statistik Deskriptif
variabel_deskriptif = [
    "tka_rerata_bind",
    "tka_rerata_mtk",
    "rasio_murid_guru",
    "rasio_murid_sekolah",
    "pct_penduduk_miskin",
    "garis_kemiskinan_rp",
    "pct_tkp_s1_ke_atas",
]

statistik = df[variabel_deskriptif].describe().T
statistik["median"] = df[variabel_deskriptif].median()
statistik = statistik[["count", "mean", "median", "std", "min", "max"]].round(2)

stat_csv_path = OUTPUT_TABLES / "statistik_deskriptif.csv"
statistik.to_csv(stat_csv_path)
print(f"[OK] Statistik deskriptif disimpan ke:\n     -> {stat_csv_path}")

# 4. Matriks Korelasi Multivariat
fitur_korelasi = [
    "sekolah_total",
    "guru_total",
    "murid_total",
    "rasio_murid_guru",
    "rasio_murid_sekolah",
    "tka_rerata_bind",
    "tka_rerata_mtk",
    "garis_kemiskinan_rp",
    "penduduk_miskin_ribu",
    "pct_penduduk_miskin",
    "tkp_total",
    "tkp_s1_ke_atas",
    "pct_tkp_s1_ke_atas",
    "guru_per_tkp",
]

corr = df[fitur_korelasi].corr()
corr_csv_path = OUTPUT_TABLES / "matriks_korelasi.csv"
corr.round(4).to_csv(corr_csv_path)
print(f"[OK] Matriks korelasi disimpan ke:\n     -> {corr_csv_path}")

# 5. Identifikasi Multikolinearitas Tinggi (|r| >= 0.85)
pasangan_tinggi = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) >= 0.85:
            pasangan_tinggi.append({
                "Variabel_1": corr.columns[i],
                "Variabel_2": corr.columns[j],
                "Korelasi_r": round(r, 4),
                "Kategori": "Sangat Kuat (Multikolinearitas)"
            })

df_pasangan = pd.DataFrame(pasangan_tinggi).sort_values("Korelasi_r", ascending=False)
pasangan_csv_path = OUTPUT_TABLES / "pasangan_korelasi_tinggi.csv"
df_pasangan.to_csv(pasangan_csv_path, index=False)
print(f"[OK] Pasangan korelasi tinggi (|r| >= 0.85) disimpan ke:\n     -> {pasangan_csv_path}")

# 6. Visualisasi Heatmap Korelasi
fig, ax = plt.subplots(figsize=(11, 9), dpi=200)
cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        nilai = corr.iloc[i, j]
        teks_warna = "white" if abs(nilai) > 0.55 else "black"
        ax.text(j, i, f"{nilai:.2f}", ha="center", va="center", color=teks_warna, fontsize=7.5)

plt.colorbar(cax, fraction=0.046, pad=0.04, label="Koefisien Korelasi Pearson (r)")
plt.title("Matriks Korelasi Antar-Variabel Pendidikan SMP Jawa Barat\n(Identifikasi Multikolinearitas Count-Bias)",
          fontsize=12, fontweight="bold", pad=15)
plt.tight_layout()

heatmap_path = OUTPUT_FIGURES / "matriks_korelasi.png"
fig.savefig(heatmap_path, bbox_inches="tight")
# Simpan juga salinan untuk heatmap_korelasi.png
fig.savefig(OUTPUT_FIGURES / "heatmap_korelasi.png", bbox_inches="tight")
plt.close(fig)

print(f"[OK] Grafik heatmap korelasi tersimpan di:\n     -> {heatmap_path}")
print("Selesai Tahap 01.\n")
