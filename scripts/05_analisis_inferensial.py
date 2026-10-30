"""
05_ANALISIS_INFERENSIAL.PY
==========================
Tahap 5 Pipeline: Analisis Inferensial Hubungan Variabel Pendukung,
Uji Signifikansi Pemisahan Klaster (ANOVA & Kruskal-Wallis),
serta Pemodelan Regresi Linier Berganda terhadap Capaian TKA.

Input:
  - dataset/processed/hasil_klaster_kmeans_smp.csv
Output:
  - outputs/tables/korelasi_tka_variabel_pendukung.csv
  - outputs/tables/regresi_berganda_tka.csv
  - outputs/figures/scatter_regresi_tka.png
  - outputs/figures/boxplot_tka_per_klaster.png
"""

from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# 1. Konfigurasi Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "processed" / "hasil_klaster_kmeans_smp.csv"
OUTPUT_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUT_FIGURES = BASE_DIR / "outputs" / "figures"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[05] ANALISIS INFERENSIAL, ANOVA & REGRESI BERGANDA")
print("=" * 60)

# 2. Muat Data
hasil_df = pd.read_csv(DATA_PATH)
target = "tka_rerata_mtk"
pendukung = ["rasio_murid_guru", "rasio_murid_sekolah", "pct_penduduk_miskin"]

# 3. Uji Korelasi Pearson & Spearman
tabel_korelasi = []
for p in pendukung:
    r_val, p_pearson = stats.pearsonr(hasil_df[target], hasil_df[p])
    rho_val, p_spearman = stats.spearmanr(hasil_df[target], hasil_df[p])
    slope, intercept, _, _, stderr = stats.linregress(hasil_df[p], hasil_df[target])
    tabel_korelasi.append({
        "variabel_pendukung": p,
        "pearson_r": round(r_val, 4),
        "pearson_p": round(p_pearson, 4),
        "spearman_rho": round(rho_val, 4),
        "spearman_p": round(p_spearman, 4),
        "slope_regresi": round(slope, 4),
        "intercept": round(intercept, 4),
        "stderr": round(stderr, 4),
    })

df_kor = pd.DataFrame(tabel_korelasi)
kor_csv_path = OUTPUT_TABLES / "korelasi_tka_variabel_pendukung.csv"
df_kor.to_csv(kor_csv_path, index=False)
print(f"[OK] Korelasi variabel pendukung disimpan ke:\n     -> {kor_csv_path}")
print(df_kor.to_string(index=False))

# 4. Regresi Linier Berganda (OLS via Linear Algebra)
X_cols = ["rasio_murid_guru", "rasio_murid_sekolah", "pct_penduduk_miskin"]
X_mat = hasil_df[X_cols].values
X_mat = np.column_stack([np.ones(len(X_mat)), X_mat])  # tambah intercept
y_vec = hasil_df[target].values

# Estimasi koefisien beta: (X^T X)^(-1) X^T y
beta, residuals, rank, s = np.linalg.lstsq(X_mat, y_vec, rcond=None)
y_pred = X_mat @ beta
ss_tot = np.sum((y_vec - y_vec.mean()) ** 2)
ss_res = np.sum((y_vec - y_pred) ** 2)
r2 = 1 - (ss_res / ss_tot)
n = len(y_vec)
p = len(X_cols)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
f_stat = ((ss_tot - ss_res) / p) / (ss_res / (n - p - 1))
f_pval = 1 - stats.f.cdf(f_stat, p, n - p - 1)

df_regresi = pd.DataFrame([{
    "R_squared": round(r2, 4),
    "Adjusted_R_squared": round(r2_adj, 4),
    "F_statistic": round(f_stat, 4),
    "p_value": round(f_pval, 6),
    "intercept": round(beta[0], 4),
    "coef_rasio_murid_guru": round(beta[1], 4),
    "coef_rasio_murid_sekolah": round(beta[2], 4),
    "coef_pct_penduduk_miskin": round(beta[3], 4),
}])

reg_csv_path = OUTPUT_TABLES / "regresi_berganda_tka.csv"
df_regresi.to_csv(reg_csv_path, index=False)
print(f"\n[OK] Model regresi berganda disimpan ke:\n     -> {reg_csv_path}")
print(f"Model Regresi: R² = {r2:.4f}, Adj R² = {r2_adj:.4f}, F = {f_stat:.2f} (p = {f_pval:.4e})")

# 5. Uji ANOVA & Kruskal-Wallis Antar Klaster
grup_klaster = [g[target].values for _, g in hasil_df.groupby("klaster_k4")]
f_anova, p_anova = stats.f_oneway(*grup_klaster)
h_kruskal, p_kruskal = stats.kruskal(*grup_klaster)

print(f"\nUji Signifikansi Pemisahan Klaster:")
print(f"- ANOVA Satu Arah: F = {f_anova:.4f}, p-value = {p_anova:.4e}")
print(f"- Kruskal-Wallis:  H = {h_kruskal:.4f}, p-value = {p_kruskal:.4e}")

# 6. Visualisasi Scatter Regresi
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=200)
labels_x = [
    "Rasio Murid per Guru",
    "Rasio Murid per Sekolah",
    "Persentase Penduduk Miskin (%)"
]
warna_klaster = {1: "#2563eb", 2: "#0ea5e9", 3: "#f59e0b", 4: "#dc2626"}

for i, col in enumerate(pendukung):
    ax = axes[i]
    for k_val, grp in hasil_df.groupby("klaster_k4"):
        ax.scatter(grp[col], grp[target], c=warna_klaster[k_val], s=60, edgecolors="black", alpha=0.85)
    
    # Garis tren regresi sederhana
    sl = df_kor.loc[df_kor["variabel_pendukung"] == col, "slope_regresi"].values[0]
    ic = df_kor.loc[df_kor["variabel_pendukung"] == col, "intercept"].values[0]
    r_val = df_kor.loc[df_kor["variabel_pendukung"] == col, "pearson_r"].values[0]
    p_val = df_kor.loc[df_kor["variabel_pendukung"] == col, "pearson_p"].values[0]
    
    x_line = np.linspace(hasil_df[col].min(), hasil_df[col].max(), 100)
    ax.plot(x_line, ic + sl * x_line, color="#dc2626", linestyle="--", linewidth=1.5,
            label=f"r = {r_val:.2f} (p={p_val:.3f})")
    
    ax.set_title(f"TKA vs {labels_x[i]}", fontsize=10, fontweight="bold")
    ax.set_xlabel(labels_x[i], fontsize=9)
    ax.set_ylabel("Nilai Rerata TKA MTK", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8.5)

plt.suptitle("Analisis Hubungan Capaian Akademik (TKA) dengan Variabel Pendukung",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES / "scatter_regresi_tka.png", bbox_inches="tight")
plt.close(fig)

# 7. Visualisasi Boxplot TKA per Klaster
fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
nama_klaster_label = [f"K{k}\n(n={len(g)})" for k, g in hasil_df.groupby("klaster_k4")]
bplot = ax.boxplot(grup_klaster, tick_labels=nama_klaster_label, patch_artist=True)

for patch, color in zip(bplot["boxes"], [warna_klaster[k] for k in sorted(warna_klaster.keys())]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title(f"Sebaran Nilai TKA Matematika per Klaster\n(ANOVA F = {f_anova:.2f}, p < 0.001)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Klaster Wilayah", fontsize=10)
ax.set_ylabel("Nilai Rerata TKA MTK", fontsize=10)
ax.grid(True, axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES / "boxplot_tka_per_klaster.png", bbox_inches="tight")
plt.close(fig)

print(f"[OK] Grafik inferensial dan boxplot tersimpan di:\n     -> {OUTPUT_FIGURES}")
print("Selesai Tahap 05.\n")
