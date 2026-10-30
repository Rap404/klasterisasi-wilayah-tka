"""
06_INDEKS_PRIORITAS.PY
======================
Tahap 6 Pipeline: Penghitungan Indeks Prioritas Intervensi (IPI),
Simulasi Sensitivitas Bobot Monte Carlo (5.000 Iterasi),
dan Peringkat Kebutuhan Afirmasi 27 Kabupaten/Kota Jawa Barat.

Formula Komposit (Skor 0 - 100):
  IPI = 100 * (0.50 * D_tka + 0.25 * D_miskin + 0.125 * D_sekolah + 0.125 * D_guru)

Input:
  - dataset/processed/hasil_klaster_kmeans_smp.csv
Output:
  - dataset/processed/indeks_prioritas_intervensi_smp.csv
  - outputs/tables/peringkat_prioritas_intervensi.csv
  - outputs/figures/ranking_prioritas_intervensi.png
  - outputs/figures/kontribusi_komponen_indeks.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

# 1. Konfigurasi Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "processed" / "hasil_klaster_kmeans_smp.csv"
DATA_PROCESSED = BASE_DIR / "dataset" / "processed"
OUTPUT_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUT_FIGURES = BASE_DIR / "outputs" / "figures"

OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[06] FORMULA INDEKS PRIORITAS INTERVENSI (IPI) & MONTE CARLO")
print("=" * 60)

# 2. Muat Data
df = pd.read_csv(DATA_PATH, dtype={"kode_wilayah": str})

# 3. Bobot Indeks Komposit
BOBOT = {
    "norm_defisit_tka": 0.50,
    "norm_kemiskinan": 0.25,
    "norm_sekolah_kecil": 0.125,
    "norm_beban_guru": 0.125,
}
assert abs(sum(BOBOT.values()) - 1.0) < 1e-9
KOMPONEN = list(BOBOT.keys())
LABEL_KOMPONEN = {
    "norm_defisit_tka": "Defisit capaian TKA Mat (50%)",
    "norm_kemiskinan": "Kemiskinan (25%)",
    "norm_sekolah_kecil": "Sekolah kecil (12,5%)",
    "norm_beban_guru": "Beban guru (12,5%)",
}
WARNA_KOMPONEN = {
    "norm_defisit_tka": "#dc2626",
    "norm_kemiskinan": "#f59e0b",
    "norm_sekolah_kecil": "#2563eb",
    "norm_beban_guru": "#16a34a",
}


def minmax(s):
    return (s - s.min()) / (s.max() - s.min())


# 4. Normalisasi Min-Max Searah Urgensi (0 = Paling Aman, 1 = Paling Mendesak)
df["norm_defisit_tka"] = minmax(-df["tka_rerata_mtk"])          # TKA rendah -> mendesak
df["norm_kemiskinan"] = minmax(df["pct_penduduk_miskin"])       # miskin tinggi -> mendesak
df["norm_sekolah_kecil"] = minmax(-df["rasio_murid_sekolah"])   # sekolah kecil -> mendesak
df["norm_beban_guru"] = minmax(df["rasio_murid_guru"])          # beban guru tinggi -> mendesak

# 5. Skor Indeks Utama
df["skor_ipi"] = sum(BOBOT[k] * df[k] for k in KOMPONEN) * 100

# Pembanding: Hanya capaian + kemiskinan (50:50)
df["skor_pembanding_2komponen"] = (0.5 * df["norm_defisit_tka"] + 0.5 * df["norm_kemiskinan"]) * 100
df["peringkat_pembanding_2komponen"] = df["skor_pembanding_2komponen"].rank(ascending=False, method="min").astype(int)

# 6. Simulasi Sensitivitas Bobot Monte Carlo (5.000 Iterasi)
rng = np.random.default_rng(42)
N_SIM = 5000
matriks = df[KOMPONEN].values
peringkat_sim = np.zeros((N_SIM, len(df)), dtype=int)
for i in range(N_SIM):
    w_tka = rng.uniform(0.40, 0.60)
    w_lain = rng.dirichlet([1, 1, 1]) * (1 - w_tka)
    w = np.array([w_tka, *w_lain])
    skor_sim = matriks @ w
    peringkat_sim[i] = pd.Series(skor_sim).rank(ascending=False, method="min").values

df["median_peringkat_simulasi"] = np.median(peringkat_sim, axis=0)
df["pct_top5_simulasi"] = (peringkat_sim <= 5).mean(axis=0).round(2)
df["pct_top8_simulasi"] = (peringkat_sim <= 8).mean(axis=0).round(2)
df["rentang_peringkat_simulasi"] = [f"{a}-{b}" for a, b in zip(peringkat_sim.min(axis=0), peringkat_sim.max(axis=0))]


# 7. Tiering Kategori Prioritas
def tentukan_kategori(skor):
    if skor >= 70.0:
        return "Prioritas 1 (Mendesak / Urgent)"
    elif skor >= 60.0:
        return "Prioritas 2 (Tinggi / High)"
    elif skor >= 40.0:
        return "Prioritas 3 (Menengah / Moderate)"
    else:
        return "Prioritas 4 (Mandiri / Low)"


df["kategori_prioritas"] = df["skor_ipi"].apply(tentukan_kategori)

# 8. Urutkan Berdasarkan Skor
hasil_ranking = df.sort_values("skor_ipi", ascending=False).reset_index(drop=True)
hasil_ranking["peringkat"] = hasil_ranking.index + 1
hasil_ranking["skor_ipi"] = hasil_ranking["skor_ipi"].round(2)
hasil_ranking["skor_pembanding_2komponen"] = hasil_ranking["skor_pembanding_2komponen"].round(2)
for k in KOMPONEN:
    hasil_ranking[k] = hasil_ranking[k].round(3)

kolom_output = [
    "peringkat", "kode_wilayah", "nama_kabkota", "nama_pendek", "nama_klaster_k4",
    "skor_ipi", "kategori_prioritas",
    "tka_rerata_mtk", "pct_penduduk_miskin", "rasio_murid_sekolah", "rasio_murid_guru",
    *KOMPONEN,
    "median_peringkat_simulasi", "rentang_peringkat_simulasi", "pct_top5_simulasi", "pct_top8_simulasi",
    "skor_pembanding_2komponen", "peringkat_pembanding_2komponen",
]
hasil_final = hasil_ranking[kolom_output]

csv_dataset = DATA_PROCESSED / "indeks_prioritas_intervensi_smp.csv"
csv_tabel = OUTPUT_TABLES / "peringkat_prioritas_intervensi.csv"
hasil_final.to_csv(csv_dataset, index=False)
hasil_final.to_csv(csv_tabel, index=False)

print(f"[OK] Dataset IPI disimpan ke:\n     -> {csv_dataset}")
print(f"[OK] Tabel peringkat IPI disimpan ke:\n     -> {csv_tabel}")

print("\n=== TOP 10 WILAYAH PRIORITAS INTERVENSI ===")
print(hasil_final.head(10)[["peringkat", "nama_pendek", "skor_ipi", "kategori_prioritas",
                            "pct_top5_simulasi", "nama_klaster_k4"]].to_string(index=False))

# 9. Visualisasi Bar Chart Peringkat IPI
fig, ax = plt.subplots(figsize=(12, 11), dpi=200)
warna_kategori = {
    "Prioritas 1 (Mendesak / Urgent)": "#dc2626",
    "Prioritas 2 (Tinggi / High)": "#f97316",
    "Prioritas 3 (Menengah / Moderate)": "#eab308",
    "Prioritas 4 (Mandiri / Low)": "#16a34a",
}
plot_df = hasil_ranking.iloc[::-1].reset_index(drop=True)
plot_colors = [warna_kategori[k] for k in plot_df["kategori_prioritas"]]

bars = ax.barh(plot_df["nama_pendek"], plot_df["skor_ipi"], color=plot_colors,
              edgecolor="black", alpha=0.9, height=0.7)
ax.set_xlabel("Skor Indeks Prioritas Intervensi (0 - 100)", fontsize=11, fontweight="bold")
ax.set_ylabel("Kabupaten / Kota", fontsize=11, fontweight="bold")
ax.set_title("Peringkat Indeks Prioritas Intervensi Pembelajaran Matematika SMP Jawa Barat\n"
             "(50% Capaian TKA, 25% Kemiskinan, 12,5% Sekolah Kecil, 12,5% Beban Guru)",
             fontsize=12, fontweight="bold", pad=15)
ax.set_xlim(0, 100)
ax.grid(True, linestyle="--", alpha=0.5, axis="x")

for bar in bars:
    xval = bar.get_width()
    ax.text(xval + 1.0, bar.get_y() + bar.get_height() / 2, f"{xval:.1f}",
            va="center", ha="left", fontsize=9, fontweight="bold", color="#1e293b")

leg_elements = [Patch(facecolor=c, edgecolor="black", label=l) for l, c in warna_kategori.items()]
ax.legend(handles=leg_elements, loc="lower right", framealpha=0.95, fontsize=9.5)
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES / "ranking_prioritas_intervensi.png", bbox_inches="tight")
plt.close(fig)

# 10. Visualisasi Kontribusi Komponen Bobot IPI (Stacked Bar)
fig, ax = plt.subplots(figsize=(12, 11), dpi=200)
bottom = np.zeros(len(plot_df))

for komp in KOMPONEN:
    kontribusi = plot_df[komp] * BOBOT[komp] * 100
    ax.barh(plot_df["nama_pendek"], kontribusi, left=bottom,
            color=WARNA_KOMPONEN[komp], edgecolor="white", linewidth=0.5,
            label=LABEL_KOMPONEN[komp], height=0.7)
    bottom += kontribusi

ax.set_xlabel("Skor IPI Kumulatif (0 - 100)", fontsize=11, fontweight="bold")
ax.set_ylabel("Kabupaten / Kota", fontsize=11, fontweight="bold")
ax.set_title("Komposisi Kontribusi Indikator dalam Skor Indeks Prioritas Intervensi",
             fontsize=12, fontweight="bold", pad=15)
ax.set_xlim(0, 100)
ax.legend(loc="lower right", framealpha=0.95, fontsize=9.5)
ax.grid(True, linestyle="--", alpha=0.5, axis="x")
plt.tight_layout()
fig.savefig(OUTPUT_FIGURES / "kontribusi_komponen_indeks.png", bbox_inches="tight")
plt.close(fig)

print(f"[OK] Visualisasi peringkat & kontribusi komponen tersimpan di:\n     -> {OUTPUT_FIGURES}")
print("Selesai Tahap 06.\n")
