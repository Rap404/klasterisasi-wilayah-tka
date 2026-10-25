"""
INDEKS PRIORITAS INTERVENSI (IPI) PEMBELAJARAN MATEMATIKA SMP - JAWA BARAT

Rumus (skor 0-100, makin tinggi makin prioritas):

    IPI = 100 x ( 0,50 x D_capaian      defisit TKA Matematika
                + 0,25 x D_kemiskinan   persentase penduduk miskin
                + 0,125 x D_sekolah     sekolah berukuran kecil (murid per sekolah rendah)
                + 0,125 x D_guru )      beban guru tinggi (rasio murid:guru tinggi)

Setiap indikator di-min-max ke rentang 0-1 dengan arah "makin tinggi = makin mendesak":
    D_capaian    = (TKA_maks - TKA) / (TKA_maks - TKA_min)
    D_kemiskinan = (Miskin - Miskin_min) / (Miskin_maks - Miskin_min)
    D_sekolah    = (MS_maks - MS) / (MS_maks - MS_min)          MS = murid per sekolah
    D_guru       = (MG - MG_min) / (MG_maks - MG_min)           MG = murid per guru

Dasar pemilihan (lihat juga analisis_variabel_pendukung.py, bagian regresi berganda):
  - Capaian (50%): hasil yang ingin diperbaiki, jadi menjadi poros utama.
  - Kemiskinan (25%): faktor yang paling kuat berkaitan dengan capaian (rho = -0,58).
  - Ukuran sekolah (12,5%): berkaitan dengan capaian bahkan setelah kemiskinan dan status kota
    dikendalikan; sekolah kecil/tersebar menandai layanan yang sulit dijangkau dan mahal.
  - Beban guru (12,5%): korelasi sederhananya dengan TKA lemah (rho = 0,11) karena wilayah miskin
    justru punya rasio longgar; setelah kemiskinan dan ukuran sekolah dikendalikan, rasio tinggi
    berkaitan dengan capaian lebih rendah. Dibuat berbobot kecil karena bukti ini eksploratif (n = 27).
  - Klaster TIDAK dimasukkan ke rumus agar tidak terjadi hitung ganda; klaster dipakai untuk
    menafsirkan hasil (strategi intervensi per tipe wilayah).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from scipy.stats import spearmanr

# 1. Konfigurasi direktori
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "hasil_klaster_kmeans_smp.csv"
OUTPUT_DIR = BASE_DIR / "clustering" / "statistik_grafik"
GRAFIK_DIR = OUTPUT_DIR / "grafik_eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)

# 2. Muat dataset hasil klasterisasi
df = pd.read_csv(DATA_PATH, dtype={"kode_wilayah": str})

# 3. Bobot indeks
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


# 4. Normalisasi Min-Max searah urgensi (0 = paling aman, 1 = paling mendesak)
df["norm_defisit_tka"] = minmax(-df["tka_rerata_mtk"])          # TKA rendah -> mendesak
df["norm_kemiskinan"] = minmax(df["pct_penduduk_miskin"])       # miskin tinggi -> mendesak
df["norm_sekolah_kecil"] = minmax(-df["rasio_murid_sekolah"])   # sekolah kecil -> mendesak
df["norm_beban_guru"] = minmax(df["rasio_murid_guru"])          # beban guru tinggi -> mendesak

# 5. Skor indeks utama
df["skor_ipi"] = sum(BOBOT[k] * df[k] for k in KOMPONEN) * 100

# Pembanding: hanya capaian + kemiskinan (50:50)
df["skor_pembanding_2komponen"] = (0.5 * df["norm_defisit_tka"] + 0.5 * df["norm_kemiskinan"]) * 100
df["peringkat_pembanding_2komponen"] = df["skor_pembanding_2komponen"].rank(ascending=False, method="min").astype(int)

# 6. Uji sensitivitas bobot (simulasi): bobot capaian 40-60%, sisanya dibagi acak ke tiga indikator lain
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


# 7. Klasifikasi kategori prioritas (tiering)
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

# 8. Urutkan berdasarkan skor (Peringkat 1 s/d N)
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

csv_dataset = BASE_DIR / "dataset" / "indeks_prioritas_intervensi_smp.csv"
csv_statistik = OUTPUT_DIR / "peringkat_prioritas_intervensi.csv"
hasil_final.to_csv(csv_dataset, index=False)
hasil_final.to_csv(csv_statistik, index=False)

print("=== 10 BESAR DAERAH PRIORITAS UTAMA INTERVENSI PENDIDIKAN ===")
print(hasil_final.head(10)[["peringkat", "nama_pendek", "skor_ipi", "kategori_prioritas",
                            "pct_top5_simulasi", "nama_klaster_k4"]].to_string(index=False))

rho = spearmanr(hasil_ranking["skor_ipi"], hasil_ranking["skor_pembanding_2komponen"])[0]
print(f"\nKorelasi peringkat indeks utama vs pembanding 2 komponen: {rho:.2f}")
print("Ringkasan simulasi bobot (5000 variasi):")
konsisten = hasil_final[hasil_final["pct_top8_simulasi"] >= 0.80]["nama_pendek"].tolist()
print(f"  Wilayah masuk 8 besar pada >= 80% variasi bobot: {konsisten}")
print("\nJumlah wilayah per kategori:")
print(hasil_final["kategori_prioritas"].value_counts().sort_index().to_string())

print("\n=== 5 DAERAH PALING MANDIRI (PRIORITAS RENDAH) ===")
print(hasil_final.tail(5)[["peringkat", "nama_pendek", "skor_ipi", "kategori_prioritas",
                           "nama_klaster_k4"]].to_string(index=False))

# 9. Grafik 1: peringkat berdasarkan kategori prioritas
plt.figure(figsize=(12, 11))
warna_kategori = {
    "Prioritas 1 (Mendesak / Urgent)": "#dc2626",
    "Prioritas 2 (Tinggi / High)": "#f97316",
    "Prioritas 3 (Menengah / Moderate)": "#eab308",
    "Prioritas 4 (Mandiri / Low)": "#16a34a",
}
plot_df = hasil_ranking.iloc[::-1].reset_index(drop=True)
plot_colors = [warna_kategori[k] for k in plot_df["kategori_prioritas"]]

bars = plt.barh(plot_df["nama_pendek"], plot_df["skor_ipi"], color=plot_colors,
                edgecolor="black", alpha=0.9, height=0.7)
plt.xlabel("Skor Indeks Prioritas Intervensi (0 - 100)", fontsize=11, fontweight="bold")
plt.ylabel("Kabupaten / Kota", fontsize=11, fontweight="bold")
plt.title("Peringkat Indeks Prioritas Intervensi Pembelajaran Matematika SMP Jawa Barat\n"
          "(50% Capaian TKA, 25% Kemiskinan, 12,5% Sekolah Kecil, 12,5% Beban Guru)",
          fontsize=13, fontweight="bold", pad=15)
plt.xlim(0, 100)
plt.grid(True, linestyle="--", alpha=0.5, axis="x")
for bar in bars:
    xval = bar.get_width()
    plt.text(xval + 1.0, bar.get_y() + bar.get_height() / 2, f"{xval:.1f}",
             va="center", fontsize=8.5, fontweight="bold")
legend_elements = [
    Patch(facecolor="#dc2626", edgecolor="black", label="Prioritas 1: Mendesak (Skor >= 70)"),
    Patch(facecolor="#f97316", edgecolor="black", label="Prioritas 2: Tinggi (Skor 60 - 69,9)"),
    Patch(facecolor="#eab308", edgecolor="black", label="Prioritas 3: Menengah (Skor 40 - 59,9)"),
    Patch(facecolor="#16a34a", edgecolor="black", label="Prioritas 4: Mandiri (Skor < 40)"),
]
plt.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.95)
plt.tight_layout()
grafik_path = GRAFIK_DIR / "ranking_prioritas_intervensi.png"
plt.savefig(grafik_path, dpi=200, bbox_inches="tight")
plt.close()

# 10. Grafik 2: kontribusi tiap komponen terhadap skor (menjelaskan "mengapa" peringkat tersebut)
fig, ax = plt.subplots(figsize=(12, 11))
kiri = np.zeros(len(plot_df))
for k in KOMPONEN:
    kontribusi = BOBOT[k] * plot_df[k].values * 100
    ax.barh(plot_df["nama_pendek"], kontribusi, left=kiri, color=WARNA_KOMPONEN[k],
            edgecolor="black", height=0.7, label=LABEL_KOMPONEN[k], alpha=0.9)
    kiri += kontribusi
ax.set_xlabel("Kontribusi terhadap Skor Indeks (0 - 100)", fontsize=11, fontweight="bold")
ax.set_title("Dekomposisi Skor Indeks Prioritas per Komponen", fontsize=13, fontweight="bold", pad=12)
ax.set_xlim(0, 100)
ax.grid(True, linestyle="--", alpha=0.5, axis="x")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
plt.tight_layout()
grafik_kontribusi = GRAFIK_DIR / "kontribusi_komponen_indeks.png"
plt.savefig(grafik_kontribusi, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nGrafik ranking prioritas berhasil disimpan di: {grafik_path}")
print(f"Grafik dekomposisi komponen berhasil disimpan di: {grafik_kontribusi}")
print(f"Data ranking prioritas berhasil disimpan di: {csv_dataset}")