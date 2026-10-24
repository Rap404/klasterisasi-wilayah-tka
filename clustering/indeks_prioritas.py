import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Konfigurasi direktori
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset" / "hasil_klaster_kmeans_smp.csv"
OUTPUT_DIR = BASE_DIR / "clustering" / "statistik_grafik"
GRAFIK_DIR = OUTPUT_DIR / "grafik_eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)

# 2. Muat dataset hasil klasterisasi
df = pd.read_csv(DATA_PATH)

# 3. Normalisasi Min-Max Selaras Arah Urgensi (0 = paling mandiri/aman, 1 = paling darurat)
# A. Defisit TKA Matematika (makin rendah nilai TKA, makin darurat butuh pemulihan)
tka_max = df["tka_rerata_mtk"].max()
tka_min = df["tka_rerata_mtk"].min()
df["norm_defisit_tka"] = (tka_max - df["tka_rerata_mtk"]) / (tka_max - tka_min)

# B. Persentase Penduduk Miskin (makin tinggi kemiskinan, makin darurat butuh afirmasi)
miskin_max = df["pct_penduduk_miskin"].max()
miskin_min = df["pct_penduduk_miskin"].min()
df["norm_kemiskinan"] = (df["pct_penduduk_miskin"] - miskin_min) / (miskin_max - miskin_min)

# C. Rasio Murid per Guru (makin tinggi rasio, makin darurat krisis guru)
guru_max = df["rasio_murid_guru"].max()
guru_min = df["rasio_murid_guru"].min()
df["norm_rasio_guru"] = (df["rasio_murid_guru"] - guru_min) / (guru_max - guru_min)

# D. Rasio Murid per Sekolah (makin tinggi rasio, makin darurat kepadatan gedung/kelas)
sekolah_max = df["rasio_murid_sekolah"].max()
sekolah_min = df["rasio_murid_sekolah"].min()
df["norm_rasio_sekolah"] = (df["rasio_murid_sekolah"] - sekolah_min) / (sekolah_max - sekolah_min)

# 4. Formulasi Pembobotan Indeks
# SKENARIO UTAMA (Opsi A - Kebijakan Seimbang):
# 30% Defisit TKA, 30% Kemiskinan, 20% Rasio Guru, 20% Rasio Sekolah
w_tka, w_miskin, w_guru, w_sekolah = 0.30, 0.30, 0.20, 0.20

df["skor_ipi"] = (
    w_tka * df["norm_defisit_tka"]
    + w_miskin * df["norm_kemiskinan"]
    + w_guru * df["norm_rasio_guru"]
    + w_sekolah * df["norm_rasio_sekolah"]
) * 100

# Hitung juga skenario pembanding
# Skenario B (Afirmasi Mutu & Kesejahteraan: 35% TKA, 35% Miskin, 15% Guru, 15% Sekolah)
df["skor_ipi_skenario_b"] = (
    0.35 * df["norm_defisit_tka"]
    + 0.35 * df["norm_kemiskinan"]
    + 0.15 * df["norm_rasio_guru"]
    + 0.15 * df["norm_rasio_sekolah"]
) * 100

# 5. Klasifikasi Kategori Prioritas Intervensi (Tiering)
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

# 6. Urutkan berdasarkan skor prioritas tertinggi (Peringkat 1 s/d 27)
hasil_ranking = df.sort_values("skor_ipi", ascending=False).reset_index(drop=True)
hasil_ranking["peringkat"] = hasil_ranking.index + 1

# Bulatkan skor
hasil_ranking["skor_ipi"] = hasil_ranking["skor_ipi"].round(2)
hasil_ranking["skor_ipi_skenario_b"] = hasil_ranking["skor_ipi_skenario_b"].round(2)

# Kolom yang akan disimpan
kolom_output = [
    "peringkat",
    "kode_wilayah",
    "nama_kabkota",
    "nama_pendek",
    "nama_klaster_k3",
    "skor_ipi",
    "kategori_prioritas",
    "tka_rerata_mtk",
    "pct_penduduk_miskin",
    "rasio_murid_guru",
    "rasio_murid_sekolah",
    "skor_ipi_skenario_b"
]

hasil_final = hasil_ranking[kolom_output]

# Simpan ke CSV di direktori dataset dan statistik_grafik
csv_dataset = BASE_DIR / "dataset" / "indeks_prioritas_intervensi_smp.csv"
csv_statistik = OUTPUT_DIR / "peringkat_prioritas_intervensi.csv"

hasil_final.to_csv(csv_dataset, index=False)
hasil_final.to_csv(csv_statistik, index=False)

print("=== 10 BESAR DAERAH PRIORITAS UTAMA INTERVENSI PENDIDIKAN ===")
print(hasil_final.head(10)[["peringkat", "nama_pendek", "skor_ipi", "kategori_prioritas", "nama_klaster_k3"]].to_string(index=False))

print("\n=== 5 DAERAH PALING MANDIRI (PRIORITAS RENDAH) ===")
print(hasil_final.tail(5)[["peringkat", "nama_pendek", "skor_ipi", "kategori_prioritas", "nama_klaster_k3"]].to_string(index=False))

# 7. Visualisasi Diagram Batang Peringkat Prioritas Intervensi (27 Daerah)
plt.figure(figsize=(12, 11))

# Warna berdasarkan kategori prioritas
warna_kategori = {
    "Prioritas 1 (Mendesak / Urgent)": "#dc2626",     # Merah tua
    "Prioritas 2 (Tinggi / High)": "#f97316",         # Oranye terang
    "Prioritas 3 (Menengah / Moderate)": "#eab308",     # Kuning emas
    "Prioritas 4 (Mandiri / Low)": "#16a34a"          # Hijau
}

bar_colors = [warna_kategori[kat] for kat in hasil_ranking["kategori_prioritas"]]

# Urutkan terbalik untuk horizontal bar plot (ranking 1 berada di paling atas)
plot_df = hasil_ranking.iloc[::-1].reset_index(drop=True)
plot_colors = bar_colors[::-1]

bars = plt.barh(plot_df["nama_pendek"], plot_df["skor_ipi"], color=plot_colors, edgecolor="black", alpha=0.9, height=0.7)

plt.xlabel("Skor Indeks Prioritas Intervensi (0 - 100)", fontsize=11, fontweight="bold")
plt.ylabel("Kabupaten / Kota", fontsize=11, fontweight="bold")
plt.title("Peringkat Indeks Prioritas Intervensi Pendidikan SMP Jawa Barat\n(Formula Seimbang: 30% TKA, 30% Kemiskinan, 20% Guru, 20% Sarpras)", fontsize=13, fontweight="bold", pad=15)
plt.xlim(0, 100)
plt.grid(True, linestyle="--", alpha=0.5, axis="x")

# Tambahkan label angka skor di ujung batang
for bar in bars:
    xval = bar.get_width()
    plt.text(xval + 1.0, bar.get_y() + bar.get_height() / 2, f"{xval:.1f}", va="center", fontsize=8.5, fontweight="bold")

# Tambahkan legenda kategori
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#dc2626", edgecolor="black", label="Prioritas 1: Mendesak (Skor ≥ 70)"),
    Patch(facecolor="#f97316", edgecolor="black", label="Prioritas 2: Tinggi (Skor 60 - 69.9)"),
    Patch(facecolor="#eab308", edgecolor="black", label="Prioritas 3: Menengah (Skor 40 - 59.9)"),
    Patch(facecolor="#16a34a", edgecolor="black", label="Prioritas 4: Mandiri (Skor < 40)")
]
plt.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.95)

plt.tight_layout()
grafik_path = GRAFIK_DIR / "ranking_prioritas_intervensi.png"
plt.savefig(grafik_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nGrafik ranking prioritas berhasil disimpan di: {grafik_path}")
print(f"Data ranking prioritas berhasil disimpan di: {csv_dataset}")
