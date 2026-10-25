"""
PETA TEMATIK SMP JAWA BARAT
  1) peta_klaster_k4.png            -> klasterisasi wilayah (K-Means, k=4)
  2) peta_prioritas_intervensi.png  -> kategori prioritas intervensi (IPI)

Input : dataset/jabar_smp_klaster_indeks.geojson  (poligon GADM + atribut klaster & indeks)
Output: clustering/statistik_grafik/grafik_eda/
Dependensi: geopandas, matplotlib
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon

# 1. Konfigurasi path
BASE_DIR = Path(__file__).resolve().parent.parent
GEOJSON_PATH = BASE_DIR / "dataset" / "jabar_smp_klaster_indeks.geojson"
GRAFIK_DIR = BASE_DIR / "clustering" / "statistik_grafik" / "grafik_eda"
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)

CRS_PROYEKSI = 32748          # UTM 48S (meter) agar bentuk wilayah dan skala bar akurat
AMBANG_KOTA_KECIL_KM2 = 250   # wilayah lebih kecil dari ini dilabeli dengan callout di luar peta
AMBANG_ROBUST = 0.80          # masuk 8 besar pada >= 80% variasi bobot

# 2. Muat data
gdf = gpd.read_file(GEOJSON_PATH)
gdf = gdf.to_crs(CRS_PROYEKSI)
gdf["luas_km2"] = gdf.geometry.area / 1e6
gdf["geometry"] = gdf.geometry.simplify(150, preserve_topology=True)   # ringankan tanpa mengubah tampilan
gdf["label"] = gdf["nama_kabkota"].str.replace("Kabupaten", "Kab.", regex=False)
gdf["robust"] = gdf["ipi_pct_top8_simulasi"] >= AMBANG_ROBUST
gdf["titik"] = gdf.geometry.representative_point()

# Pengelompokan callout (sisi peta tempat label diletakkan agar garis penunjuk tidak saling silang)
SISI_CALLOUT = {
    "Kota Depok": "barat", "Kota Bogor": "barat", "Kota Sukabumi": "barat",
    "Kota Bekasi": "utara", "Kota Cirebon": "utara",
    "Kota Cimahi": "selatan", "Kota Bandung": "selatan",
    "Kota Tasikmalaya": "selatan", "Kota Banjar": "selatan",
}


# Geser manual label wilayah besar yang saling berdekatan (meter; +x = timur, +y = utara)
GESER_LABEL = {"Kabupaten Cirebon": (20_000, -9_000)}


def fmt(x, d=1):
    """Format angka gaya Indonesia (koma desimal)."""
    return f"{x:.{d}f}".replace(".", ",")


def sebar(posisi_ideal, jarak_min):
    """Geser posisi label yang berdekatan agar tidak saling tumpang tindih (urut naik)."""
    hasil = []
    for p in posisi_ideal:
        hasil.append(p if not hasil else max(p, hasil[-1] + jarak_min))
    geser = (hasil[-1] - posisi_ideal[-1]) / 2 if hasil else 0
    return [h - geser for h in hasil]


def hiasan_peta(ax, batas):
    """Panah utara dan skala bar 50 km di pojok kiri bawah."""
    minx, miny, maxx, maxy = batas
    W, H = maxx - minx, maxy - miny
    x0, y0 = minx - 0.12 * W, miny - 0.07 * H
    ax.add_patch(Rectangle((x0, y0), 25_000, 3_500, fc="black", ec="black", zorder=8))
    ax.add_patch(Rectangle((x0 + 25_000, y0), 25_000, 3_500, fc="white", ec="black", zorder=8))
    for dx, teks in [(0, "0"), (25_000, "25"), (50_000, "50 km")]:
        ax.text(x0 + dx, y0 + 8_000, teks, ha="center", va="bottom", fontsize=8, zorder=8)
    ax.add_patch(Polygon([[x0 + 25_000, y0 + 40_000], [x0 + 19_000, y0 + 22_000], [x0 + 31_000, y0 + 22_000]],
                         fc="black", ec="black", zorder=8))
    ax.text(x0 + 25_000, y0 + 43_000, "U", ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=8)


def gambar_label(ax, gdf, fungsi_teks, ukuran_besar=7.2):
    """Label di dalam poligon untuk wilayah besar, dan callout di luar peta untuk kota kecil."""
    minx, miny, maxx, maxy = gdf.total_bounds
    W, H = maxx - minx, maxy - miny
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]

    # Wilayah besar: label di dalam poligon
    for _, r in gdf[gdf["luas_km2"] >= AMBANG_KOTA_KECIL_KM2].iterrows():
        dx, dy = GESER_LABEL.get(r["nama_kabkota"], (0, 0))
        ax.text(r["titik"].x + dx, r["titik"].y + dy, fungsi_teks(r), ha="center", va="center",
                fontsize=ukuran_besar, fontweight="bold", color="black", path_effects=halo, zorder=7)

    # Kota kecil: callout di luar peta
    kecil = gdf[gdf["luas_km2"] < AMBANG_KOTA_KECIL_KM2].copy()
    kecil["sisi"] = kecil["nama_kabkota"].map(SISI_CALLOUT)
    # titik jangkar agar kota kecil mudah ditemukan
    ax.scatter([p.x for p in kecil["titik"]], [p.y for p in kecil["titik"]], s=14, c="black",
               edgecolors="white", linewidths=0.8, zorder=10)
    pad = 0.06
    for sisi, grup in kecil.groupby("sisi"):
        if sisi in ("utara", "selatan"):
            grup = grup.sort_values("titik", key=lambda s: s.map(lambda p: p.x))
            xs = sebar([p.x for p in grup["titik"]], 0.20 * W)
            y_label = maxy + pad * H if sisi == "utara" else miny - pad * H
            for (_, r), x_label in zip(grup.iterrows(), xs):
                ax.annotate(fungsi_teks(r), xy=(r["titik"].x, r["titik"].y), xytext=(x_label, y_label),
                            ha="center", va="bottom" if sisi == "utara" else "top",
                            fontsize=7.6, fontweight="bold", zorder=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", lw=0.6),
                            arrowprops=dict(arrowstyle="-", color="0.25", lw=0.8, shrinkA=0, shrinkB=1.5))
        else:  # barat
            grup = grup.sort_values("titik", key=lambda s: s.map(lambda p: p.y))
            ys = sebar([p.y for p in grup["titik"]], 0.085 * H)
            x_label = minx - pad * W
            for (_, r), y_label in zip(grup.iterrows(), ys):
                ax.annotate(fungsi_teks(r), xy=(r["titik"].x, r["titik"].y), xytext=(x_label, y_label),
                            ha="right", va="center", fontsize=7.6, fontweight="bold", zorder=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", lw=0.6),
                            arrowprops=dict(arrowstyle="-", color="0.25", lw=0.8, shrinkA=0, shrinkB=1.5))


def siapkan_kanvas(judul, subjudul):
    fig = plt.figure(figsize=(15.5, 9.6), dpi=200)
    ax = fig.add_axes([0.01, 0.07, 0.63, 0.80])
    panel = fig.add_axes([0.655, 0.07, 0.335, 0.80])
    panel.axis("off")
    panel.set_xlim(0, 1)
    panel.set_ylim(0, 1)
    fig.text(0.01, 0.955, judul, fontsize=16, fontweight="bold", ha="left", va="center")
    fig.text(0.01, 0.915, subjudul, fontsize=10.5, color="0.3", ha="left", va="center")
    return fig, ax, panel


def atur_sumbu(ax, gdf):
    minx, miny, maxx, maxy = gdf.total_bounds
    W, H = maxx - minx, maxy - miny
    ax.set_xlim(minx - 0.20 * W, maxx + 0.05 * W)
    ax.set_ylim(miny - 0.15 * H, maxy + 0.12 * H)
    ax.set_aspect("equal")
    ax.set_facecolor("#eef3f8")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for s in ax.spines.values():
        s.set_edgecolor("0.6")
    hiasan_peta(ax, (minx, miny, maxx, maxy))


SUMBER = ("Sumber data: TKA SMP & data pokok pendidikan Kemendikdasmen, BPS (kemiskinan); "
          "batas administrasi: GADM. Pengolahan: tim peneliti.")

# =====================================================================
# PETA 1: KLASTER (K-Means, k = 4)
# =====================================================================
PALET_KLASTER = {1: "#2563eb", 2: "#16a34a", 3: "#f59e0b", 4: "#dc2626"}   # biru -> merah (capaian tinggi -> rendah)
gdf["warna_klaster"] = gdf["klaster_k4"].astype(int).map(PALET_KLASTER)

fig, ax, panel = siapkan_kanvas(
    "Peta Tematik Klasterisasi Wilayah SMP di Jawa Barat",
    "K-Means (k = 4) berdasarkan capaian TKA Matematika, rasio murid:guru, murid per sekolah, dan kemiskinan; 27 kabupaten/kota")

gdf.plot(ax=ax, color=gdf["warna_klaster"], edgecolor="white", linewidth=0.9, zorder=2)
atur_sumbu(ax, gdf)
gambar_label(ax, gdf, lambda r: r["label"])

# Panel legenda klaster
y = 0.985
panel.text(0, y, "Klaster wilayah", fontsize=12, fontweight="bold", va="top")
y -= 0.055
for no in sorted(PALET_KLASTER):
    grup = gdf[gdf["klaster_k4"].astype(int) == no].sort_values("tka_rerata_mtk", ascending=False)
    nama = grup["nama_klaster_k4"].iloc[0].split(": ", 1)[1]
    panel.add_patch(Rectangle((0, y - 0.028), 0.045, 0.028, fc=PALET_KLASTER[no], ec="black", lw=0.6))
    panel.text(0.06, y, f"Klaster {no}: {nama}", fontsize=10, fontweight="bold", va="top")
    y -= 0.038
    ringkas = (f"{len(grup)} wilayah | TKA Mat {fmt(grup['tka_rerata_mtk'].mean(), 2)} | "
               f"miskin {fmt(grup['pct_penduduk_miskin'].mean(), 2)}% | "
               f"murid:guru {fmt(grup['rasio_murid_guru'].mean())} | murid/sekolah {fmt(grup['rasio_murid_sekolah'].mean(), 0)}")
    panel.text(0.06, y, textwrap.fill(ringkas, 62), fontsize=8, color="0.25", va="top")
    y -= 0.062
    anggota = ", ".join(grup["label"].str.replace("Kota ", "Kota ").tolist())
    isi = textwrap.fill(anggota, 70)
    panel.text(0.06, y, isi, fontsize=8, va="top")
    y -= 0.027 * (isi.count("\n") + 1) + 0.04

panel.text(0, 0.02, textwrap.fill("Catatan: silhouette k = 4 sebesar 0,32 (struktur klaster tergolong lemah); "
                                  "batas antarklaster dibaca sebagai kecenderungan. " , 78),
           fontsize=8, color="0.3", va="bottom", style="italic")
fig.text(0.01, 0.03, SUMBER, fontsize=8, color="0.35", ha="left", va="center")
out1 = GRAFIK_DIR / "peta_klaster_k4.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Peta klaster disimpan di: {out1}")

# =====================================================================
# PETA 2: PRIORITAS INTERVENSI
# =====================================================================
PALET_PRIORITAS = {
    "Prioritas 1 (Mendesak / Urgent)": "#dc2626",
    "Prioritas 2 (Tinggi / High)": "#f97316",
    "Prioritas 3 (Menengah / Moderate)": "#eab308",
    "Prioritas 4 (Mandiri / Low)": "#16a34a",
}
KET_PRIORITAS = {
    "Prioritas 1 (Mendesak / Urgent)": "Prioritas 1 - Mendesak (skor >= 70)",
    "Prioritas 2 (Tinggi / High)": "Prioritas 2 - Tinggi (60 - 69,9)",
    "Prioritas 3 (Menengah / Moderate)": "Prioritas 3 - Menengah (40 - 59,9)",
    "Prioritas 4 (Mandiri / Low)": "Prioritas 4 - Mandiri (< 40)",
}
gdf["warna_prioritas"] = gdf["kategori_prioritas"].map(PALET_PRIORITAS)

fig, ax, panel = siapkan_kanvas(
    "Peta Tematik Prioritas Intervensi Pembelajaran Matematika SMP di Jawa Barat",
    "Indeks Prioritas Intervensi: 50% defisit capaian TKA, 25% kemiskinan, 12,5% sekolah kecil, 12,5% beban guru")

gdf.plot(ax=ax, color=gdf["warna_prioritas"], edgecolor="white", linewidth=0.9, zorder=2)
# Garis tebal hitam untuk wilayah yang konsisten masuk 8 besar pada variasi bobot
gdf[gdf["robust"]].boundary.plot(ax=ax, color="black", linewidth=2.4, zorder=5)
atur_sumbu(ax, gdf)
gambar_label(ax, gdf, lambda r: f"{r['label']}\n#{int(r['peringkat_prioritas'])}" + (" ★" if r["robust"] else ""))

# Panel legenda prioritas
y = 0.985
panel.text(0, y, "Kategori prioritas (peringkat, skor)", fontsize=12, fontweight="bold", va="top")
y -= 0.05
for kat, warna in PALET_PRIORITAS.items():
    grup = gdf[gdf["kategori_prioritas"] == kat].sort_values("peringkat_prioritas")
    panel.add_patch(Rectangle((0, y - 0.026), 0.045, 0.026, fc=warna, ec="black", lw=0.6))
    panel.text(0.06, y, f"{KET_PRIORITAS[kat]}  -  {len(grup)} wilayah", fontsize=9.5, fontweight="bold", va="top")
    y -= 0.034
    for _, r in grup.iterrows():
        tanda = " ★" if r["robust"] else ""
        panel.text(0.06, y, f"{int(r['peringkat_prioritas']):>2}.  {r['label']}{tanda}", fontsize=8, va="top", family="DejaVu Sans")
        panel.text(0.62, y, fmt(r["skor_ipi"], 1), fontsize=8, va="top", ha="right")
        y -= 0.0215
    y -= 0.022

panel.add_patch(Rectangle((0, 0.108), 0.045, 0.024, fc="white", ec="black", lw=2.2))
panel.text(0.06, 0.132, "Garis tebal / ★: masuk 8 besar pada >= 80%\nvariasi bobot (5.000 simulasi)", fontsize=8.3, va="top")
panel.text(0, 0.02, textwrap.fill("Catatan: peringkat 6-12 sensitif terhadap bobot; baca sebagai kelompok, "
                                  "bukan urutan pasti. Pembanding dua komponen (capaian + kemiskinan) berkorelasi peringkat 0,92.", 78),
           fontsize=8, color="0.3", va="bottom", style="italic")
fig.text(0.01, 0.03, SUMBER, fontsize=8, color="0.35", ha="left", va="center")
out2 = GRAFIK_DIR / "peta_prioritas_intervensi.png"
fig.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Peta prioritas disimpan di: {out2}")