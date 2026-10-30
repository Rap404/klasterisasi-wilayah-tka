"""
PETA TEMATIK SMP JAWA BARAT
  1) peta_klaster_k4.png            -> klasterisasi wilayah (K-Means, k=4)
  2) peta_prioritas_intervensi.png  -> kategori prioritas intervensi (IPI)

Input : dataset/jabar_smp_klaster_indeks.geojson  (poligon GADM + atribut klaster & indeks)
Output: clustering/statistik_grafik/grafik_eda/
Dependensi: numpy, pandas, matplotlib (pure python, tanpa geopandas)
"""
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# 1. Konfigurasi path
BASE_DIR = Path(__file__).resolve().parent.parent
GEOJSON_PATH = BASE_DIR / "dataset" / "jabar_smp_klaster_indeks.geojson"
GRAFIK_DIR = BASE_DIR / "clustering" / "statistik_grafik" / "grafik_eda"
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)

AMBANG_KOTA_KECIL_KM2 = 250   # wilayah lebih kecil dari ini dilabeli dengan callout di luar peta
AMBANG_ROBUST = 0.80          # masuk 8 besar pada >= 80% variasi bobot

# Pengelompokan callout (sisi peta tempat label diletakkan agar garis penunjuk tidak saling silang)
SISI_CALLOUT = {
    "Kota Depok": "barat", "Kota Bogor": "barat", "Kota Sukabumi": "barat",
    "Kota Bekasi": "utara", "Kota Cirebon": "utara",
    "Kota Cimahi": "selatan", "Kota Bandung": "selatan",
    "Kota Tasikmalaya": "selatan", "Kota Banjar": "selatan",
}

# Geser manual label wilayah besar yang saling berdekatan (meter; +x = timur, +y = utara)
GESER_LABEL = {"Kabupaten Cirebon": (20_000, -9_000)}


def to_utm48s(lon, lat):
    """Konversi WGS84 (lon, lat) ke UTM Zone 48S (EPSG:32748) dalam satuan meter."""
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    e_prime2 = e2 / (1 - e2)
    k0 = 0.9996
    lon0 = np.radians(105.0)

    phi = np.radians(lat)
    lam = np.radians(lon)

    N = a / np.sqrt(1 - e2 * np.sin(phi)**2)
    T = np.tan(phi)**2
    C = e_prime2 * np.cos(phi)**2
    A = (lam - lon0) * np.cos(phi)

    M = a * (
        (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * phi
        - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * np.sin(2*phi)
        + (15*e2**2/256 + 45*e2**3/1024) * np.sin(4*phi)
        - (35*e2**3/3072) * np.sin(6*phi)
    )

    x = k0 * N * (A + (1 - T + C) * A**3 / 6 + (5 - 18*T + T**2 + 72*C - 58*e_prime2) * A**5 / 120) + 500000.0
    y = k0 * (M + N * np.tan(phi) * (A**2 / 2 + (5 - T + 9*C + 4*C**2) * A**4 / 24 + (61 - 58*T + T**2 + 600*C - 330*e_prime2) * A**6 / 720)) + 10000000.0
    return x, y


def hitung_luas_km2(coords):
    """Hitung luas poligon dalam km2 menggunakan formula Shoelace."""
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 1e6


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
    ax.add_patch(MplPolygon([[x0 + 25_000, y0 + 40_000], [x0 + 19_000, y0 + 22_000], [x0 + 31_000, y0 + 22_000]],
                            fc="black", ec="black", zorder=8))
    ax.text(x0 + 25_000, y0 + 43_000, "U", ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=8)


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


# 2. Muat dan parse GeoJSON secara native
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    geojson_raw = json.load(f)

wilayah_list = []
all_xs, all_ys = [], []

for feat in geojson_raw["features"]:
    props = feat["properties"]
    geom = feat["geometry"]
    
    # Konversi koordinat ke UTM Zone 48S
    rings_utm = []
    total_luas = 0.0
    all_pts_feature = []

    for poly in geom["coordinates"]:
        exterior = poly[0]
        pts_utm = []
        for lon, lat in exterior:
            ux, uy = to_utm48s(lon, lat)
            pts_utm.append([ux, uy])
            all_xs.append(ux)
            all_ys.append(uy)
        
        arr = np.array(pts_utm)
        rings_utm.append(arr)
        total_luas += hitung_luas_km2(arr)
        all_pts_feature.append(arr)
    
    # Hitung titik jangkar/tengah (representative point)
    pts_concat = np.vstack(all_pts_feature)
    cx, cy = pts_concat[:, 0].mean(), pts_concat[:, 1].mean()
    titik = SimpleNamespace(x=cx, y=cy)
    
    item = dict(props)
    item["luas_km2"] = total_luas
    item["titik"] = titik
    item["rings_utm"] = rings_utm
    item["label"] = props["nama_kabkota"].replace("Kabupaten", "Kab.")
    item["robust"] = props.get("ipi_pct_top8_simulasi", 0.0) >= AMBANG_ROBUST
    wilayah_list.append(item)

df_wilayah = pd.DataFrame(wilayah_list)

minx, maxx = min(all_xs), max(all_xs)
miny, maxy = min(all_ys), max(all_ys)
total_bounds = (minx, miny, maxx, maxy)


def atur_sumbu(ax, bounds):
    minx, miny, maxx, maxy = bounds
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
    hiasan_peta(ax, bounds)


def gambar_poligon(ax, df, warna_kolom):
    patches = []
    colors = []
    for _, r in df.iterrows():
        c = r[warna_kolom]
        for ring in r["rings_utm"]:
            patches.append(MplPolygon(ring, closed=True))
            colors.append(c)
    pc = PatchCollection(patches, facecolors=colors, edgecolors="white", linewidths=0.9, zorder=2)
    ax.add_collection(pc)


def gambar_label(ax, df, fungsi_teks, bounds, ukuran_besar=7.2):
    minx, miny, maxx, maxy = bounds
    W, H = maxx - minx, maxy - miny
    halo = [pe.withStroke(linewidth=2.2, foreground="white")]

    # Wilayah besar: label di dalam poligon
    for _, r in df[df["luas_km2"] >= AMBANG_KOTA_KECIL_KM2].iterrows():
        dx, dy = GESER_LABEL.get(r["nama_kabkota"], (0, 0))
        ax.text(r["titik"].x + dx, r["titik"].y + dy, fungsi_teks(r), ha="center", va="center",
                fontsize=ukuran_besar, fontweight="bold", color="black", path_effects=halo, zorder=7)

    # Kota kecil: callout di luar peta
    kecil = df[df["luas_km2"] < AMBANG_KOTA_KECIL_KM2].copy()
    kecil["sisi"] = kecil["nama_kabkota"].map(SISI_CALLOUT)
    
    # Titik jangkar
    ax.scatter([r["titik"].x for _, r in kecil.iterrows()],
               [r["titik"].y for _, r in kecil.iterrows()],
               s=14, c="black", edgecolors="white", linewidths=0.8, zorder=10)
    pad = 0.06
    for sisi, grup in kecil.groupby("sisi"):
        if sisi in ("utara", "selatan"):
            grup = grup.sort_values("titik", key=lambda s: s.map(lambda p: p.x))
            xs = sebar([r["titik"].x for _, r in grup.iterrows()], 0.20 * W)
            y_label = maxy + pad * H if sisi == "utara" else miny - pad * H
            for (_, r), x_label in zip(grup.iterrows(), xs):
                ax.annotate(fungsi_teks(r), xy=(r["titik"].x, r["titik"].y), xytext=(x_label, y_label),
                            ha="center", va="bottom" if sisi == "utara" else "top",
                            fontsize=7.6, fontweight="bold", zorder=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", lw=0.6),
                            arrowprops=dict(arrowstyle="-", color="0.25", lw=0.8, shrinkA=0, shrinkB=1.5))
        else:  # barat
            grup = grup.sort_values("titik", key=lambda s: s.map(lambda p: p.y))
            ys = sebar([r["titik"].y for _, r in grup.iterrows()], 0.085 * H)
            x_label = minx - pad * W
            for (_, r), y_label in zip(grup.iterrows(), ys):
                ax.annotate(fungsi_teks(r), xy=(r["titik"].x, r["titik"].y), xytext=(x_label, y_label),
                            ha="right", va="center", fontsize=7.6, fontweight="bold", zorder=9,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.4", lw=0.6),
                            arrowprops=dict(arrowstyle="-", color="0.25", lw=0.8, shrinkA=0, shrinkB=1.5))


SUMBER = ("Sumber data: TKA SMP & data pokok pendidikan Kemendikdasmen, BPS (kemiskinan); "
          "batas administrasi: GADM. Pengolahan: tim peneliti.")

# =====================================================================
# PETA 1: KLASTER (K-Means, k = 4)
# =====================================================================
PALET_KLASTER = {1: "#2563eb", 2: "#0284c7", 3: "#d97706", 4: "#dc2626"}
df_wilayah["warna_klaster"] = df_wilayah["klaster_k4"].astype(int).map(PALET_KLASTER)

fig, ax, panel = siapkan_kanvas(
    "Peta Tematik Klasterisasi Wilayah SMP di Jawa Barat",
    "K-Means (k = 4) berdasarkan capaian TKA Matematika, rasio murid:guru, murid per sekolah, dan kemiskinan; 27 kabupaten/kota"
)

gambar_poligon(ax, df_wilayah, "warna_klaster")
atur_sumbu(ax, total_bounds)
gambar_label(ax, df_wilayah, lambda r: r["label"], total_bounds)

# Panel legenda klaster
y = 0.985
panel.text(0, y, "Klaster wilayah", fontsize=12, fontweight="bold", va="top")
y -= 0.055

for no in sorted(PALET_KLASTER):
    grup = df_wilayah[df_wilayah["klaster_k4"].astype(int) == no].sort_values("tka_rerata_mtk", ascending=False)
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
                                  "batas antarklaster dibaca sebagai kecenderungan.", 78),
           fontsize=8, color="0.3", va="bottom", style="italic")
fig.text(0.01, 0.03, SUMBER, fontsize=8, color="0.35", ha="left", va="center")

out1 = GRAFIK_DIR / "peta_klaster_k4.png"
fig.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Peta klaster berhasil diperbarui & disimpan di: {out1}")

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
df_wilayah["warna_prioritas"] = df_wilayah["kategori_prioritas"].map(PALET_PRIORITAS)

fig, ax, panel = siapkan_kanvas(
    "Peta Tematik Prioritas Intervensi Pembelajaran Matematika SMP di Jawa Barat",
    "Indeks Prioritas Intervensi: 50% defisit capaian TKA, 25% kemiskinan, 12,5% sekolah kecil, 12,5% beban guru"
)

gambar_poligon(ax, df_wilayah, "warna_prioritas")

# Garis tebal hitam untuk wilayah yang konsisten masuk 8 besar pada variasi bobot
for _, r in df_wilayah[df_wilayah["robust"]].iterrows():
    for ring in r["rings_utm"]:
        ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=2.4, zorder=5)

atur_sumbu(ax, total_bounds)
gambar_label(ax, df_wilayah, lambda r: f"{r['label']}\n#{int(r['peringkat_prioritas'])}" + (" ★" if r["robust"] else ""), total_bounds)

# Panel legenda prioritas
y = 0.985
panel.text(0, y, "Kategori prioritas (peringkat, skor)", fontsize=12, fontweight="bold", va="top")
y -= 0.05
for kat, warna in PALET_PRIORITAS.items():
    grup = df_wilayah[df_wilayah["kategori_prioritas"] == kat].sort_values("peringkat_prioritas")
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
print(f"Peta prioritas berhasil disimpan di: {out2}")