"""
07_VISUALISASI_GEOSPASIAL.PY
============================
Tahap 7 Pipeline: Integrasi Data Geospasial Poligon Administrasi,
Pembuatan Peta Tematik Kartografi Resolusi Tinggi (UTM 48S),
serta Pembuatan Peta Web GIS Interaktif (Leaflet.js).

Input:
  - dataset/raw/geospasial/kabkota_jabar_keyed.geojson
  - dataset/processed/indeks_prioritas_intervensi_smp.csv
Output:
  - dataset/processed/kabkota_jabar_klaster_indeks.geojson
  - outputs/figures/peta_klaster_k4.png
  - outputs/figures/peta_prioritas_intervensi.png
  - outputs/figures/peta_spasial_klaster_dan_prioritas.png
  - outputs/peta_interaktif_jabar.html
  - docs/index.html
"""

import json
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# 1. Konfigurasi Path
BASE_DIR = Path(__file__).resolve().parent.parent
GEOJSON_RAW = BASE_DIR / "dataset" / "raw" / "geospasial" / "kabkota_jabar_keyed.geojson"
CSV_IPI = BASE_DIR / "dataset" / "processed" / "indeks_prioritas_intervensi_smp.csv"
GEOJSON_OUT = BASE_DIR / "dataset" / "processed" / "kabkota_jabar_klaster_indeks.geojson"
OUTPUT_FIGURES = BASE_DIR / "outputs" / "figures"
HTML_OUT = BASE_DIR / "outputs" / "peta_interaktif_jabar.html"
DOCS_OUT = BASE_DIR / "docs" / "index.html"

OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)
DOCS_OUT.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("[07] INTEGRASI GEOSPASIAL, PETA TEMATIK & WEB GIS INTERAKTIF")
print("=" * 60)

# 2. Muat Data
df_ipi = pd.read_csv(CSV_IPI)
df_ipi["kode_wilayah_std"] = df_ipi["kode_wilayah"].apply(lambda x: f"{float(x):.2f}")

ipi_lookup = {}
for _, row in df_ipi.iterrows():
    d = row.to_dict()
    ipi_lookup[str(row["kode_wilayah_std"])] = d
    ipi_lookup[str(row["nama_kabkota"])] = d
    if "kode_bps" in row and pd.notnull(row["kode_bps"]):
        ipi_lookup[str(int(row["kode_bps"]))] = d

with open(GEOJSON_RAW, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

# 3. Gabungkan Data Atribut ke GeoJSON Properties
matched_count = 0
for feat in geojson_data["features"]:
    props = feat["properties"]
    bps_k = str(props.get("kode_bps", ""))
    wil_k = str(props.get("kode_wilayah", ""))
    nm_k = str(props.get("nama_kabkota", ""))

    matched = ipi_lookup.get(bps_k) or ipi_lookup.get(wil_k) or ipi_lookup.get(nm_k)
    if matched:
        matched_count += 1
        props["nama_pendek"] = matched["nama_pendek"]
        props["klaster_k4"] = matched.get("nama_klaster_k4", "")
        props["skor_ipi"] = float(matched["skor_ipi"])
        props["peringkat_ipi"] = int(matched["peringkat"])
        props["peringkat_prioritas"] = int(matched["peringkat"])
        props["kategori_prioritas"] = matched["kategori_prioritas"]
        props["tka_rerata_mtk"] = float(matched["tka_rerata_mtk"])
        props["pct_penduduk_miskin"] = float(matched["pct_penduduk_miskin"])
        props["rasio_murid_guru"] = float(matched["rasio_murid_guru"])
        props["rasio_murid_sekolah"] = float(matched["rasio_murid_sekolah"])
        if "pct_top8_simulasi" in matched:
            props["ipi_pct_top8_simulasi"] = float(matched["pct_top8_simulasi"])
        if "rentang_peringkat_simulasi" in matched:
            props["ipi_rentang_peringkat_simulasi"] = str(matched["rentang_peringkat_simulasi"])

with open(GEOJSON_OUT, "w", encoding="utf-8") as f:
    json.dump(geojson_data, f, ensure_ascii=False, indent=2)

print(f"[OK] GeoJSON terintegrasi berhasil disimpan ({matched_count}/27 kab/kota):\n     -> {GEOJSON_OUT}")

# 4. Fungsi Proyeksi UTM Zone 48S & Kartografi Resolusi Tinggi
def to_utm48s(lon, lat):
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
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 1e6


def fmt(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def sebar(posisi_ideal, jarak_min):
    hasil = []
    for p in posisi_ideal:
        hasil.append(p if not hasil else max(p, hasil[-1] + jarak_min))
    geser = (hasil[-1] - posisi_ideal[-1]) / 2 if hasil else 0
    return [h - geser for h in hasil]


def hiasan_peta(ax, batas):
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
    fig.text(0.01, 0.955, judul, fontsize=15, fontweight="bold", ha="left", va="center")
    fig.text(0.01, 0.915, subjudul, fontsize=10, color="0.3", ha="left", va="center")
    return fig, ax, panel


# Parsing data spasial poligon UTM
wilayah_list = []
all_xs, all_ys = [], []

for feat in geojson_data["features"]:
    props = feat["properties"]
    geom = feat["geometry"]
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
    
    pts_gabungan = np.vstack(all_pts_feature)
    cx, cy = float(pts_gabungan[:, 0].mean()), float(pts_gabungan[:, 1].mean())

    pct_top8 = props.get("ipi_pct_top8_simulasi", 0.0)
    wilayah_list.append({
        "nama": props["nama_kabkota"],
        "label": props.get("nama_pendek", props["nama_kabkota"]),
        "luas_km2": total_luas,
        "cx": cx, "cy": cy,
        "rings_utm": rings_utm,
        "klaster_k4": props.get("klaster_k4", ""),
        "kategori_prioritas": props.get("kategori_prioritas", ""),
        "skor_ipi": props.get("skor_ipi", 0.0),
        "peringkat_prioritas": props.get("peringkat_prioritas", 0),
        "tka_rerata_mtk": props.get("tka_rerata_mtk", 0.0),
        "pct_penduduk_miskin": props.get("pct_penduduk_miskin", 0.0),
        "rasio_murid_sekolah": props.get("rasio_murid_sekolah", 0.0),
        "rasio_murid_guru": props.get("rasio_murid_guru", 0.0),
        "robust": pct_top8 >= 0.80,
    })

df_wilayah = pd.DataFrame(wilayah_list)
total_bounds = (min(all_xs), min(all_ys), max(all_xs), max(all_ys))

AMBANG_KOTA_KECIL_KM2 = 250
SISI_CALLOUT = {
    "Kota Depok": "barat", "Kota Bogor": "barat", "Kota Sukabumi": "barat",
    "Kota Bekasi": "utara", "Kota Cirebon": "utara",
    "Kota Cimahi": "selatan", "Kota Bandung": "selatan",
    "Kota Tasikmalaya": "selatan", "Kota Banjar": "selatan",
}
GESER_LABEL = {"Kabupaten Cirebon": (20_000, -9_000)}

def gambar_poligon(ax, df_w, kolom_warna):
    for _, r in df_w.iterrows():
        warna = r[kolom_warna]
        for ring in r["rings_utm"]:
            ax.add_patch(MplPolygon(ring, closed=True, facecolor=warna, edgecolor="#ffffff", linewidth=0.7, zorder=3))


def atur_sumbu(ax, batas):
    minx, miny, maxx, maxy = batas
    W, H = maxx - minx, maxy - miny
    ax.set_xlim(minx - 0.16 * W, maxx + 0.08 * W)
    ax.set_ylim(miny - 0.10 * H, maxy + 0.10 * H)
    ax.set_aspect("equal")
    ax.axis("off")
    hiasan_peta(ax, batas)


def gambar_label(ax, df_w, pembuat_teks, batas):
    minx, miny, maxx, maxy = batas
    W, H = maxx - minx, maxy - miny
    efek = [pe.withStroke(linewidth=2.5, foreground="white")]

    besar = df_w[df_w["luas_km2"] >= AMBANG_KOTA_KECIL_KM2]
    for _, r in besar.iterrows():
        tx, ty = r["cx"], r["cy"]
        if r["nama"] in GESER_LABEL:
            dx, dy = GESER_LABEL[r["nama"]]
            tx += dx
            ty += dy
        ax.text(tx, ty, pembuat_teks(r), fontsize=7.5, ha="center", va="center",
                color="black", path_effects=efek, zorder=6, fontweight="bold")

    kecil = df_w[df_w["luas_km2"] < AMBANG_KOTA_KECIL_KM2].copy()
    kecil["sisi"] = kecil["nama"].map(SISI_CALLOUT).fillna("utara")

    grup_sisi = {
        "barat": kecil[kecil["sisi"] == "barat"].sort_values("cy"),
        "utara": kecil[kecil["sisi"] == "utara"].sort_values("cx"),
        "selatan": kecil[kecil["sisi"] == "selatan"].sort_values("cx"),
    }
    y_barat = sebar(grup_sisi["barat"]["cy"].tolist(), 0.075 * H)
    x_utara = sebar(grup_sisi["utara"]["cx"].tolist(), 0.095 * W)
    x_selatan = sebar(grup_sisi["selatan"]["cx"].tolist(), 0.095 * W)

    target_posisi = {}
    for (_, r), y in zip(grup_sisi["barat"].iterrows(), y_barat):
        target_posisi[r["nama"]] = (minx - 0.095 * W, y, "right", "center")
    for (_, r), x in zip(grup_sisi["utara"].iterrows(), x_utara):
        target_posisi[r["nama"]] = (x, maxy + 0.055 * H, "center", "bottom")
    for (_, r), x in zip(grup_sisi["selatan"].iterrows(), x_selatan):
        target_posisi[r["nama"]] = (x, miny - 0.065 * H, "center", "top")

    for _, r in kecil.iterrows():
        tx, ty, ha, va = target_posisi[r["nama"]]
        ax.annotate(
            pembuat_teks(r),
            xy=(r["cx"], r["cy"]),
            xytext=(tx, ty),
            fontsize=7.5,
            fontweight="bold",
            ha=ha, va=va,
            color="#0f172a",
            path_effects=efek,
            zorder=7,
            arrowprops=dict(arrowstyle="-", color="#334155", lw=0.9, alpha=0.8, shrinkA=2, shrinkB=2)
        )


# --- 5. Cetak Peta 1: Peta Klaster K-Means (k=4) ---
PALET_KLASTER = {
    "Klaster 1: Perkotaan, Capaian Tinggi": "#2563eb",
    "Klaster 2: Beban Kelas Tinggi, Capaian Menengah": "#0ea5e9",
    "Klaster 3: Kemiskinan Tinggi, Capaian Menengah": "#f59e0b",
    "Klaster 4: Capaian Rendah, Skala Sekolah Terbatas": "#dc2626",
}
KET_KLASTER = {
    "Klaster 1: Perkotaan, Capaian Tinggi": "Perkotaan - Capaian Tinggi, Kemiskinan Rendah",
    "Klaster 2: Beban Kelas Tinggi, Capaian Menengah": "Beban Kelas Tinggi - Rasio Murid:Guru Padat",
    "Klaster 3: Kemiskinan Tinggi, Capaian Menengah": "Tantangan Sosioekonomi - Kemiskinan Tinggi",
    "Klaster 4: Capaian Rendah, Skala Sekolah Terbatas": "Capaian Rendah & Skala Sekolah Terbatas",
}

df_wilayah["warna_klaster"] = df_wilayah["klaster_k4"].map(PALET_KLASTER)

fig, ax, panel = siapkan_kanvas(
    "Peta Tematik Klasterisasi Wilayah Pendidikan SMP Jawa Barat",
    "Hasil K-Means Clustering (k=4) Berdasarkan Mutu Akademik, Beban Guru, Sarana Sekolah & Kemiskinan"
)
gambar_poligon(ax, df_wilayah, "warna_klaster")
atur_sumbu(ax, total_bounds)
gambar_label(ax, df_wilayah, lambda r: r["label"], total_bounds)

y = 0.985
panel.text(0, y, "Tipologi Klaster Wilayah", fontsize=12, fontweight="bold", va="top")
y -= 0.05
for k_nama, warna in PALET_KLASTER.items():
    grup = df_wilayah[df_wilayah["klaster_k4"] == k_nama].sort_values("tka_rerata_mtk", ascending=False)
    panel.add_patch(Rectangle((0, y - 0.026), 0.045, 0.026, fc=warna, ec="black", lw=0.6))
    panel.text(0.06, y, f"{k_nama.split(':')[0]} ({len(grup)} wilayah)", fontsize=9.5, fontweight="bold", va="top")
    y -= 0.024
    panel.text(0.06, y, KET_KLASTER[k_nama], fontsize=8.5, color="0.25", va="top", style="italic")
    y -= 0.025
    for _, r in grup.iterrows():
        panel.text(0.06, y, f"- {r['label']}", fontsize=8, va="top")
        panel.text(0.60, y, f"TKA: {fmt(r['tka_rerata_mtk'], 1)} | Miskin: {fmt(r['pct_penduduk_miskin'], 1)}%",
                   fontsize=7.5, color="0.3", va="top")
        y -= 0.019
    y -= 0.02

peta_klaster_path = OUTPUT_FIGURES / "peta_klaster_k4.png"
fig.savefig(peta_klaster_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[OK] Peta tematik klaster tersimpan di:\n     -> {peta_klaster_path}")

# --- 6. Cetak Peta 2: Peta Prioritas Intervensi (IPI) ---
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
    "Indeks Prioritas Intervensi: 50% Defisit TKA, 25% Kemiskinan, 12,5% Sekolah Kecil, 12,5% Beban Guru"
)
gambar_poligon(ax, df_wilayah, "warna_prioritas")

for _, r in df_wilayah[df_wilayah["robust"]].iterrows():
    for ring in r["rings_utm"]:
        ax.plot(ring[:, 0], ring[:, 1], color="black", linewidth=2.2, zorder=5)

atur_sumbu(ax, total_bounds)
gambar_label(ax, df_wilayah, lambda r: f"{r['label']}\n#{int(r['peringkat_prioritas'])}" + (" ★" if r["robust"] else ""), total_bounds)

y = 0.985
panel.text(0, y, "Kategori Prioritas (Peringkat & Skor IPI)", fontsize=12, fontweight="bold", va="top")
y -= 0.05
for kat, warna in PALET_PRIORITAS.items():
    grup = df_wilayah[df_wilayah["kategori_prioritas"] == kat].sort_values("peringkat_prioritas")
    panel.add_patch(Rectangle((0, y - 0.026), 0.045, 0.026, fc=warna, ec="black", lw=0.6))
    panel.text(0.06, y, f"{KET_PRIORITAS[kat]} - {len(grup)} wilayah", fontsize=9.5, fontweight="bold", va="top")
    y -= 0.03
    for _, r in grup.iterrows():
        tanda = " ★" if r["robust"] else ""
        panel.text(0.06, y, f"{int(r['peringkat_prioritas']):>2}. {r['label']}{tanda}", fontsize=8, va="top")
        panel.text(0.62, y, fmt(r["skor_ipi"], 1), fontsize=8, va="top", ha="right")
        y -= 0.02
    y -= 0.02

panel.add_patch(Rectangle((0, 0.108), 0.045, 0.024, fc="white", ec="black", lw=2.2))
panel.text(0.06, 0.132, "Garis tebal / ★: Masuk 8 besar pada >= 80%\nsimulasi variasi bobot (5.000 iterasi)", fontsize=8, va="top")

peta_prioritas_path = OUTPUT_FIGURES / "peta_prioritas_intervensi.png"
fig.savefig(peta_prioritas_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[OK] Peta prioritas intervensi tersimpan di:\n     -> {peta_prioritas_path}")

# --- 7. Cetak Peta 3: Komparasi 2-Panel Spasial ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 17), dpi=200)

# Subplot 1
gambar_poligon(ax1, df_wilayah, "warna_klaster")
atur_sumbu(ax1, total_bounds)
gambar_label(ax1, df_wilayah, lambda r: r["label"], total_bounds)
ax1.set_title("A. Tipologi Klasterisasi Wilayah (K-Means, k=4)", fontsize=13, fontweight="bold", pad=10)

# Subplot 2
gambar_poligon(ax2, df_wilayah, "warna_prioritas")
for _, r in df_wilayah[df_wilayah["robust"]].iterrows():
    for ring in r["rings_utm"]:
        ax2.plot(ring[:, 0], ring[:, 1], color="black", linewidth=2.0, zorder=5)
atur_sumbu(ax2, total_bounds)
gambar_label(ax2, df_wilayah, lambda r: f"{r['label']} (#{int(r['peringkat_prioritas'])})", total_bounds)
ax2.set_title("B. Peta Prioritas Intervensi (IPI 1 s/d 27)", fontsize=13, fontweight="bold", pad=10)

plt.suptitle("Analisis Geospasial Pendidikan SMP Jawa Barat: Tipologi Klaster vs Urgensi Intervensi",
             fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout()
peta_komparatif_path = OUTPUT_FIGURES / "peta_spasial_klaster_dan_prioritas.png"
fig.savefig(peta_komparatif_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[OK] Peta komparatif 2-panel tersimpan di:\n     -> {peta_komparatif_path}")

# --- 8. Generate Peta Web Interaktif Leaflet.js (HTML) ---
geojson_str = json.dumps(geojson_data)

html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Peta Interaktif Klasterisasi & Prioritas Pendidikan SMP Jawa Barat</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      display: flex;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
      background: #0f172a;
      color: #1e293b;
    }}
    #sidebar {{
      width: 440px;
      min-width: 400px;
      height: 100%;
      background: #ffffff;
      box-shadow: 4px 0 24px rgba(0,0,0,0.12);
      z-index: 1000;
      display: flex;
      flex-direction: column;
      overflow-y: auto;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    #map-container {{ flex: 1; height: 100%; position: relative; }}
    #map {{ width: 100%; height: 100%; }}
    .sidebar-header {{
      padding: 24px 28px 20px;
      background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
      color: #ffffff;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .badge-sub {{
      display: inline-block;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      background: rgba(37, 99, 235, 0.3);
      color: #60a5fa;
      padding: 4px 10px;
      border-radius: 999px;
      margin-bottom: 10px;
      border: 1px solid rgba(96, 165, 250, 0.3);
    }}
    .sidebar-header h1 {{ font-size: 20px; font-weight: 800; line-height: 1.3; margin-bottom: 6px; }}
    .sidebar-header p {{ font-size: 13px; color: #94a3b8; line-height: 1.5; }}
    .mode-switch-container {{ padding: 16px 28px; background: #f8fafc; border-bottom: 1px solid #e2e8f0; }}
    .mode-switch-title {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; margin-bottom: 8px; }}
    .btn-group {{ display: flex; background: #e2e8f0; border-radius: 10px; padding: 3px; gap: 4px; }}
    .btn-toggle {{
      flex: 1;
      padding: 8px 12px;
      border: none;
      background: transparent;
      font-size: 12.5px;
      font-weight: 600;
      color: #475569;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-toggle.active {{ background: #2563eb; color: #ffffff; box-shadow: 0 2px 6px rgba(37, 99, 235, 0.3); }}
    .filter-section {{ padding: 12px 28px 16px; border-bottom: 1px solid #f1f5f9; }}
    .select-region {{
      width: 100%;
      padding: 10px 14px;
      border: 1.5px solid #cbd5e1;
      border-radius: 8px;
      font-size: 13px;
      font-family: inherit;
      color: #1e293b;
      outline: none;
      background: #ffffff;
      cursor: pointer;
    }}
    .content-section {{ padding: 20px 28px; flex: 1; }}
    .card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px; }}
    .card-title {{ font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 12px; }}
    .empty-state {{ text-align: center; padding: 40px 20px; color: #94a3b8; font-size: 13.5px; }}
    .region-detail-header {{ margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1.5px solid #e2e8f0; }}
    .region-title {{ font-size: 20px; font-weight: 800; color: #0f172a; margin-bottom: 6px; }}
    .badges-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; }}
    .pill-badge {{ font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 6px; color: #ffffff; }}
    .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .metric-box {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 12px; }}
    .metric-box .label {{ font-size: 11px; color: #64748b; font-weight: 600; margin-bottom: 4px; }}
    .metric-box .val {{ font-size: 18px; font-weight: 800; color: #0f172a; }}
    .legend-control {{
      background: rgba(255, 255, 255, 0.96);
      padding: 14px 16px;
      border-radius: 10px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      font-size: 12px;
      line-height: 1.6;
      border: 1px solid #e2e8f0;
      backdrop-filter: blur(4px);
      max-width: 270px;
    }}
    .legend-title {{ font-weight: 800; font-size: 12px; margin-bottom: 8px; text-transform: uppercase; color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }}
    .legend-color {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.15); flex-shrink: 0; }}
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="sidebar-header">
      <span class="badge-sub">Jawa Barat &bull; Jenjang SMP</span>
      <h1>Klasterisasi & Urgensi Intervensi Pendidikan</h1>
      <p>Sistem Informasi Geospasial Pemetaan 27 Kabupaten/Kota Berdasarkan Mutu Akademik, Rasio Sarpras, & Sosio-Ekonomi</p>
    </div>
    <div class="mode-switch-container">
      <div class="mode-switch-title">Layer Analisis</div>
      <div class="btn-group">
        <button id="btn-klaster" class="btn-toggle active" onclick="switchMode('klaster')">Tipologi Klaster (k=4)</button>
        <button id="btn-prioritas" class="btn-toggle" onclick="switchMode('prioritas')">Prioritas Intervensi (IPI)</button>
      </div>
    </div>
    <div class="filter-section">
      <select id="region-dropdown" class="select-region" onchange="focusRegion(this.value)">
        <option value="">-- Pilih Kabupaten / Kota --</option>
      </select>
    </div>
    <div class="content-section" id="detail-panel">
      <div class="empty-state">
        <div style="font-size: 32px; margin-bottom: 12px;">🗺️</div>
        Arahkan kursor atau klik wilayah pada peta untuk melihat profil lengkap indikator pendidikan.
      </div>
    </div>
  </div>
  <div id="map-container">
    <div id="map"></div>
  </div>

  <script>
    const geojsonData = {geojson_str};
    const map = L.map('map', {{ zoomControl: false }}).setView([-6.9, 107.6], 9);
    L.control.zoom({{ position: 'topright' }}).addTo(map);

    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; CartoDB & Pusmendik / Kemendikbud',
      maxZoom: 14, minZoom: 7
    }}).addTo(map);

    let currentMode = 'klaster';
    let geojsonLayer;
    let selectedLayer = null;
    const layerMap = {{}};

    function getKlasterColor(nama) {{
      if (!nama) return '#94a3b8';
      if (nama.includes('Klaster 1')) return '#2563eb';
      if (nama.includes('Klaster 2')) return '#0ea5e9';
      if (nama.includes('Klaster 3')) return '#f59e0b';
      if (nama.includes('Klaster 4')) return '#dc2626';
      return '#64748b';
    }}

    function getPrioritasColor(kat) {{
      if (!kat) return '#94a3b8';
      if (kat.includes('Prioritas 1')) return '#b91c1c';
      if (kat.includes('Prioritas 2')) return '#ea580c';
      if (kat.includes('Prioritas 3')) return '#d97706';
      if (kat.includes('Prioritas 4')) return '#15803d';
      return '#64748b';
    }}

    function styleFeature(feature) {{
      const props = feature.properties;
      const fill = (currentMode === 'klaster') ? getKlasterColor(props.klaster_k4) : getPrioritasColor(props.kategori_prioritas);
      return {{
        fillColor: fill,
        weight: 1.5,
        opacity: 1,
        color: '#ffffff',
        fillOpacity: 0.82
      }};
    }}

    function renderLayer() {{
      if (geojsonLayer) map.removeLayer(geojsonLayer);
      geojsonLayer = L.geoJSON(geojsonData, {{
        style: styleFeature,
        onEachFeature: (feature, layer) => {{
          const p = feature.properties;
          const key = p.kode_wilayah || p.nama_kabkota;
          layerMap[key] = layer;

          layer.on({{
            mouseover: (e) => {{
              if (selectedLayer !== layer) {{
                e.target.setStyle({{ weight: 2.8, color: '#0f172a', fillOpacity: 0.95 }});
                e.target.bringToFront();
              }}
            }},
            mouseout: (e) => {{
              if (selectedLayer !== layer) geojsonLayer.resetStyle(e.target);
            }},
            click: (e) => {{
              if (selectedLayer) geojsonLayer.resetStyle(selectedLayer);
              selectedLayer = layer;
              layer.setStyle({{ weight: 3, color: '#0f172a', fillOpacity: 1 }});
              updateDetail(p);
              document.getElementById('region-dropdown').value = key;
              map.fitBounds(layer.getBounds(), {{ maxZoom: 10, padding: [40, 40] }});
            }}
          }});

          const tipText = `<strong>${{p.nama_kabkota}}</strong><br>` +
            (currentMode === 'klaster' ? (p.klaster_k4 || 'N/A') : `${{p.kategori_prioritas || ''}} (Skor IPI: ${{p.skor_ipi?.toFixed(1) || '-'}})`);
          layer.bindTooltip(tipText, {{ sticky: true, className: 'custom-tooltip' }});
        }}
      }}).addTo(map);
      updateLegend();
    }}

    function updateDetail(props) {{
      const panel = document.getElementById('detail-panel');
      if (!props) {{
        panel.innerHTML = `<div class="empty-state">Pilih wilayah pada peta untuk melihat profil lengkap.</div>`;
        return;
      }}

      const klasterColor = getKlasterColor(props.klaster_k4);
      const prioritasColor = getPrioritasColor(props.kategori_prioritas);

      panel.innerHTML = `
        <div class="region-detail-header">
          <div class="region-title">${{props.nama_kabkota}}</div>
          <div class="badges-row">
            <span class="pill-badge" style="background:${{klasterColor}}">${{props.klaster_k4 ? props.klaster_k4.split(':')[0] : 'Klaster'}}</span>
            <span class="pill-badge" style="background:${{prioritasColor}}">Peringkat #${{props.peringkat_ipi || props.peringkat_prioritas || '-'}} (${{props.kategori_prioritas ? props.kategori_prioritas.split(' ')[0] + ' ' + props.kategori_prioritas.split(' ')[1] : ''}})</span>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Skor Indeks Prioritas Intervensi</div>
          <div style="display:flex; align-items:baseline; gap:8px;">
            <span style="font-size:32px; font-weight:800; color:${{prioritasColor}}">${{props.skor_ipi ? props.skor_ipi.toFixed(1) : '-'}}</span>
            <span style="font-size:12px; color:#64748b;">/ 100 (${{props.kategori_prioritas || '-'}})</span>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Indikator 4 Pilar Pendidikan SMP</div>
          <div class="metric-grid">
            <div class="metric-box">
              <div class="label">Rerata TKA MTK</div>
              <div class="val">${{props.tka_rerata_mtk ? props.tka_rerata_mtk.toFixed(2) : '-'}}</div>
            </div>
            <div class="metric-box">
              <div class="label">Penduduk Miskin</div>
              <div class="val">${{props.pct_penduduk_miskin ? props.pct_penduduk_miskin.toFixed(2) + '%' : '-'}}</div>
            </div>
            <div class="metric-box">
              <div class="label">Rasio Murid/Guru</div>
              <div class="val">${{props.rasio_murid_guru ? props.rasio_murid_guru.toFixed(1) : '-'}}</div>
            </div>
            <div class="metric-box">
              <div class="label">Rasio Murid/Sekolah</div>
              <div class="val">${{props.rasio_murid_sekolah ? props.rasio_murid_sekolah.toFixed(0) : '-'}}</div>
            </div>
          </div>
        </div>
      `;
    }}

    const legendCtrl = L.control({{ position: 'bottomright' }});
    legendCtrl.onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend-control');
      div.id = 'legend-content';
      return div;
    }};
    legendCtrl.addTo(map);

    function updateLegend() {{
      const leg = document.getElementById('legend-content');
      if (!leg) return;
      if (currentMode === 'klaster') {{
        leg.innerHTML = `
          <div class="legend-title">Tipologi Klaster (k=4)</div>
          <div class="legend-row"><div class="legend-color" style="background:#2563eb"></div>Klaster 1: Perkotaan Capaian Tinggi</div>
          <div class="legend-row"><div class="legend-color" style="background:#0ea5e9"></div>Klaster 2: Beban Guru Tinggi</div>
          <div class="legend-row"><div class="legend-color" style="background:#f59e0b"></div>Klaster 3: Kemiskinan Tinggi</div>
          <div class="legend-row"><div class="legend-color" style="background:#dc2626"></div>Klaster 4: Capaian Rendah & Terbatas</div>
        `;
      }} else {{
        leg.innerHTML = `
          <div class="legend-title">Prioritas Intervensi</div>
          <div class="legend-row"><div class="legend-color" style="background:#b91c1c"></div>Prioritas 1: Mendesak (&ge; 70)</div>
          <div class="legend-row"><div class="legend-color" style="background:#ea580c"></div>Prioritas 2: Tinggi (60 &ndash; 69.9)</div>
          <div class="legend-row"><div class="legend-color" style="background:#d97706"></div>Prioritas 3: Menengah (40 &ndash; 59.9)</div>
          <div class="legend-row"><div class="legend-color" style="background:#15803d"></div>Prioritas 4: Mandiri (&lt; 40)</div>
        `;
      }}
    }}

    function switchMode(mode) {{
      currentMode = mode;
      document.getElementById('btn-klaster').classList.toggle('active', mode === 'klaster');
      document.getElementById('btn-prioritas').classList.toggle('active', mode === 'prioritas');
      renderLayer();
      if (selectedLayer) updateDetail(selectedLayer.feature.properties);
    }}

    function focusRegion(key) {{
      if (!key || !layerMap[key]) {{
        if (selectedLayer) geojsonLayer.resetStyle(selectedLayer);
        selectedLayer = null;
        updateDetail(null);
        map.fitBounds(geojsonLayer.getBounds(), {{ padding: [30, 30] }});
        return;
      }}
      if (selectedLayer) geojsonLayer.resetStyle(selectedLayer);
      selectedLayer = layerMap[key];
      selectedLayer.setStyle({{ weight: 3, color: '#0f172a', fillOpacity: 0.95 }});
      map.fitBounds(selectedLayer.getBounds(), {{ maxZoom: 10, padding: [60, 60] }});
      updateDetail(selectedLayer.feature.properties);
    }}

    function initDropdown() {{
      const select = document.getElementById('region-dropdown');
      const features = [...geojsonData.features].sort((a, b) => 
        a.properties.nama_kabkota.localeCompare(b.properties.nama_kabkota)
      );
      features.forEach(f => {{
        const opt = document.createElement('option');
        const key = f.properties.kode_wilayah || f.properties.nama_kabkota;
        opt.value = key;
        const rank = f.properties.peringkat_ipi || f.properties.peringkat_prioritas || '-';
        opt.textContent = `${{f.properties.nama_kabkota}} (#${{rank}})`;
        select.appendChild(opt);
      }});
    }}

    renderLayer();
    initDropdown();
    map.fitBounds(geojsonLayer.getBounds(), {{ padding: [30, 30] }});
  </script>
</body>
</html>
"""

with open(HTML_OUT, "w", encoding="utf-8") as f:
    f.write(html_content)

with open(DOCS_OUT, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"[OK] Peta Web GIS Interaktif tersimpan di:\n     -> {HTML_OUT}\n     -> {DOCS_OUT}")
print("Selesai Tahap 07.\n")
