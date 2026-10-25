import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# 1. Konfigurasi direktori
BASE_DIR = Path(__file__).resolve().parent.parent
GEOJSON_IN = BASE_DIR / "dataset" / "kabkota_jabar_keyed.geojson"
CSV_IPI = BASE_DIR / "dataset" / "indeks_prioritas_intervensi_smp.csv"
OUTPUT_DIR = BASE_DIR / "clustering" / "statistik_grafik"
GRAFIK_DIR = OUTPUT_DIR / "grafik_eda"
GEOJSON_OUT_DIR = BASE_DIR / "dataset" / "geospasial"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)
GEOJSON_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Baca data tabel Indeks Prioritas & Klaster
df_ipi = pd.read_csv(CSV_IPI)

# Standardisasi kode_wilayah dan nama_kabkota untuk pencocokan kunci
df_ipi["kode_wilayah_std"] = df_ipi["kode_wilayah"].apply(lambda x: f"{float(x):.2f}")

ipi_lookup = {}
for _, row in df_ipi.iterrows():
    # Lookup berbasis kode_wilayah dan nama_kabkota
    ipi_lookup[row["kode_wilayah_std"]] = row.to_dict()
    ipi_lookup[row["nama_kabkota"]] = row.to_dict()

# 3. Baca GeoJSON Poligon Jawa Barat
with open(GEOJSON_IN, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

matched_count = 0
for feature in geojson_data["features"]:
    props = feature["properties"]
    bps_key = str(props.get("kode_bps", ""))
    wil_key = str(props.get("kode_wilayah", ""))
    nama_key = str(props.get("nama_kabkota", ""))

    matched_data = (
        ipi_lookup.get(bps_key)
        or ipi_lookup.get(wil_key)
        or ipi_lookup.get(nama_key)
    )

    if matched_data:
        matched_count += 1
        props["nama_pendek"] = matched_data["nama_pendek"]
        props["klaster_k4"] = matched_data.get("nama_klaster_k4", "")
        props["skor_ipi"] = float(matched_data["skor_ipi"])
        props["peringkat_ipi"] = int(matched_data["peringkat"])
        props["kategori_prioritas"] = matched_data["kategori_prioritas"]
        props["tka_rerata_mtk"] = float(matched_data["tka_rerata_mtk"])
        props["pct_penduduk_miskin"] = float(matched_data["pct_penduduk_miskin"])
        props["rasio_murid_guru"] = float(matched_data["rasio_murid_guru"])
        props["rasio_murid_sekolah"] = float(matched_data["rasio_murid_sekolah"])

print(f"Berhasil menggabungkan data klaster & IPI: {matched_count} dari {len(geojson_data['features'])} wilayah poligon.")

# 4. Simpan GeoJSON Gabungan
geojson_out_path = GEOJSON_OUT_DIR / "kabkota_jabar_klaster_dan_indeks.geojson"
with open(geojson_out_path, "w", encoding="utf-8") as f:
    json.dump(geojson_data, f, ensure_ascii=False, indent=2)

print(f"File GeoJSON hasil penggabungan tersimpan di:\n-> {geojson_out_path}")

# 5. Visualisasi Peta Tematik (Choropleth Map) Resolusi Tinggi
# Hitung titik tengah poligon untuk penempatan label
def get_polygon_center(coordinates):
    all_points = []
    for poly in coordinates:
        for pt in poly[0]:
            all_points.append(pt)
    arr = np.array(all_points)
    return arr[:, 0].mean(), arr[:, 1].mean()

# Definisi palet warna
warna_klaster_map = {
    "Klaster 1: Perkotaan, Capaian Tertinggi": "#2563eb",         # Biru (Perkotaan Unggul)
    "Klaster 2: Beban Kelas Tinggi, Capaian Menengah": "#0ea5e9", # Cyan (Beban Murid/Guru Tinggi)
    "Klaster 3: Kemiskinan Tertinggi, Capaian Menengah": "#f59e0b", # Amber/Oranye (Kemiskinan Tinggi)
    "Klaster 4: Capaian Terendah, Sekolah Kecil": "#dc2626",      # Merah (TKA Rendah & Sekolah Kecil)
}

warna_prioritas_map = {
    "Prioritas 1 (Mendesak / Urgent)": "#dc2626",     # Merah tua
    "Prioritas 2 (Tinggi / High)": "#f97316",         # Oranye
    "Prioritas 3 (Menengah / Moderate)": "#eab308",     # Kuning
    "Prioritas 4 (Mandiri / Low)": "#16a34a"          # Hijau
}

# --- PETA 2-PANEL (KLASTERISASI + INDEKS PRIORITAS) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))

# Subplot 1: Peta Klasterisasi K-Means
patches_k1 = []
colors_k1 = []

for feature in geojson_data["features"]:
    geom = feature["geometry"]
    klaster_nama = feature["properties"].get("klaster_k4", "")
    color = warna_klaster_map.get(klaster_nama, "#94a3b8")
    
    for poly in geom["coordinates"]:
        ring = poly[0]
        patches_k1.append(Polygon(ring, closed=True))
        colors_k1.append(color)

    # Tampilkan label nama daerah
    cx, cy = get_polygon_center(geom["coordinates"])
    nama_pendek = feature["properties"].get("nama_pendek", "")
    ax1.text(cx, cy, nama_pendek, fontsize=7.5, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"))

pc1 = PatchCollection(patches_k1, facecolors=colors_k1, edgecolors="#1e293b", linewidths=0.7)
ax1.add_collection(pc1)
ax1.autoscale()
ax1.set_aspect("equal")
ax1.set_title("A. Peta Persebaran Klaster Pendidikan SMP Jawa Barat (K-Means, k=4)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("Bujur / Longitude (°E)", fontsize=9)
ax1.set_ylabel("Lintang / Latitude (°S)", fontsize=9)
ax1.grid(True, linestyle=":", alpha=0.4)

from matplotlib.patches import Patch
leg_k1 = [
    Patch(facecolor="#2563eb", edgecolor="black", label="Klaster 1: Perkotaan, Capaian Tertinggi"),
    Patch(facecolor="#0ea5e9", edgecolor="black", label="Klaster 2: Beban Kelas Tinggi, Capaian Menengah"),
    Patch(facecolor="#f59e0b", edgecolor="black", label="Klaster 3: Kemiskinan Tertinggi, Capaian Menengah"),
    Patch(facecolor="#dc2626", edgecolor="black", label="Klaster 4: Capaian Terendah, Sekolah Kecil")
]
ax1.legend(handles=leg_k1, loc="lower left", fontsize=9, framealpha=0.95)

# Subplot 2: Peta Indeks Prioritas Intervensi (IPI)
patches_k2 = []
colors_k2 = []

for feature in geojson_data["features"]:
    geom = feature["geometry"]
    kat_prioritas = feature["properties"].get("kategori_prioritas", "")
    color = warna_prioritas_map.get(kat_prioritas, "#94a3b8")
    
    for poly in geom["coordinates"]:
        ring = poly[0]
        patches_k2.append(Polygon(ring, closed=True))
        colors_k2.append(color)

    # Tampilkan label nama dan skor IPI
    cx, cy = get_polygon_center(geom["coordinates"])
    nama_pendek = feature["properties"].get("nama_pendek", "")
    skor_val = feature["properties"].get("skor_ipi", 0.0)
    label_text = f"{nama_pendek}\n({skor_val:.1f})"
    ax2.text(cx, cy, label_text, fontsize=7, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"))

pc2 = PatchCollection(patches_k2, facecolors=colors_k2, edgecolors="#1e293b", linewidths=0.7)
ax2.add_collection(pc2)
ax2.autoscale()
ax2.set_aspect("equal")
ax2.set_title("B. Peta Peringkat Indeks Prioritas Intervensi (IPI) SMP Jawa Barat (Skor 0 - 100)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Bujur / Longitude (°E)", fontsize=9)
ax2.set_ylabel("Lintang / Latitude (°S)", fontsize=9)
ax2.grid(True, linestyle=":", alpha=0.4)

leg_k2 = [
    Patch(facecolor="#dc2626", edgecolor="black", label="Prioritas 1: Mendesak (Skor ≥ 70)"),
    Patch(facecolor="#f97316", edgecolor="black", label="Prioritas 2: Tinggi (Skor 60 - 69.9)"),
    Patch(facecolor="#eab308", edgecolor="black", label="Prioritas 3: Menengah (Skor 40 - 59.9)"),
    Patch(facecolor="#16a34a", edgecolor="black", label="Prioritas 4: Mandiri (Skor < 40)")
]
ax2.legend(handles=leg_k2, loc="lower left", fontsize=9.5, framealpha=0.95)

plt.suptitle("Analisis Spasial Klasterisasi & Prioritas Intervensi Pendidikan SMP Jawa Barat", fontsize=15, fontweight="bold", y=0.995)
plt.tight_layout()

peta_png_path = GRAFIK_DIR / "peta_spasial_klaster_dan_prioritas.png"
plt.savefig(peta_png_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Peta tematik spasial PNG (gabungan) berhasil disimpan di:\n-> {peta_png_path}")

# Simpan juga peta Klaster K-Means secara terpisah
fig_k1, ax_ind1 = plt.subplots(figsize=(15, 9))
pc1_ind = PatchCollection(patches_k1, facecolors=colors_k1, edgecolors="#1e293b", linewidths=0.7)
ax_ind1.add_collection(pc1_ind)
ax_ind1.autoscale()
ax_ind1.set_aspect("equal")
ax_ind1.set_title("Peta Persebaran Klaster Pendidikan SMP Jawa Barat (K-Means, k=4)", fontsize=14, fontweight="bold", pad=12)
ax_ind1.set_xlabel("Bujur / Longitude (°E)", fontsize=10)
ax_ind1.set_ylabel("Lintang / Latitude (°S)", fontsize=10)
ax_ind1.grid(True, linestyle=":", alpha=0.4)
for feature in geojson_data["features"]:
    cx, cy = get_polygon_center(feature["geometry"]["coordinates"])
    ax_ind1.text(cx, cy, feature["properties"].get("nama_pendek", ""), fontsize=8, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"))
ax_ind1.legend(handles=leg_k1, loc="lower left", fontsize=10, framealpha=0.95)
plt.tight_layout()
peta_k1_path = GRAFIK_DIR / "peta_klaster_kmeans.png"
plt.savefig(peta_k1_path, dpi=200, bbox_inches="tight")
plt.close()

# Simpan juga peta Indeks Prioritas secara terpisah
fig_k2, ax_ind2 = plt.subplots(figsize=(15, 9))
pc2_ind = PatchCollection(patches_k2, facecolors=colors_k2, edgecolors="#1e293b", linewidths=0.7)
ax_ind2.add_collection(pc2_ind)
ax_ind2.autoscale()
ax_ind2.set_aspect("equal")
ax_ind2.set_title("Peta Peringkat Indeks Prioritas Intervensi (IPI) SMP Jawa Barat (Skor 0 - 100)", fontsize=14, fontweight="bold", pad=12)
ax_ind2.set_xlabel("Bujur / Longitude (°E)", fontsize=10)
ax_ind2.set_ylabel("Lintang / Latitude (°S)", fontsize=10)
ax_ind2.grid(True, linestyle=":", alpha=0.4)
for feature in geojson_data["features"]:
    cx, cy = get_polygon_center(feature["geometry"]["coordinates"])
    nama_pendek = feature["properties"].get("nama_pendek", "")
    skor_val = feature["properties"].get("skor_ipi", 0.0)
    ax_ind2.text(cx, cy, f"{nama_pendek}\n({skor_val:.1f})", fontsize=7.5, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"))
ax_ind2.legend(handles=leg_k2, loc="lower left", fontsize=10, framealpha=0.95)
plt.tight_layout()
peta_k2_path = GRAFIK_DIR / "peta_prioritas_intervensi.png"
plt.savefig(peta_k2_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Peta individu klaster & prioritas berhasil disimpan:\n-> {peta_k1_path}\n-> {peta_k2_path}")

# 6. Pembuatan Peta Web Interaktif Mandiri (Leaflet.js HTML)
geojson_str = json.dumps(geojson_data)

html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Peta Interaktif Klaster & Prioritas Intervensi SMP Jawa Barat</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
    #map {{ height: 100vh; width: 100%; }}
    .header-box {{
      position: absolute; top: 15px; left: 60px; z-index: 1000;
      background: rgba(255, 255, 255, 0.95); padding: 12px 20px;
      border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);
      max-width: 450px;
    }}
    .header-box h2 {{ margin: 0 0 5px 0; font-size: 16px; color: #1e293b; }}
    .header-box p {{ margin: 0; font-size: 12px; color: #64748b; line-height: 1.4; }}
    .legend-box {{
      position: absolute; bottom: 25px; right: 20px; z-index: 1000;
      background: rgba(255, 255, 255, 0.95); padding: 14px 18px;
      border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);
      font-size: 12px;
    }}
    .legend-item {{ display: flex; align-items: center; margin-bottom: 6px; }}
    .legend-color {{ width: 18px; height: 18px; border-radius: 4px; margin-right: 8px; border: 1px solid #333; }}
    .mode-switch {{
      margin-top: 8px; padding-top: 8px; border-top: 1px solid #e2e8f0;
      display: flex; gap: 10px;
    }}
    .btn {{
      padding: 5px 12px; border-radius: 6px; font-size: 11px; font-weight: bold;
      border: 1px solid #cbd5e1; cursor: pointer; background: white; color: #334155;
    }}
    .btn.active {{ background: #2563eb; color: white; border-color: #2563eb; }}
    .custom-popup table {{ width: 100%; font-size: 12px; border-collapse: collapse; margin-top: 6px; }}
    .custom-popup td {{ padding: 3px 6px; border-bottom: 1px solid #f1f5f9; }}
    .custom-popup td:first-child {{ font-weight: bold; color: #475569; }}
  </style>
</head>
<body>
  <div class="header-box">
    <h2>Sistem Spasial Klasterisasi & Prioritas Pendidikan Jawa Barat</h2>
    <p>Klik poligon kabupaten/kota untuk melihat rincian capaian TKA, kemiskinan, beban guru, dan indeks prioritas.</p>
    <div class="mode-switch">
      <button id="btn-klaster" class="btn active" onclick="switchMode('klaster')">Mode Klaster K-Means</button>
      <button id="btn-prioritas" class="btn" onclick="switchMode('prioritas')">Mode Indeks Prioritas (IPI)</button>
    </div>
  </div>

  <div id="legend" class="legend-box"></div>
  <div id="map"></div>

  <script>
    const geojsonData = {geojson_str};

    const map = L.map('map').setView([-6.9175, 107.6191], 8);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    let currentMode = 'klaster';
    let geojsonLayer;

    const klasterColors = {{
      "Klaster 1: Perkotaan, Capaian Tertinggi": "#2563eb",
      "Klaster 2: Beban Kelas Tinggi, Capaian Menengah": "#0ea5e9",
      "Klaster 3: Kemiskinan Tertinggi, Capaian Menengah": "#f59e0b",
      "Klaster 4: Capaian Terendah, Sekolah Kecil": "#dc2626"
    }};

    const prioritasColors = {{
      "Prioritas 1 (Mendesak / Urgent)": "#dc2626",
      "Prioritas 2 (Tinggi / High)": "#f97316",
      "Prioritas 3 (Menengah / Moderate)": "#eab308",
      "Prioritas 4 (Mandiri / Low)": "#16a34a"
    }};

    function getColor(props) {{
      if (currentMode === 'klaster') {{
        return klasterColors[props.klaster_k4] || '#94a3b8';
      }} else {{
        return prioritasColors[props.kategori_prioritas] || '#94a3b8';
      }}
    }}

    function style(feature) {{
      return {{
        fillColor: getColor(feature.properties),
        weight: 1.5,
        opacity: 1,
        color: '#1e293b',
        dashArray: '2',
        fillOpacity: 0.75
      }};
    }}

    function onEachFeature(feature, layer) {{
      const p = feature.properties;
      const popupContent = `
        <div class="custom-popup">
          <h3 style="margin:0 0 4px 0; color:#1e293b; font-size:14px;">${{p.nama_kabkota}}</h3>
          <div style="font-size:11px; color:#64748b; margin-bottom:6px;">Status: <b>${{p.klaster_k4}}</b></div>
          <table>
            <tr><td>Peringkat IPI</td><td><b>#${{p.peringkat_ipi}} dari 27</b></td></tr>
            <tr><td>Skor Prioritas (IPI)</td><td><b>${{p.skor_ipi}}</b> (${{p.kategori_prioritas}})</td></tr>
            <tr><td>Rerata TKA MTK</td><td>${{p.tka_rerata_mtk}}</td></tr>
            <tr><td>Penduduk Miskin</td><td>${{p.pct_penduduk_miskin}}%</td></tr>
            <tr><td>Rasio Murid/Guru</td><td>${{p.rasio_murid_guru}}</td></tr>
            <tr><td>Rasio Murid/Sekolah</td><td>${{p.rasio_murid_sekolah}}</td></tr>
          </table>
        </div>
      `;
      layer.bindPopup(popupContent);
      layer.on({{
        mouseover: function(e) {{
          const l = e.target;
          l.setStyle({{ weight: 3, color: '#000', dashArray: '', fillOpacity: 0.9 }});
        }},
        mouseout: function(e) {{
          geojsonLayer.resetStyle(e.target);
        }}
      }});
    }}

    function renderLayer() {{
      if (geojsonLayer) map.removeLayer(geojsonLayer);
      geojsonLayer = L.geoJson(geojsonData, {{
        style: style,
        onEachFeature: onEachFeature
      }}).addTo(map);
      updateLegend();
    }}

    function updateLegend() {{
      const leg = document.getElementById('legend');
      if (currentMode === 'klaster') {{
        leg.innerHTML = `
          <b>Legenda Klaster K-Means (k=4)</b><br><br>
          <div class="legend-item"><div class="legend-color" style="background:#2563eb"></div>Klaster 1: Perkotaan, Capaian Tertinggi</div>
          <div class="legend-item"><div class="legend-color" style="background:#0ea5e9"></div>Klaster 2: Beban Kelas Tinggi</div>
          <div class="legend-item"><div class="legend-color" style="background:#f59e0b"></div>Klaster 3: Kemiskinan Tertinggi</div>
          <div class="legend-item"><div class="legend-color" style="background:#dc2626"></div>Klaster 4: Capaian Terendah, Sekolah Kecil</div>
        `;
      }} else {{
        leg.innerHTML = `
          <b>Legenda Prioritas Intervensi (IPI)</b><br><br>
          <div class="legend-item"><div class="legend-color" style="background:#dc2626"></div>Prioritas 1: Mendesak (≥ 70)</div>
          <div class="legend-item"><div class="legend-color" style="background:#f97316"></div>Prioritas 2: Tinggi (60 - 69.9)</div>
          <div class="legend-item"><div class="legend-color" style="background:#eab308"></div>Prioritas 3: Menengah (40 - 59.9)</div>
          <div class="legend-item"><div class="legend-color" style="background:#16a34a"></div>Prioritas 4: Mandiri (< 40)</div>
        `;
      }}
    }}

    function switchMode(mode) {{
      currentMode = mode;
      document.getElementById('btn-klaster').classList.toggle('active', mode === 'klaster');
      document.getElementById('btn-prioritas').classList.toggle('active', mode === 'prioritas');
      renderLayer();
    }}

    renderLayer();
  </script>
</body>
</html>
"""

html_out_path = OUTPUT_DIR / "peta_interaktif_jabar.html"
with open(html_out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Peta Web Interaktif HTML berhasil disimpan di:\n-> {html_out_path}")
