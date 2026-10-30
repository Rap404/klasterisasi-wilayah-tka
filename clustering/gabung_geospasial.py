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
    "Klaster 1: Perkotaan, Capaian Tinggi": "#2563eb",         # Biru (Perkotaan Unggul)
    "Klaster 2: Beban Kelas Tinggi, Capaian Menengah": "#0ea5e9", # Cyan (Beban Murid/Guru Tinggi)
    "Klaster 3: Kemiskinan Tinggi, Capaian Menengah": "#f59e0b", # Amber/Oranye (Kemiskinan Tinggi)
    "Klaster 4: Capaian Rendah, Skala Sekolah Terbatas": "#dc2626", # Merah (TKA Rendah & Skala Terbatas)
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
    Patch(facecolor="#2563eb", edgecolor="black", label="Klaster 1: Perkotaan, Capaian Tinggi"),
    Patch(facecolor="#0ea5e9", edgecolor="black", label="Klaster 2: Beban Kelas Tinggi, Capaian Menengah"),
    Patch(facecolor="#f59e0b", edgecolor="black", label="Klaster 3: Kemiskinan Tinggi, Capaian Menengah"),
    Patch(facecolor="#dc2626", edgecolor="black", label="Klaster 4: Capaian Rendah, Skala Sekolah Terbatas")
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

# 6. Pembuatan Peta Web Interaktif Mandiri (Leaflet.js - Desain Minimalis & Bersih)
CARTO_API_KEY = "cb1_4u6s_1_4aee1d00816286e303ed17f1"
geojson_str = json.dumps(geojson_data)

html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Peta Spasial Pendidikan SMP Jawa Barat</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background: #f8fafc;
      color: #0f172a;
      overflow: hidden;
    }}
    #map {{
      height: 100vh;
      width: 100%;
      background: #e2e8f0;
    }}

    /* Sidebar Panel (Minimalist, Crisp) */
    .dashboard-panel {{
      position: absolute;
      top: 14px;
      left: 14px;
      z-index: 1000;
      width: 320px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}

    .panel-header {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .panel-meta {{
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #64748b;
    }}
    .panel-title {{
      font-size: 14px;
      font-weight: 700;
      color: #0f172a;
      line-height: 1.3;
    }}

    /* Segmented Control */
    .segmented-control {{
      display: flex;
      background: #f1f5f9;
      padding: 2px;
      border-radius: 6px;
      gap: 2px;
    }}
    .segment-btn {{
      flex: 1;
      padding: 6px 8px;
      font-size: 11px;
      font-weight: 500;
      border: none;
      border-radius: 4px;
      background: transparent;
      color: #64748b;
      cursor: pointer;
      transition: background 0.15s, color 0.15s;
      text-align: center;
    }}
    .segment-btn.active {{
      background: #ffffff;
      color: #0f172a;
      font-weight: 600;
      box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
    }}

    /* Select Dropdown */
    .field-group {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .field-label {{
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #94a3b8;
    }}
    .region-select {{
      width: 100%;
      height: 32px;
      padding: 0 8px;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      font-size: 12px;
      color: #334155;
      background: #ffffff;
      outline: none;
      cursor: pointer;
    }}
    .region-select:focus {{
      border-color: #94a3b8;
    }}

    /* Real-Time Inspector / Detail Card */
    .detail-card {{
      border-top: 1px solid #f1f5f9;
      padding-top: 10px;
    }}
    .detail-placeholder {{
      font-size: 11px;
      color: #94a3b8;
      font-style: italic;
      padding: 8px 0;
    }}
    .detail-region {{
      font-size: 13px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 2px;
    }}
    .detail-badge {{
      display: inline-block;
      font-size: 10px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 4px;
      margin-bottom: 8px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }}
    .stat-box {{
      background: #f8fafc;
      border: 1px solid #f1f5f9;
      border-radius: 4px;
      padding: 6px 8px;
    }}
    .stat-label {{
      font-size: 10px;
      color: #64748b;
      margin-bottom: 2px;
    }}
    .stat-val {{
      font-size: 12px;
      font-weight: 700;
      color: #0f172a;
      font-feature-settings: "tnum";
    }}

    /* Minimalist Legend (Bottom-Right) */
    .legend-panel {{
      position: absolute;
      bottom: 20px;
      right: 16px;
      z-index: 1000;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
      padding: 10px 12px;
      min-width: 210px;
    }}
    .legend-title {{
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #64748b;
      margin-bottom: 6px;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 11px;
      color: #334155;
      margin-bottom: 4px;
    }}
    .legend-row:last-child {{
      margin-bottom: 0;
    }}
    .legend-color {{
      width: 10px;
      height: 10px;
      border-radius: 2px;
      flex-shrink: 0;
    }}

    /* Tooltip */
    .leaflet-tooltip.clean-tooltip {{
      background: #0f172a;
      color: #ffffff;
      border: none;
      border-radius: 4px;
      padding: 3px 6px;
      font-size: 11px;
      font-weight: 600;
      box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }}
    .leaflet-tooltip.clean-tooltip:before {{
      border-top-color: #0f172a;
    }}

    .leaflet-control-attribution {{
      background: rgba(255, 255, 255, 0.75) !important;
      font-size: 10px !important;
      color: #94a3b8 !important;
    }}
  </style>
</head>
<body>
  <div class="dashboard-panel">
    <div class="panel-header">
      <span class="panel-meta">Pendidikan SMP Jawa Barat</span>
      <h1 class="panel-title">Klasterisasi & Prioritas Wilayah</h1>
    </div>

    <div class="segmented-control">
      <button id="btn-klaster" class="segment-btn active" onclick="switchMode('klaster')">Klaster K-Means</button>
      <button id="btn-prioritas" class="segment-btn" onclick="switchMode('prioritas')">Prioritas Intervensi</button>
    </div>

    <div class="field-group">
      <label class="field-label" for="region-dropdown">Cari Wilayah</label>
      <select id="region-dropdown" class="region-select" onchange="focusRegion(this.value)">
        <option value="">Pilih Kabupaten / Kota</option>
      </select>
    </div>

    <div class="detail-card" id="detail-card">
      <div class="detail-placeholder">Arahkan kursor atau klik wilayah untuk melihat ringkasan indikator.</div>
    </div>
  </div>

  <div id="legend" class="legend-panel"></div>
  <div id="map"></div>

  <script>
    const geojsonData = {geojson_str};
    const cartoKey = "{CARTO_API_KEY}";

    const map = L.map('map', {{
      center: [-6.9175, 107.6191],
      zoom: 8,
      zoomControl: false
    }});

    L.control.zoom({{ position: 'topright' }}).addTo(map);

    L.tileLayer(
      'https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}.png?key=' + cartoKey,
      {{
        subdomains: 'abcd',
        maxZoom: 20,
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
      }}
    ).addTo(map);

    let currentMode = 'klaster';
    let geojsonLayer;
    let selectedLayer = null;
    const layerMap = {{}};

    const klasterColors = {{
      "Klaster 1: Perkotaan, Capaian Tinggi": "#2563eb",
      "Klaster 2: Beban Kelas Tinggi, Capaian Menengah": "#0284c7",
      "Klaster 3: Kemiskinan Tinggi, Capaian Menengah": "#d97706",
      "Klaster 4: Capaian Rendah, Skala Sekolah Terbatas": "#dc2626"
    }};

    const prioritasColors = {{
      "Prioritas 1 (Mendesak / Urgent)": "#b91c1c",
      "Prioritas 2 (Tinggi / High)": "#ea580c",
      "Prioritas 3 (Menengah / Moderate)": "#d97706",
      "Prioritas 4 (Mandiri / Low)": "#15803d"
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
        weight: 1.2,
        opacity: 1,
        color: '#ffffff',
        fillOpacity: 0.75
      }};
    }}

    function updateDetail(props) {{
      const container = document.getElementById('detail-card');
      if (!props) {{
        container.innerHTML = '<div class="detail-placeholder">Arahkan kursor atau klik wilayah untuk melihat ringkasan indikator.</div>';
        return;
      }}
      const color = currentMode === 'klaster' 
        ? (klasterColors[props.klaster_k4] || '#64748b') 
        : (prioritasColors[props.kategori_prioritas] || '#64748b');
      
      const badgeText = currentMode === 'klaster'
        ? props.klaster_k4.split(':')[0]
        : props.kategori_prioritas.split(' ')[0] + ' ' + (props.kategori_prioritas.split(' ')[1] || '');

      container.innerHTML = `
        <div class="detail-region">${{props.nama_kabkota}}</div>
        <div class="detail-badge" style="background:${{color}}15; color:${{color}}; border:1px solid ${{color}}30;">
          ${{badgeText}} &bull; Peringkat #${{props.peringkat_ipi}}
        </div>
        <div class="stats-grid">
          <div class="stat-box">
            <div class="stat-label">Skor IPI</div>
            <div class="stat-val">${{props.skor_ipi}}</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">TKA MTK</div>
            <div class="stat-val">${{props.tka_rerata_mtk}}</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Penduduk Miskin</div>
            <div class="stat-val">${{props.pct_penduduk_miskin}}%</div>
          </div>
          <div class="stat-box">
            <div class="stat-label">Rasio Murid/Guru</div>
            <div class="stat-val">${{props.rasio_murid_guru}}</div>
          </div>
        </div>
      `;
    }}

    function onEachFeature(feature, layer) {{
      const p = feature.properties;
      const key = p.kode_wilayah || p.nama_kabkota;
      layerMap[key] = layer;

      layer.bindTooltip(p.nama_pendek || p.nama_kabkota, {{
        permanent: false,
        direction: 'center',
        className: 'clean-tooltip'
      }});

      layer.on({{
        mouseover: function(e) {{
          const l = e.target;
          l.setStyle({{
            weight: 2,
            color: '#0f172a',
            fillOpacity: 0.9
          }});
          l.bringToFront();
          updateDetail(p);
        }},
        mouseout: function(e) {{
          if (selectedLayer !== e.target) {{
            geojsonLayer.resetStyle(e.target);
          }}
          if (!selectedLayer) {{
            updateDetail(null);
          }} else {{
            updateDetail(selectedLayer.feature.properties);
          }}
        }},
        click: function(e) {{
          if (selectedLayer) geojsonLayer.resetStyle(selectedLayer);
          selectedLayer = e.target;
          selectedLayer.setStyle({{
            weight: 2.5,
            color: '#0f172a',
            fillOpacity: 0.95
          }});
          updateDetail(p);
          document.getElementById('region-dropdown').value = key;
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
          <div class="legend-title">Klaster K-Means (k=4)</div>
          <div class="legend-row"><div class="legend-color" style="background:#2563eb"></div>Klaster 1: Perkotaan, Capaian Tinggi</div>
          <div class="legend-row"><div class="legend-color" style="background:#0284c7"></div>Klaster 2: Beban Kelas Tinggi</div>
          <div class="legend-row"><div class="legend-color" style="background:#d97706"></div>Klaster 3: Kemiskinan Tinggi</div>
          <div class="legend-row"><div class="legend-color" style="background:#dc2626"></div>Klaster 4: Capaian Rendah, Skala Terbatas</div>
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
      if (selectedLayer) {{
        updateDetail(selectedLayer.feature.properties);
      }}
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
      selectedLayer.setStyle({{
        weight: 2.5,
        color: '#0f172a',
        fillOpacity: 0.95
      }});
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
        opt.textContent = `${{f.properties.nama_kabkota}} (#${{f.properties.peringkat_ipi}})`;
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

html_out_path = OUTPUT_DIR / "peta_interaktif_jabar.html"
with open(html_out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

# Simpan juga ke folder docs/index.html untuk kemudahan hosting di GitHub Pages, Vercel, Netlify
docs_out_path = BASE_DIR / "docs" / "index.html"
docs_out_path.parent.mkdir(parents=True, exist_ok=True)
with open(docs_out_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Peta Web Interaktif HTML berhasil disimpan di:\n-> {html_out_path}\n-> {docs_out_path}")

