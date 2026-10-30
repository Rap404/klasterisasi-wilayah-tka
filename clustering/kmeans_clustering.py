import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import textwrap
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 0. Pengaturan utama
K_UTAMA = 4          # klaster utama yang dipakai di seluruh analisis
K_PEMBANDING = 3     # hanya untuk pembanding / uji kekokohan
RANDOM_STATE = 42
N_INIT = 10

# 1. Konfigurasi direktori
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_STD_PATH = BASE_DIR / "dataset" / "fitur_kmeans_smp_standardisasi.csv"
DATA_RAW_PATH = BASE_DIR / "dataset" / "fitur_kandidat_kmeans_smp.csv"
OUTPUT_DIR = BASE_DIR / "clustering" / "statistik_grafik"
GRAFIK_DIR = OUTPUT_DIR / "grafik_eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)


def perbaiki_kode_wilayah(df):
    """kode_wilayah sering terbaca float (32.10 -> 32.1). Bangun ulang dari kode_bps sebagai teks."""
    if "kode_bps" in df.columns:
        kb = df["kode_bps"].astype(str).str.replace(r"\.0$", "", regex=True)
        df["kode_wilayah"] = kb.str[:2] + "." + kb.str[2:]
    return df


# 2. Muat data
df_std = pd.read_csv(DATA_STD_PATH)
df_raw = perbaiki_kode_wilayah(pd.read_csv(DATA_RAW_PATH))

fitur = [
    "tka_rerata_mtk",
    "rasio_murid_guru",
    "rasio_murid_sekolah",
    "pct_penduduk_miskin"
]

X = df_std[fitur].values

# 3. Fitting K-Means: k=4 (utama) dan k=3 (pembanding)
kmeans_utama = KMeans(n_clusters=K_UTAMA, random_state=RANDOM_STATE, n_init=N_INIT)
labels_utama = kmeans_utama.fit_predict(X)

kmeans_pemb = KMeans(n_clusters=K_PEMBANDING, random_state=RANDOM_STATE, n_init=N_INIT)
labels_pemb = kmeans_pemb.fit_predict(X)

print("=== METRIK KUALITAS KLASTER ===")
for nama, lab, k in [("utama", labels_utama, K_UTAMA), ("pembanding", labels_pemb, K_PEMBANDING)]:
    print(f"k={k} ({nama}): silhouette = {silhouette_score(X, lab):.3f}, "
          f"Davies-Bouldin = {davies_bouldin_score(X, lab):.3f}")


# 4. Label deskriptif k=4 berbasis aturan (tidak bergantung pada nomor acak K-Means)
#    - TKA tertinggi              -> "Perkotaan, Capaian Tertinggi"
#    - TKA terendah               -> "Capaian Terendah, Sekolah Kecil"
#    - dua klaster di tengah      -> yang rasio murid:guru lebih tinggi = "Beban Kelas Tinggi",
#                                    yang lain = "Kemiskinan Tertinggi"
#    Nomor klaster diurutkan menurut rerata TKA Matematika (1 = tertinggi).
def beri_label(df_data, labels):
    tmp = df_data.copy()
    tmp["cid"] = labels
    rata = tmp.groupby("cid")[fitur].mean()
    urut = rata.sort_values("tka_rerata_mtk", ascending=False).index.tolist()
    assert len(urut) == 4, "Aturan label ini dirancang untuk k=4"

    tertinggi, terendah, tengah = urut[0], urut[-1], urut[1:-1]
    beban = max(tengah, key=lambda c: rata.loc[c, "rasio_murid_guru"])
    miskin = [c for c in tengah if c != beban][0]

    nama = {
        tertinggi: "Perkotaan, Capaian Tinggi",
        beban: "Beban Kelas Tinggi, Capaian Menengah",
        miskin: "Kemiskinan Tinggi, Capaian Menengah",
        terendah: "Capaian Rendah, Skala Sekolah Terbatas",
    }
    peta_no = {cid: no for no, cid in enumerate(urut, start=1)}
    peta_nama = {cid: f"Klaster {peta_no[cid]}: {nama[cid]}" for cid in urut}

    # Pemeriksaan kewajaran label (peringatan, tidak menghentikan program)
    if "jenis" in tmp.columns:
        porsi_kota = tmp.groupby("cid")["jenis"].apply(lambda s: (s == "Kota").mean())
        if porsi_kota[tertinggi] < 0.8:
            print("PERINGATAN: klaster 'Perkotaan' berisi < 80% kota. Tinjau ulang label.")
    if rata["rasio_murid_sekolah"].idxmin() != terendah:
        print("PERINGATAN: klaster capaian rendah bukan yang murid/sekolah-nya terkecil. Tinjau label 'Skala Sekolah Terbatas'.")
    if rata["pct_penduduk_miskin"].idxmax() != miskin:
        print("PERINGATAN: klaster 'Kemiskinan Tinggi' bukan yang % miskin-nya tinggi. Tinjau label.")
    return peta_no, peta_nama


peta_no, peta_nama = beri_label(df_raw, labels_utama)

# 5. Gabungkan hasil klasterisasi ke dataset final
hasil_df = df_raw.copy()
hasil_df["klaster_k4"] = pd.Series(labels_utama).map(peta_no).values
hasil_df["nama_klaster_k4"] = pd.Series(labels_utama).map(peta_nama).values
hasil_df["klaster_k3"] = labels_pemb  # pembanding (nomor mentah K-Means)

output_csv = BASE_DIR / "dataset" / "hasil_klaster_kmeans_smp.csv"
hasil_df.to_csv(output_csv, index=False)
print(f"\nDataset hasil klasterisasi berhasil disimpan di: {output_csv}")

# 6. Profil statistik tiap klaster (nilai asli)
profil_mean = hasil_df.groupby("nama_klaster_k4")[fitur].mean().round(2)
profil_count = hasil_df.groupby("nama_klaster_k4")["nama_kabkota"].count().rename("jumlah_wilayah")
profil_parts = [profil_count, profil_mean]
if "jenis" in hasil_df.columns:
    profil_parts.insert(1, hasil_df.groupby("nama_klaster_k4")["jenis"].apply(lambda s: (s == "Kota").sum()).rename("jumlah_kota"))

profil_ringkas = pd.concat(profil_parts, axis=1).reset_index()
profil_csv = OUTPUT_DIR / "profil_klaster_smp.csv"
profil_ringkas.to_csv(profil_csv, index=False)
print("\n=== PROFIL STATISTIK RATA-RATA NILAI ASLI PER KLASTER ===")
print(profil_ringkas.to_string(index=False))

print("\n=== ANGGOTA KABUPATEN/KOTA PER KLASTER ===")
for nama_klaster, grup in hasil_df.groupby("nama_klaster_k4"):
    print(f"\n[{nama_klaster}] (Total: {len(grup)} daerah)")
    print(", ".join(grup["nama_pendek"].tolist()))

# 7. Visualisasi hasil klasterisasi (Grafik Terpisah & Gabungan Resolusi Tinggi)
palet = {1: "#2563eb", 2: "#0284c7", 3: "#d97706", 4: "#dc2626"}   # biru, sky, amber, merah
urut_nama = sorted(hasil_df["nama_klaster_k4"].unique())            # "Klaster 1..4"
warna_klaster = {n: palet[int(n.split(":")[0].split()[1])] for n in urut_nama}


def label_pendek(nama, lebar=14):
    return textwrap.fill(nama.split(": ", 1)[1], lebar)


# --- GRAFIK 1 TERPISAH: TKA MTK vs Persentase Kemiskinan ---
fig1, ax1 = plt.subplots(figsize=(10, 7), dpi=300)
for nama_klaster in urut_nama:
    grup = hasil_df[hasil_df["nama_klaster_k4"] == nama_klaster]
    ax1.scatter(grup["pct_penduduk_miskin"], grup["tka_rerata_mtk"], label=nama_klaster,
                color=warna_klaster[nama_klaster], s=90, alpha=0.9, edgecolors="black", linewidths=0.8)
    for _, r in grup.iterrows():
        ax1.annotate(r["nama_pendek"], (r["pct_penduduk_miskin"], r["tka_rerata_mtk"]),
                     fontsize=8.5, xytext=(5, 5), textcoords="offset points")

ax1.axhline(hasil_df["tka_rerata_mtk"].mean(), color="gray", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Rerata TKA Jabar ({hasil_df['tka_rerata_mtk'].mean():.2f})")
ax1.axvline(hasil_df["pct_penduduk_miskin"].mean(), color="gray", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Rerata Kemiskinan Jabar ({hasil_df['pct_penduduk_miskin'].mean():.2f}%)")

ax1.set_title("Capaian TKA Matematika vs Persentase Kemiskinan (K-Means, k=4)", fontsize=13, fontweight="bold", pad=12)
ax1.set_xlabel("Persentase Penduduk Miskin (%)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Rerata Nilai TKA Matematika", fontsize=11, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
plt.tight_layout()
g1_path = GRAFIK_DIR / "klaster_scatter_tka_vs_kemiskinan.png"
plt.savefig(g1_path, dpi=300, bbox_inches="tight")
plt.close(fig1)

# --- GRAFIK 2 TERPISAH: Rasio Murid/Guru vs Rasio Murid/Sekolah ---
fig2, ax2 = plt.subplots(figsize=(10, 7), dpi=300)
for nama_klaster in urut_nama:
    grup = hasil_df[hasil_df["nama_klaster_k4"] == nama_klaster]
    ax2.scatter(grup["rasio_murid_sekolah"], grup["rasio_murid_guru"], label=nama_klaster,
                color=warna_klaster[nama_klaster], s=90, alpha=0.9, edgecolors="black", linewidths=0.8)
    for _, r in grup.iterrows():
        ax2.annotate(r["nama_pendek"], (r["rasio_murid_sekolah"], r["rasio_murid_guru"]),
                     fontsize=8.5, xytext=(5, 5), textcoords="offset points")

ax2.axhline(hasil_df["rasio_murid_guru"].mean(), color="gray", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Rerata Rasio M/G Jabar ({hasil_df['rasio_murid_guru'].mean():.2f})")
ax2.axvline(hasil_df["rasio_murid_sekolah"].mean(), color="gray", linestyle=":", linewidth=1, alpha=0.7,
            label=f"Rerata Rasio M/S Jabar ({hasil_df['rasio_murid_sekolah'].mean():.2f})")

ax2.set_title("Rasio Beban Guru vs Skala Kepadatan Sekolah (K-Means, k=4)", fontsize=13, fontweight="bold", pad=12)
ax2.set_xlabel("Rasio Murid per Sekolah (Kepadatan/Ukuran Sekolah)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Rasio Murid per Guru (Beban Pengajaran Guru)", fontsize=11, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
plt.tight_layout()
g2_path = GRAFIK_DIR / "klaster_scatter_guru_vs_sekolah.png"
plt.savefig(g2_path, dpi=300, bbox_inches="tight")
plt.close(fig2)

# --- GRAFIK 3 TERPISAH: Fingerprint Z-Score Antar-Klaster ---
fig3, ax3 = plt.subplots(figsize=(11, 6.5), dpi=300)
df_std_labeled = df_std.copy()
df_std_labeled["nama_klaster_k4"] = hasil_df["nama_klaster_k4"].values
centroid_z = df_std_labeled.groupby("nama_klaster_k4")[fitur].mean().T

posisi = np.arange(len(fitur))
lebar = 0.8 / K_UTAMA
fitur_labels = ["TKA Matematika", "Rasio Murid/Guru", "Rasio Murid/Sekolah", "% Penduduk Miskin"]
for idx, nama_klaster in enumerate(urut_nama):
    vals = centroid_z[nama_klaster]
    x_pos = posisi + (idx - (K_UTAMA - 1) / 2) * lebar
    bars = ax3.bar(x_pos, vals, width=lebar,
                   label=nama_klaster, color=warna_klaster[nama_klaster], alpha=0.9, edgecolor="black", linewidth=0.7)
    for b in bars:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width() / 2, h + (0.04 if h >= 0 else -0.1),
                 f"{h:+.2f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8, fontweight="bold")

ax3.axhline(0, color="black", linestyle="--", linewidth=1.0)
ax3.set_xticks(posisi)
ax3.set_xticklabels(fitur_labels, fontsize=10.5, fontweight="bold")
ax3.set_title("Karakteristik Z-Score Rata-Rata Antar-Klaster (Fingerprint Mutu & Konteks)", fontsize=13, fontweight="bold", pad=12)
ax3.set_ylabel("Rata-rata Nilai Standardisasi (Z-Score)", fontsize=11, fontweight="bold")
ax3.set_ylim(centroid_z.values.min() - 0.35, centroid_z.values.max() + 0.35)
ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
plt.tight_layout()
g3_path = GRAFIK_DIR / "klaster_fingerprint_zscore.png"
plt.savefig(g3_path, dpi=300, bbox_inches="tight")
plt.close(fig3)

# --- GRAFIK 4 TERPISAH: Distribusi Jumlah Wilayah per Klaster ---
fig4, ax4 = plt.subplots(figsize=(9.5, 6), dpi=300)
jumlah = hasil_df["nama_klaster_k4"].value_counts().reindex(urut_nama)
bars4 = ax4.bar(range(len(urut_nama)), jumlah.values, color=[warna_klaster[n] for n in urut_nama],
                width=0.55, edgecolor="black", linewidth=0.8, alpha=0.9)
ax4.set_xticks(range(len(urut_nama)))
ax4.set_xticklabels([f"K{n.split(':')[0].split()[1]}: {label_pendek(n, 16)}" for n in urut_nama],
                    fontsize=9.5, fontweight="bold")
ax4.set_title(f"Distribusi Jumlah Kabupaten/Kota per Klaster (Total N = {len(hasil_df)})", fontsize=13, fontweight="bold", pad=12)
ax4.set_ylabel("Jumlah Wilayah (Kabupaten / Kota)", fontsize=11, fontweight="bold")
ax4.set_ylim(0, jumlah.max() + 3)
ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
for bar in bars4:
    yval = bar.get_height()
    pct = (yval / len(hasil_df)) * 100
    ax4.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f"{int(yval)} daerah\n({pct:.1f}%)",
             ha="center", va="bottom", fontweight="bold", fontsize=10)

plt.tight_layout()
g4_path = GRAFIK_DIR / "klaster_distribusi_wilayah.png"
plt.savefig(g4_path, dpi=300, bbox_inches="tight")
plt.close(fig4)

# --- GRAFIK GABUNGAN 2x2 (OVERVIEW) ---
fig, axes = plt.subplots(2, 2, figsize=(17, 13), dpi=200)

# Subplot 1: TKA MTK vs persentase kemiskinan
ax1 = axes[0, 0]
for nama_klaster in urut_nama:
    grup = hasil_df[hasil_df["nama_klaster_k4"] == nama_klaster]
    ax1.scatter(grup["pct_penduduk_miskin"], grup["tka_rerata_mtk"], label=nama_klaster,
                color=warna_klaster[nama_klaster], s=80, alpha=0.85, edgecolors="black")
    for _, r in grup.iterrows():
        ax1.annotate(r["nama_pendek"], (r["pct_penduduk_miskin"], r["tka_rerata_mtk"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
ax1.set_title("Capaian TKA Matematika vs Persentase Kemiskinan", fontsize=11, fontweight="bold")
ax1.set_xlabel("Persentase Penduduk Miskin (%)")
ax1.set_ylabel("Rerata TKA Matematika")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="upper right", fontsize=7)

# Subplot 2: rasio murid/guru vs rasio murid/sekolah
ax2 = axes[0, 1]
for nama_klaster in urut_nama:
    grup = hasil_df[hasil_df["nama_klaster_k4"] == nama_klaster]
    ax2.scatter(grup["rasio_murid_sekolah"], grup["rasio_murid_guru"], label=nama_klaster,
                color=warna_klaster[nama_klaster], s=80, alpha=0.85, edgecolors="black")
    for _, r in grup.iterrows():
        ax2.annotate(r["nama_pendek"], (r["rasio_murid_sekolah"], r["rasio_murid_guru"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
ax2.set_title("Rasio Murid/Guru vs Rasio Murid/Sekolah", fontsize=11, fontweight="bold")
ax2.set_xlabel("Rasio Murid per Sekolah")
ax2.set_ylabel("Rasio Murid per Guru")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper left", fontsize=7)

# Subplot 3: z-score centroid antar-klaster (fingerprint)
ax3 = axes[1, 0]
for idx, nama_klaster in enumerate(urut_nama):
    ax3.bar(posisi + (idx - (K_UTAMA - 1) / 2) * lebar, centroid_z[nama_klaster], width=lebar,
            label=nama_klaster, color=warna_klaster[nama_klaster], alpha=0.85, edgecolor="black")
ax3.axhline(0, color="black", linestyle="--", linewidth=0.8)
ax3.set_xticks(posisi)
ax3.set_xticklabels(fitur_labels, fontsize=9, fontweight="bold")
ax3.set_title("Karakteristik Z-Score Rata-Rata Antar-Klaster (Fingerprint)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Rata-rata Z-Score")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.legend(loc="upper right", fontsize=7)

# Subplot 4: distribusi jumlah wilayah per klaster
ax4 = axes[1, 1]
bars = ax4.bar(range(len(urut_nama)), jumlah.values, color=[warna_klaster[n] for n in urut_nama],
               width=0.55, edgecolor="black")
ax4.set_xticks(range(len(urut_nama)))
ax4.set_xticklabels([f"K{n.split(':')[0].split()[1]}\n{label_pendek(n)}" for n in urut_nama],
                    fontsize=8, fontweight="bold")
ax4.set_title(f"Distribusi Jumlah Kabupaten/Kota per Klaster (N={len(hasil_df)})", fontsize=11, fontweight="bold")
ax4.set_ylabel("Jumlah Kabupaten/Kota")
ax4.set_ylim(0, jumlah.max() + 4)
ax4.grid(True, linestyle="--", alpha=0.5, axis="y")
for bar in bars:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2, yval + 0.4, f"{int(yval)} daerah",
             ha="center", va="bottom", fontweight="bold", fontsize=10)

plt.suptitle(f"Hasil Klasterisasi Wilayah SMP Jawa Barat (K-Means, k={K_UTAMA})", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()

grafik_output = GRAFIK_DIR / "visualisasi_klaster_kmeans.png"
plt.savefig(grafik_output, dpi=200, bbox_inches="tight")
plt.close()

print(f"\nGrafik visualisasi klasterisasi berhasil disimpan:")
print(f"1. Scatter TKA vs Kemiskinan    : {g1_path}")
print(f"2. Scatter Rasio Guru vs Sekolah: {g2_path}")
print(f"3. Fingerprint Z-Score          : {g3_path}")
print(f"4. Distribusi Jumlah Wilayah    : {g4_path}")
print(f"5. Panel Gabungan (2x2)         : {grafik_output}")