# 8. Analisis Hubungan Capaian TKA dengan Variabel Pendukung
from pathlib import Path
import textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "dataset" / "hasil_klaster_kmeans_smp.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
GRAFIK_DIR = OUTPUT_DIR / "grafik_eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GRAFIK_DIR.mkdir(parents=True, exist_ok=True)

# Baca dataset hasil klasterisasi
hasil_df = pd.read_csv(DATA_PATH)

fitur = [
    "tka_rerata_mtk",
    "rasio_murid_guru",
    "rasio_murid_sekolah",
    "pct_penduduk_miskin"
]
fitur_labels = ["TKA MTK", "Rasio M/G", "Rasio M/S", "Pmiskin (%)"]

# Warna & urutan klaster dibaca dari data (Klaster 1 = capaian tertinggi ... Klaster 4 = terendah)
palet = {1: "#2563eb", 2: "#16a34a", 3: "#f59e0b", 4: "#dc2626"}
nama_urut = (
    hasil_df[["klaster_k4", "nama_klaster_k4"]]
    .drop_duplicates()
    .sort_values("klaster_k4")["nama_klaster_k4"]
    .tolist()
)
warna_klaster = {
    n: palet[int(hasil_df.loc[hasil_df["nama_klaster_k4"] == n, "klaster_k4"].iloc[0])]
    for n in nama_urut
}


def label_pendek(nama, lebar=16):
    no = nama.split(":")[0].replace("Klaster ", "K")
    return f"{no}\n" + textwrap.fill(nama.split(": ", 1)[1], lebar)


target = "tka_rerata_mtk"
pendukung = ["rasio_murid_guru", "rasio_murid_sekolah", "pct_penduduk_miskin"]
label_pendukung = {
    "rasio_murid_guru": "Rasio Murid/Guru",
    "rasio_murid_sekolah": "Rasio Murid/Sekolah",
    "pct_penduduk_miskin": "Persentase Penduduk Miskin (%)"
}

# 8a. Korelasi Pearson & Spearman + uji signifikansi
baris = []
for v in pendukung:
    r_p, p_p = stats.pearsonr(hasil_df[v], hasil_df[target])
    r_s, p_s = stats.spearmanr(hasil_df[v], hasil_df[target])
    slope, intercept, r_val, p_reg, se = stats.linregress(hasil_df[v], hasil_df[target])
    baris.append({
        "variabel": v,
        "pearson_r": round(r_p, 3), "pearson_p": round(p_p, 4),
        "spearman_rho": round(r_s, 3), "spearman_p": round(p_s, 4),
        "slope": round(slope, 4), "r_kuadrat": round(r_val ** 2, 3),
    })

korelasi_df = pd.DataFrame(baris)
korelasi_csv = OUTPUT_DIR / "korelasi_tka_variabel_pendukung.csv"
korelasi_df.to_csv(korelasi_csv, index=False)
print("\n=== KORELASI TKA MATEMATIKA DENGAN VARIABEL PENDUKUNG ===")
print(korelasi_df.to_string(index=False))

# 8b. Matriks korelasi seluruh fitur (heatmap)
corr = hasil_df[fitur].corr(method="pearson")
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(fitur)))
ax.set_yticks(range(len(fitur)))
ax.set_xticklabels(fitur_labels, rotation=30, ha="right")
ax.set_yticklabels(fitur_labels)
for i in range(len(fitur)):
    for j in range(len(fitur)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=10, fontweight="bold")
plt.colorbar(im, ax=ax, label="Koefisien Korelasi Pearson")
ax.set_title("Matriks Korelasi Antar-Fitur", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIK_DIR / "heatmap_korelasi.png", dpi=200, bbox_inches="tight")
plt.close()

# 8c. Scatter plot + garis regresi per variabel pendukung
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for ax, v in zip(axes, pendukung):
    for nama_klaster in nama_urut:
        grup = hasil_df[hasil_df["nama_klaster_k4"] == nama_klaster]
        ax.scatter(grup[v], grup[target], color=warna_klaster[nama_klaster],
                   s=70, alpha=0.85, edgecolors="black", label=nama_klaster)
    slope, intercept, r_val, p_reg, _ = stats.linregress(hasil_df[v], hasil_df[target])
    xs = np.linspace(hasil_df[v].min(), hasil_df[v].max(), 100)
    ax.plot(xs, slope * xs + intercept, color="black", linestyle="--", linewidth=1.5)
    ax.set_title(f"TKA MTK vs {label_pendukung[v]}\nr = {r_val:.2f}, p = {p_reg:.3f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel(label_pendukung[v])
    ax.set_ylabel("Rerata TKA Matematika")
    ax.grid(True, linestyle="--", alpha=0.5)
axes[0].legend(fontsize=6.5, loc="best")
plt.suptitle("Hubungan Capaian TKA Matematika dengan Variabel Pendukung", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(GRAFIK_DIR / "scatter_regresi_tka.png", dpi=200, bbox_inches="tight")
plt.close()

# 8d. Perbedaan TKA antar-klaster (ANOVA satu arah + Kruskal-Wallis)
# CATATAN: TKA Matematika ikut menjadi fitur pembentuk klaster, sehingga perbedaan TKA antar-klaster
# sebagian bersifat by construction. Uji ini dipakai sebagai gambaran deskriptif, bukan bukti independen.
kelompok = [hasil_df.loc[hasil_df["nama_klaster_k4"] == n, target].values for n in nama_urut]
f_stat, p_anova = stats.f_oneway(*kelompok)
h_stat, p_kw = stats.kruskal(*kelompok)
print("\n=== UJI PERBEDAAN TKA MTK ANTAR-KLASTER (deskriptif) ===")
print(f"ANOVA         : F = {f_stat:.3f}, p = {p_anova:.4f}")
print(f"Kruskal-Wallis: H = {h_stat:.3f}, p = {p_kw:.4f}")

# 8e. Boxplot TKA per klaster
fig, ax = plt.subplots(figsize=(10, 6))
data_box = [hasil_df.loc[hasil_df["nama_klaster_k4"] == n, target] for n in nama_urut]
bp = ax.boxplot(data_box, patch_artist=True, tick_labels=[label_pendek(n) for n in nama_urut])
for patch, n in zip(bp["boxes"], nama_urut):
    patch.set_facecolor(warna_klaster[n])
    patch.set_alpha(0.7)
ax.set_ylabel("Rerata TKA Matematika")
ax.set_title("Sebaran TKA Matematika per Klaster", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", labelsize=8)
ax.grid(True, linestyle="--", alpha=0.5, axis="y")
plt.tight_layout()
plt.savefig(GRAFIK_DIR / "boxplot_tka_per_klaster.png", dpi=200, bbox_inches="tight")
plt.close()


# 8f. Regresi berganda EKSPLORATIF (dasar pemilihan komponen dan bobot indeks prioritas)
# CATATAN: n = 27 dan prediktor saling berkaitan; hasil dibaca sebagai gambaran asosiasi, bukan sebab-akibat.
# Rasio murid:guru berkorelasi lemah dengan TKA secara sederhana, tetapi koefisien parsialnya bisa
# berubah setelah kemiskinan dan ukuran sekolah dikendalikan (wilayah miskin cenderung berrasio longgar).
yv = hasil_df[target].values
Xm = np.column_stack([np.ones(len(yv))] + [hasil_df[v].values for v in pendukung])
b = np.linalg.lstsq(Xm, yv, rcond=None)[0]
resid = yv - Xm @ b
n_obs, k_par = Xm.shape
s2 = resid @ resid / (n_obs - k_par)
se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
t_val = b / se
p_val = 2 * (1 - stats.t.cdf(np.abs(t_val), n_obs - k_par))
sst = (yv - yv.mean()) @ (yv - yv.mean())
r2 = 1 - (resid @ resid) / sst
r2_adj = 1 - (1 - r2) * (n_obs - 1) / (n_obs - k_par)
beta_std = b[1:] * Xm[:, 1:].std(axis=0, ddof=1) / yv.std(ddof=1)

reg_df = pd.DataFrame({
    "variabel": ["konstanta"] + pendukung,
    "koefisien": np.round(b, 4),
    "std_error": np.round(se, 4),
    "p_value": np.round(p_val, 4),
    "beta_standar": [np.nan] + list(np.round(beta_std, 3)),
})
reg_csv = OUTPUT_DIR / "regresi_berganda_tka.csv"
reg_df.to_csv(reg_csv, index=False)
print("\n=== REGRESI BERGANDA EKSPLORATIF (TKA MTK ~ pendukung) ===")
print(reg_df.to_string(index=False))
print(f"R2 = {r2:.3f} | R2 terkoreksi = {r2_adj:.3f} | n = {n_obs}")

print(f"\nAnalisis hubungan TKA selesai. Output tersimpan di: {OUTPUT_DIR}")