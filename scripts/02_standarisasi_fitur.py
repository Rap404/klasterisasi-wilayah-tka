"""
02_STANDARISASI_FITUR.PY
========================
Tahap 2 Pipeline: Seleksi 4 Pilar Fitur Utama dan Transformasi Z-Score
(StandardScaler) agar terbebas dari perbedaan skala satuan.

Input:
  - dataset/processed/jabar_smp_merged.csv
Output:
  - dataset/processed/fitur_kandidat_kmeans_smp.csv
  - dataset/processed/fitur_kmeans_smp_standardisasi.csv
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Konfigurasi Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_INPUT = BASE_DIR / "dataset" / "processed" / "jabar_smp_merged.csv"
DATA_CANDIDATE = BASE_DIR / "dataset" / "processed" / "fitur_kandidat_kmeans_smp.csv"
DATA_STANDAR = BASE_DIR / "dataset" / "processed" / "fitur_kmeans_smp_standardisasi.csv"

print("=" * 60)
print("[02] SELEKSI 4 PILAR FITUR & STANDARDISASI Z-SCORE")
print("=" * 60)

# 2. Muat Data
df = pd.read_csv(DATA_INPUT)

# Perbaiki kode_wilayah jika terbaca float (contoh: 32.10 -> 32.1)
if "kode_bps" in df.columns:
    _kb = df["kode_bps"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["kode_wilayah"] = _kb.str[:2] + "." + _kb.str[2:]

# 3. Empat Fitur Utama Klasterisasi
fitur = [
    "tka_rerata_mtk",        # Pilar 1: Capaian Mutu Akademik (Numerasi)
    "rasio_murid_guru",      # Pilar 2: Beban Tenaga Pendidik
    "rasio_murid_sekolah",   # Pilar 3: Kapasitas Fisik & Skala Sekolah
    "pct_penduduk_miskin"    # Pilar 4: Konteks Sosio-Ekonomi Wilayah
]

# Simpan kandidat fitur asli
kolom_identitas = [
    c for c in ["kode_wilayah", "kode_bps", "nama_kabkota", "nama_pendek", "jenis"]
    if c in df.columns
]
df_kandidat = df[kolom_identitas + ["tka_rerata_bind"] + fitur].copy()
df_kandidat.to_csv(DATA_CANDIDATE, index=False)
print(f"[OK] Kandidat fitur mentah disimpan ke:\n     -> {DATA_CANDIDATE}")

# 4. Validasi Data
df[fitur] = df[fitur].apply(pd.to_numeric, errors="coerce")
if df[fitur].isnull().any().any():
    raise ValueError("Ditemukan nilai kosong (NaN) pada fitur klasterisasi!")

# 5. Transformasi Z-Score: z = (x - mu) / sigma
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[fitur])

df_standar = pd.DataFrame(data_scaled, columns=fitur, index=df.index)
hasil = pd.concat([df[kolom_identitas], df_standar], axis=1)

# Simpan hasil standardisasi
hasil.to_csv(DATA_STANDAR, index=False)
print(f"[OK] Fitur hasil standardisasi Z-Score disimpan ke:\n     -> {DATA_STANDAR}")

# 6. Ringkasan Verifikasi
print("\nRingkasan Nilai Fitur Pasca-Standardisasi:")
print(f"- Rata-rata (mean, target=0): {hasil[fitur].mean().round(2).to_dict()}")
print(f"- Standar deviasi (std, target=1): {hasil[fitur].std().round(2).to_dict()}")
print("Selesai Tahap 02.\n")
