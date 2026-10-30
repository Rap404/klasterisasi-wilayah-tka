"""
RUN_ALL.PY
==========
Master Pipeline Runner untuk Analisis Klasterisasi Wilayah Pendidikan SMP Jawa Barat.

Penggunaan:
  - Menjalankan seluruh pipeline (Tahap 01 s/d 07):
      python run_all.py
  - Menjalankan tahap tertentu saja (contoh Tahap 4):
      python run_all.py --step 4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    {
        "step": 1,
        "name": "Eksplorasi Data (EDA) & Matriks Korelasi",
        "script": BASE_DIR / "scripts" / "01_eda_dan_korelasi.py",
        "desc": "Statistik deskriptif, korelasi Pearson, dan analisis multikolinearitas."
    },
    {
        "step": 2,
        "name": "Seleksi Fitur & Standardisasi Z-Score",
        "script": BASE_DIR / "scripts" / "02_standarisasi_fitur.py",
        "desc": "Ekstraksi 4 pilar pendidikan dan normalisasi Z-Score (StandardScaler)."
    },
    {
        "step": 3,
        "name": "Evaluasi & Validasi Nilai k Optimal",
        "script": BASE_DIR / "scripts" / "03_evaluasi_optimal_k.py",
        "desc": "Uji Elbow (WCSS), Silhouette Score, Davies-Bouldin Index, & Calinski-Harabasz."
    },
    {
        "step": 4,
        "name": "Pemodelan K-Means & Profiling Wilayah",
        "script": BASE_DIR / "scripts" / "04_kmeans_clustering.py",
        "desc": "Fitting K-Means (k=4), penetapan 4 klaster karakteristik wilayah, dan visualisasi diagnostik."
    },
    {
        "step": 5,
        "name": "Analisis Inferensial & Regresi Berganda",
        "script": BASE_DIR / "scripts" / "05_analisis_inferensial.py",
        "desc": "Uji beda nyata ANOVA & Kruskal-Wallis antar klaster serta pemodelan regresi OLS TKA."
    },
    {
        "step": 6,
        "name": "Perhitungan Indeks Prioritas Intervensi (IPI)",
        "script": BASE_DIR / "scripts" / "06_indeks_prioritas.py",
        "desc": "Formula IPI, pembobotan multidimensi, dan simulasi sensitivitas Monte Carlo."
    },
    {
        "step": 7,
        "name": "Integrasi Geospasial & Peta Tematik / Web GIS",
        "script": BASE_DIR / "scripts" / "07_visualisasi_geospasial.py",
        "desc": "Penggabungan GeoJSON, kartografi resolusi tinggi UTM 48S, dan Peta Web Interaktif."
    },
]


def run_step(step_info):
    print("\n" + "=" * 70)
    print(f"MENJALANKAN TAHAP {step_info['step']}: {step_info['name'].upper()}")
    print(f"Skrip: {step_info['script'].name}")
    print(f"Deskripsi: {step_info['desc']}")
    print("=" * 70)

    start_time = time.time()
    result = subprocess.run([sys.executable, str(step_info["script"])])
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\n[GAGAL] Tahap {step_info['step']} berhenti dengan error (exit code: {result.returncode})!")
        return False

    print(f"[BERHASIL] Tahap {step_info['step']} selesai dalam {elapsed:.2f} detik.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Master Pipeline Runner - Klasterisasi Wilayah Pendidikan SMP Jawa Barat"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=range(1, 8),
        help="Nomor tahapan spesifik yang ingin dijalankan (1 s/d 7). Kosongkan untuk menjalankan semua."
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("#  PIPELINE KLASTERISASI & INDEKS PRIORITAS PENDIDIKAN SMP JABAR   #")
    print("#" * 70)

    start_all = time.time()

    if args.step:
        target = next((s for s in PIPELINE_STEPS if s["step"] == args.step), None)
        if target:
            success = run_step(target)
            if not success:
                sys.exit(1)
    else:
        for step_info in PIPELINE_STEPS:
            success = run_step(step_info)
            if not success:
                sys.exit(1)

    total_time = time.time() - start_all
    print("\n" + "#" * 70)
    print(f"#  SELURUH PIPELINE BERHASIL DIJALANKAN DALAM {total_time:.2f} DETIK!          #")
    print("#  Artefak data dan gambar tersimpan rapi di:                          #")
    print("#  - dataset/processed/                                                #")
    print("#  - outputs/tables/                                                   #")
    print("#  - outputs/figures/                                                  #")
    print("#  - docs/index.html (GitHub Pages)                                    #")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
