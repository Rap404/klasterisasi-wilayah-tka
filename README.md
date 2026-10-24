# Klasterisasi Wilayah Pendidikan SMP Jawa Barat Menggunakan K-Means Clustering
> **Analisis Disparitas Mutu Capaian TKA, Kebutuhan Sarana Prasarana, dan Konteks Sosio-Ekonomi di 27 Kabupaten/Kota Jawa Barat**

---

## Daftar Isi
1. [Latar Belakang & Tujuan](#1-latar-belakang--tujuan)
2. [Alur Kerja Data Mining (Pipeline)](#2-alur-kerja-data-mining-pipeline)
3. [Tahap 1: Pengumpulan & Sumber Data](#tahap-1-pengumpulan--sumber-data)
4. [Tahap 2: Integrasi & Pembersihan Data](#tahap-2-integrasi--pembersihan-data)
5. [Tahap 3: Analisis Eksploratif (EDA) & Multikolinearitas](#tahap-3-analisis-eksploratif-eda--multikolinearitas)
6. [Tahap 4: Seleksi Fitur & Standardisasi Data](#tahap-4-seleksi-fitur--standardisasi-data)
7. [Tahap 5: Evaluasi Penentuan Jumlah Klaster Optimal (k)](#tahap-5-evaluasi-penentuan-jumlah-klaster-optimal-k)
8. [Tahap 6: Eksekusi K-Means Clustering & Profiling Wilayah](#tahap-6-eksekusi-k-means-clustering--profiling-wilayah)
9. [Tahap 7: Analisis Inferensial Hubungan Variabel Pendukung (ANOVA & Regresi)](#tahap-7-analisis-inferensial-hubungan-variabel-pendukung-anova--regresi)
10. [Tahap 8: Penyusunan Formula & Bobot Indeks Prioritas Intervensi (IPI)](#tahap-8-penyusunan-formula--bobot-indeks-prioritas-intervensi-ipi)
11. [Rekomendasi Kebijakan Berbasis Data (Policy Action)](#11-rekomendasi-kebijakan-berbasis-data-policy-action)
12. [Struktur Direktori Proyek](#12-struktur-direktori-proyek)
13. [Panduan Menjalankan Skrip](#13-panduan-menjalankan-skrip)

---

## 1. Latar Belakang & Tujuan

Pemerataan kualitas pendidikan jenjang Sekolah Menengah Pertama (SMP) di Provinsi Jawa Barat menghadapi tantangan multidimensi. Disparitas mutu hasil belajar siswa tidak hanya dipengaruhi oleh faktor internal proses belajar mengajar, melainkan berkaitan erat dengan:
1. **Daya tampung sarana fisik dan kecukupan tenaga pendidik**.
2. **Kondisi kesejahteraan sosio-ekonomi keluarga siswa di wilayah tersebut**.

Proyek data mining ini bertujuan untuk:
* Mengidentifikasi pola disparitas pendidikan di 27 kabupaten/kota Jawa Barat secara objektif dan matematis.
* Mengelompokkan wilayah ke dalam klaster profil yang jelas dan interpretatif (*Wilayah Maju*, *Wilayah Darurat Sarana Prasarana*, dan *Wilayah Rentan dan Terpencil*).
* Menyusun **Indeks Prioritas Intervensi (IPI)** berbasis pemodelan kuantitatif untuk memberikan panduan urutan peringkat alokasi anggaran dan program afirmasi pendidikan.

---

## 2. Alur Kerja Data Mining (Pipeline)

Proses data mining dilakukan mengacu pada metodologi baku CRISP-DM (*Cross-Industry Standard Process for Data Mining*):

```mermaid
flowchart TD
    A["1. Data Collection\n(TKA Pusmendik, Dapodik Kemendikbud, BPS Jabar)"] --> B["2. Data Integration & Cleaning\n(Penyelarasan Kode Wilayah & Validasi 27 Kab/Kota Jenjang SMP)"]
    B --> C["3. Exploratory Data Analysis & Feature Selection\n(Pearson Correlation & Eliminasi Count-Bias)"]
    C --> D["4. Data Normalization / Standardization\n(Z-Score Transformation / StandardScaler)"]
    D --> E["5. Cluster Validation & Optimal k Evaluation\n(Elbow WCSS, Silhouette Score, Davies-Bouldin Index)"]
    E --> F["6. K-Means Modeling & Cluster Profiling\n(Penetapan k=3 & Profil Karakteristik Wilayah)"]
    F --> G["7. Inferential Analysis & Significance Testing\n(Regresi Linear, ANOVA F=10.54, Kruskal-Wallis)"]
    G --> H["8. Composite Intervention Priority Index (IPI)\n(Min-Max Normalization & Weighted Ranking 1-27)"]
    H --> I["9. Evidence-Based Policy Recommendations\n(Alokasi Anggaran & Program Afirmasi Tepat Sasaran)"]
```

---

## Tahap 1: Pengumpulan & Sumber Data

Dataset dikumpulkan dari tiga domain data primer:
1. **Domain Capaian Akademik**:
   * Data nilai rerata Tes Kemampuan Akademik (TKA) jenjang SMP bidang Matematika (`tka_rerata_mtk`) dan Bahasa Indonesia (`tka_rerata_bind`) yang diperoleh dari hasil *web scraping* resmi.
2. **Domain Infrastruktur & SDM Pendidikan (Dapodik Kemendikbud)**:
   * Jumlah sekolah SMP (negeri dan swasta).
   * Jumlah guru dan tenaga kependidikan (TKP).
   * Jumlah murid terdaftar.
3. **Domain Sosio-Ekonomi Daerah (BPS Jawa Barat)**:
   * Persentase penduduk miskin per kabupaten/kota (`pct_penduduk_miskin`).
   * Garis kemiskinan (Rp/kapita/bulan).
4. **Domain Spasial**:
   * Peta batas administrasi 27 kabupaten/kota Provinsi Jawa Barat dalam format GeoJSON.

---

## Tahap 2: Integrasi & Pembersihan Data

* **Penyelarasan Entitas**: Penggabungan multi-sumber data dilakukan menggunakan pencocokan ganda: `kode_wilayah` (Kemendagri), `kode_bps`, dan nama kabupaten/kota resmi Jawa Barat.
* **Filter Jenjang**: Difokuskan spesifik pada jenjang SMP di seluruh 27 kabupaten/kota (18 kabupaten dan 9 kota).
* **Validasi Kualitas Data**: Dilakukan pengecekan *missing value* ($0$ nilai kosong ditemukan) dan penyeragaman tipe data ke format floating-point numerik.
* **Output Tahap Ini**: Disimpan dalam file [`dataset/jabar_smp_merged.csv`](file:///dataset/jabar_smp_merged.csv) dan [`dataset/dataset_final_smp.csv`](file:///dataset/dataset_final_smp.csv).

---

## Tahap 3: Analisis Eksploratif (EDA) & Multikolinearitas

Analisis korelasi dijalankan melalui skrip [`clustering/statistik_grafik/korelasi.py`](file:///clustering/statistik_grafik/korelasi.py) menggunakan koefisien korelasi Pearson:

$$r_{xy} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

### Alasan Pengambilan Keputusan (Eliminasi *Count-Bias*):
1. **Masalah Variabel Agregat Mentah**:
   * Ditemukan multikolinearitas ekstrem pada variabel jumlah absolut:
     * `guru_total` vs `murid_total` ($r = 0.987$)
     * `sekolah_total` vs `guru_total` ($r = 0.955$)
     * `sekolah_total` vs `murid_total` ($r = 0.950$)
   * *Alasan Keputusan*: Jumlah absolut sangat bias terhadap luas geografis dan jumlah penduduk kabupaten (misal Kabupaten Bogor berpenduduk 5+ juta jiwa vs Kota Cirebon berpenduduk 300 ribu jiwa). Jika dimasukkan ke klasterisasi, algoritma hanya akan mengelompokkan wilayah berdasarkan ukuran daerah (*size-bias*), bukan mutu pendidikannya.
2. **Solusi Perhitungan Rasio**:
   * Dihitung variabel intensif:
     * **Rasio Murid per Guru**: $\text{Rasio M/G} = \frac{\text{murid\_total}}{\text{guru\_total}}$ (mengukur beban pengajaran guru).
     * **Rasio Murid per Sekolah**: $\text{Rasio M/S} = \frac{\text{murid\_total}}{\text{sekolah\_total}}$ (mengukur skala dan kepadatan gedung sekolah).
3. **Temuan Korelasi Akademik & Kemiskinan**:
   * Korelasi `tka_rerata_bind` dan `tka_rerata_mtk` sebesar **$0.978$** (hampir sempurna).
   * Korelasi antara skor TKA dengan `pct_penduduk_miskin` bernilai negatif kuat ($r = -0.62$ s/d $-0.65$), membuktikan bahwa kondisi ekonomi daerah berpengaruh signifikan terhadap capaian akademik siswa.

---

## Tahap 4: Seleksi Fitur & Standardisasi Data

Dijalankan melalui skrip [`clustering/standarisasi.py`](file:///clustering/standarisasi.py).

### 1. Lima Fitur Terpilih
Dipilih 5 fitur yang mewakili 3 pilar penentu:
* **Pilar Mutu Akademik**: `tka_rerata_mtk` dan `tka_rerata_bind`.
* **Pilar Kapasitas SDM & Sarpras**: `rasio_murid_guru` dan `rasio_murid_sekolah`.
* **Pilar Konteks Sosio-Ekonomi**: `pct_penduduk_miskin`.

### 2. Rumus Standardisasi (Z-Score Normalization)
Algoritma K-Means mengukur kemiripan antar-daerah menggunakan **jarak Euclidean**:

$$d(p, q) = \sqrt{\sum_{j=1}^{d} (p_j - q_j)^2}$$

Karena satuan masing-masing fitur sangat berbeda (rasio murid per sekolah bernilai $200 - 450$, sedangkan kemiskinan bernilai $2\% - 11\%$, dan nilai TKA berkisar $37 - 72$), fitur dengan angka ratusan akan mendominasi perhitungan jarak. Oleh karena itu, seluruh fitur ditransformasikan ke skala Z-Score:

$$z = \frac{x - \mu}{\sigma}$$

Di mana:
* $\mu = \frac{1}{N}\sum_{i=1}^{N} x_i$ (Rata-rata fitur)
* $\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N} (x_i - \mu)^2}$ (Standar deviasi fitur)

Setelah transformasi: $\mu = 0$ dan $\sigma = 1$ untuk seluruh fitur.
* **Output**: Disimpan di [`dataset/fitur_kmeans_smp_standardisasi.csv`](file:///dataset/fitur_kmeans_smp_standardisasi.csv).

---

## Tahap 5: Evaluasi Penentuan Jumlah Klaster Optimal ($k$)

Dijalankan melalui skrip [`clustering/evaluasi_k.py`](file:///clustering/evaluasi_k.py). Pengujian dilakukan untuk rentang $k = 2$ hingga $k = 8$.

### Tiga Metrik Pengujian:

#### 1. Elbow Method (Within-Cluster Sum of Squares / WCSS)
Mengukur total variansi kuadrat jarak titik data ke pusat klasternya:

$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x \in C_k} ||x - \mu_k||^2$$

#### 2. Silhouette Coefficient
Mengukur seberapa dekat suatu titik dengan klasternya sendiri dibandingkan dengan klaster terdekat lainnya:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Di mana $a(i)$ adalah rata-rata jarak ke sesama anggota klaster, dan $b(i)$ adalah rata-rata jarak ke anggota klaster terdekat berikutnya. Nilai berkisar antara $-1 \le s(i) \le 1$ (makin mendekati 1 makin optimal).

#### 3. Davies-Bouldin Index (DBI)
Mengukur rasio penyebaran internal klaster terhadap pemisahan antarklaster:

$$DB = \frac{1}{k}\sum_{i=1}^{k} \max_{j \neq i} \left( \frac{\bar{d}_i + \bar{d}_j}{d(c_i, c_j)} \right)$$

Nilai semakin kecil mendekati $0$ menunjukkan klaster yang semakin terpisah dengan baik (*well-separated*).

### Tabel Hasil Evaluasi Metrik:
| $k$ (Jumlah Klaster) | WCSS / Inertia *(Elbow)* | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz |
| :---: | :---: | :---: | :---: | :---: |
| **2** | 78.35 | 0.3481 | 1.1191 | 18.08 |
| **3** | **56.26** *(Siku Penurunan)* | **0.3270** | **0.9905** *(Drop < 1.0)* | 16.80 |
| **4** | 42.15 | 0.3155 | 1.0266 | 16.89 |
| **5** | 34.16 | 0.3426 | 0.8911 | 16.24 |
| **6** | 28.37 | 0.3305 | 0.8196 | 15.79 |

### Alasan Keputusan Memilih $k = 3$:
1. **Titik Siku Pertama (*Elbow Point*)**: Penurunan WCSS paling signifikan terjadi pada perpindahan dari $k=2$ ke $k=3$ (berkurang $28.2\%$).
2. **Pemisahan Klaster Solid**: Davies-Bouldin Index turun tajam dari $1.119$ ke **$0.9905$** (di bawah 1.0 menandakan batas antarklaster yang tegas).
3. **Paritas & Interpretasi Kebijakan**: Untuk 27 kabupaten/kota, pembagian menjadi 3 kelompok menghasilkan interpretasi kebijakan yang paling aplikatif dan seimbang tanpa risiko *over-fragmentation*.
* **Output Grafik & Data**: [`clustering/statistik_grafik/grafik_eda/evaluasi_optimal_k.png`](file:///clustering/statistik_grafik/grafik_eda/evaluasi_optimal_k.png) dan [`evaluasi_optimal_k.csv`](file:///clustering/statistik_grafik/evaluasi_optimal_k.csv).

---

## Tahap 6: Eksekusi K-Means Clustering & Profiling Wilayah

Dijalankan melalui skrip [`clustering/kmeans_clustering.py`](file:///clustering/kmeans_clustering.py).

### Algoritma K-Means:
1. Menentukan $k=3$ centroid awal secara acak terdistribusi (`k-means++`, `random_state=42`).
2. Menghitung jarak Euclidean setiap kabupaten/kota ke masing-masing centroid.
3. Menetapkan setiap wilayah ke klaster dengan jarak centroid terdekat:
   $$C(i) = \arg\min_{j} ||x_i - \mu_j||^2$$
4. Menghitung ulang posisi centroid sebagai rata-rata koordinat anggota barunya:
   $$\mu_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i$$
5. Mengulang iterasi hingga konvergensi tercapai (posisi centroid tidak lagi bergeser).

---

### Profil Karakteristik 3 Klaster Terbentuk

Berikut adalah tabel rata-rata nilai riil (asli) tiap klaster yang disimpan pada [`clustering/statistik_grafik/profil_klaster_smp.csv`](file:///clustering/statistik_grafik/profil_klaster_smp.csv):

| Fitur / Indikator | Klaster 1: Wilayah Maju | Klaster 2: Wilayah Darurat Sarpras | Klaster 3: Wilayah Rentan & Terpencil | Rerata Jabar |
| :--- | :---: | :---: | :---: | :---: |
| **Jumlah Wilayah** | **7 daerah** | **5 daerah** | **15 daerah** | 27 daerah |
| **Rerata TKA Matematika** | **43.67** *(Tertinggi)* | **39.98** *(Menengah)* | **39.36** *(Terendah)* | 40.59 |
| **Rerata TKA B. Indonesia**| **69.49** *(Tertinggi)* | **61.85** *(Menengah)* | **60.71** *(Terendah)* | 63.19 |
| **Rasio Murid per Guru** | **18.76** *(Proporsional)* | **21.59** *(Paling Padat)* | **17.23** *(Relatif Longgar)* | 18.43 |
| **Rasio Murid per Sekolah**| **349.79** | **359.19** *(Terbesar/Sesak)* | **291.37** *(Sekolah Kecil)* | 319.06 |
| **% Penduduk Miskin** | **5.10%** *(Terendah)* | **6.32%** *(Moderat)* | **9.18%** *(Tertinggi)* | 7.59% |

---

### Penjelasan & Anggota Masing-Masing Klaster

#### 1. Klaster 1: Wilayah Maju (7 Daerah)
* **Anggota**: *Kota Bandung, Kota Cimahi, Kota Bekasi, Kota Bogor, Kota Depok, Kota Sukabumi, Kota Cirebon*.
* **Karakteristik**:
  * Didominasi oleh pusat-pusat kota metropolitan.
  * Capaian prestasi TKA Matematika dan Bahasa Indonesia tertinggi di Jawa Barat.
  * Daya dukung ekonomi sangat kuat dengan persentase penduduk miskin paling rendah di Jawa Barat (Depok hanya 2.31%, Bandung 3.78%, Bekasi 3.96%).
  * Rasio murid-guru dan infrastruktur sekolah berada pada standar pelayanan minimal yang sangat mapan.

#### 2. Klaster 2: Wilayah Darurat Sarana Prasarana (5 Daerah)
* **Anggota**: *Kabupaten Bogor, Kabupaten Bandung, Kabupaten Purwakarta, Kabupaten Karawang, Kabupaten Bekasi*.
* **Karakteristik**:
  * Wilayah dengan jumlah populasi siswa sangat besar (koridor penyangga industri).
  * **Krisis Guru Terberat**: Memiliki rasio murid per guru tertinggi di Jawa Barat (**21.59**, Karawang bahkan mencapai **24.23** dan Bogor **23.00**; jauh melampaui batas SPM ideal Kemendikbud 1:15 s/d 1:18).
  * **Sekolah Mengalami Overcapacity**: Rasio murid per sekolah mencapai rata-rata **359.19 murid per sekolah** (Karawang mencapai **442.23**). Ruang kelas sangat padat dan ketersediaan sekolah tidak mampu mengimbangi membludaknya jumlah usia sekolah.

#### 3. Klaster 3: Wilayah Rentan dan Terpencil (15 Daerah)
* **Anggota**: *Kabupaten Sukabumi, Cianjur, Garut, Tasikmalaya, Ciamis, Kuningan, Cirebon, Majalengka, Sumedang, Indramayu, Subang, Bandung Barat, Pangandaran, Kota Tasikmalaya, Kota Banjar*.
* **Karakteristik**:
  * **Tantangan Kemiskinan**: Memiliki tingkat kemiskinan tertinggi di Jawa Barat (rata-rata **9.18%**, di mana Indramayu mencapai **11.02%**, Kuningan **10.74%**, dan Tasikmalaya **10.15%**).
  * **Tantangan Dispersi Geografis / Sekolah Terpencil**: Rasio murid per sekolah kecil (rata-rata hanya 291 murid, bahkan di Tasikmalaya 205, Cianjur 219, Sukabumi 223, Pangandaran 238). Ini menandakan banyaknya sekolah-sekolah kecil yang tersebar di wilayah geografis pegunungan dan pesisir (wilayah Jabar Selatan dan pedalaman) dengan tantangan aksesibilitas.
  * **Kesenjangan Capaian**: Rerata nilai TKA terendah di Jawa Barat.

---

## Tahap 7: Analisis Inferensial Hubungan Variabel Pendukung (ANOVA & Regresi)

Dijalankan melalui skrip [`clustering/statistik_grafik/analisis_variabel_pendukung.py`](file:///clustering/statistik_grafik/analisis_variabel_pendukung.py). Tahap ini bertujuan untuk memvalidasi secara ilmiah signifikansi pengaruh variabel pendukung terhadap capaian akademik, serta menguji apakah perbedaan antarklaster signifikan secara statistik.

### 1. Korelasi & Regresi Linear Sederhana
Pengujian hubungan bivariat terhadap target `tka_rerata_mtk`:
* **Persentase Kemiskinan**:
  * Pearson $r = -0.622$ ($p = 0.0005$, **sangat signifikan** pada $\alpha = 0.001$).
  * Spearman $\rho = -0.579$ ($p = 0.0015$).
  * Slope regresi $= -0.5182$, dengan Koefisien Determinasi $R^2 = 0.387$ ($38.7\%$ variasi capaian TKA dapat dijelaskan oleh tingkat kemiskinan).
* **Rasio Murid per Sekolah**:
  * Pearson $r = 0.457$ ($p = 0.0167$, **signifikan** pada $\alpha = 0.05$).
  * $R^2 = 0.208$ ($20.8\%$ variasi TKA berhubungan positif dengan ukuran/kepadatan sekolah).
* **Rasio Murid per Guru**:
  * Pearson $r = 0.065$ ($p = 0.7467$, tidak berkorelasi linier langsung secara bivariat).

### 2. Uji Beda Rerata Signifikan Antar-Klaster
* **One-Way ANOVA**: $F = 10.541$ ($p = 0.0005 < 0.001$).
* **Kruskal-Wallis Test**: $H = 13.614$ ($p = 0.0011 < 0.01$).
* **Kesimpulan**: Perbedaan nilai TKA antarklaster **terbukti nyata secara statistik** dan bukan akibat kebetulan sampling.
* **Output Grafik & Data**: [`clustering/statistik_grafik/korelasi_tka_variabel_pendukung.csv`](file:///clustering/statistik_grafik/korelasi_tka_variabel_pendukung.csv), [`grafik_eda/heatmap_korelasi.png`](file:///clustering/statistik_grafik/grafik_eda/heatmap_korelasi.png), [`grafik_eda/scatter_regresi_tka.png`](file:///clustering/statistik_grafik/grafik_eda/scatter_regresi_tka.png), dan [`grafik_eda/boxplot_tka_per_klaster.png`](file:///clustering/statistik_grafik/grafik_eda/boxplot_tka_per_klaster.png).

---

## Tahap 8: Penyusunan Formula & Bobot Indeks Prioritas Intervensi (IPI)

Dijalankan melalui skrip [`clustering/indeks_prioritas.py`](file:///clustering/indeks_prioritas.py).

### 1. Filosofi & Urgensi Indeks
Klasterisasi K-Means mengelompokkan daerah ke dalam tipe permasalahan yang serupa, namun **belum menyusun urutan prioritas penerima bantuan**. Jika anggaran bantuan pendidikan terbatas, dinas pendidikan memerlukan instrumen peringkat objektif (*ranking 1 sampai 27*) untuk menentukan wilayah mana yang harus diintervensi terlebih dahulu.

### 2. Formulasi Matematis Normalisasi Indikator
Seluruh indikator diselaraskan menggunakan **Min-Max Normalization** agar memiliki arah yang seragam: **nilai $1.0$ menunjukkan kondisi paling darurat/mendesak**, dan **nilai $0.0$ menunjukkan kondisi paling aman/mandiri**.

* **Kriteria Biaya / Urgensi (*Cost Criteria*)**:
  Variabel yang makin tinggi nilainya makin mendesak butuh intervensi:
  $$I_{\text{Miskin}} = \frac{\text{pct\_miskin} - \text{pct\_miskin}_{\min}}{\text{pct\_miskin}_{\max} - \text{pct\_miskin}_{\min}}$$
  $$I_{\text{Guru}} = \frac{\text{rasio\_guru} - \text{rasio\_guru}_{\min}}{\text{rasio\_guru}_{\max} - \text{rasio\_guru}_{\min}}$$
  $$I_{\text{Sekolah}} = \frac{\text{rasio\_sekolah} - \text{rasio\_sekolah}_{\min}}{\text{rasio\_sekolah}_{\max} - \text{rasio\_sekolah}_{\min}}$$

* **Kriteria Manfaat (*Benefit Criteria*)**:
  Nilai TKA yang rendah mencerminkan urgensi pemulihan mutu pembelajaran yang tinggi (*Defisit TKA*):
  $$I_{\text{Defisit TKA}} = \frac{\text{TKA}_{\max} - \text{TKA}_i}{\text{TKA}_{\max} - \text{TKA}_{\min}}$$

### 3. Skema Pembobotan Komposit
Formula komposit Indeks Prioritas Intervensi (IPI) berskala $0 - 100$:

$$\text{IPI}_i = \left( w_1 \cdot I_{\text{Defisit TKA}} + w_2 \cdot I_{\text{Miskin}} + w_3 \cdot I_{\text{Guru}} + w_4 \cdot I_{\text{Sekolah}} \right) \times 100$$

| Indikator | Opsi A: Kebijakan Seimbang (Utama) | Opsi B: Afirmasi Mutu & Kemiskinan | Opsi C: Teori Informasi (Entropy Weight) |
| :--- | :---: | :---: | :---: |
| **Defisit Capaian TKA** ($w_1$) | **30%** ($0.30$) | **35%** ($0.35$) | **26.6%** ($0.266$) |
| **Persentase Kemiskinan** ($w_2$) | **30%** ($0.30$) | **35%** ($0.35$) | **21.1%** ($0.211$) |
| **Rasio Murid per Guru** ($w_3$) | **20%** ($0.20$) | **15%** ($0.15$) | **26.0%** ($0.260$) |
| **Rasio Murid per Sekolah** ($w_4$)| **20%** ($0.20$) | **15%** ($0.15$) | **26.3%** ($0.263$) |
| **Justifikasi Pemilihan** | Menyeimbangkan aspek mutu ($30\%$), konteks sosial ($30\%$), dan sarana/SDM ($40\%$). | Fokus pada penanggulangan ketertinggalan belajar dan kemiskinan ekstrem. | Ditentukan murni dari dispersi variansi data (*data-driven*). |

---

### 4. Hasil Peringkat Prioritas Intervensi 27 Kabupaten/Kota Jawa Barat

Hasil komputasi lengkap disimpan pada [`dataset/indeks_prioritas_intervensi_smp.csv`](file:///dataset/indeks_prioritas_intervensi_smp.csv):

| Peringkat | Kabupaten / Kota | Skor IPI (0-100) | Kategori Prioritas | Klaster K-Means | Faktor Pemicu Utama Intervensi |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **Kab. Karawang** | **78.59** | 🔴 **Prioritas 1 (Mendesak)** | Wilayah Maju (Agregat) | Krisis rasio guru (24.2) & sekolah terpadat (442 murid/sekolah) |
| **2** | **Kab. Indramayu** | **74.40** | 🔴 **Prioritas 1 (Mendesak)** | Darurat Sarpras | Kemiskinan tertinggi (11.02%) & TKA sangat rendah (38.32) |
| **3** | **Kab. Cirebon** | **72.23** | 🔴 **Prioritas 1 (Mendesak)** | Darurat Sarpras | Kemiskinan tinggi (10.23%) & beban sekolah padat (375 murid) |
| **4** | **Kab. Subang** | **68.43** | 🟠 **Prioritas 2 (Tinggi)** | Darurat Sarpras | Kemiskinan 9.23% & ukuran sekolah padat (356 murid) |
| **5** | **Kab. Majalengka** | **67.68** | 🟠 **Prioritas 2 (Tinggi)** | Darurat Sarpras | Kemiskinan 10.31% & TKA di bawah rata-rata (39.29) |
| **6** | **Kab. Cianjur** | **67.45** | 🟠 **Prioritas 2 (Tinggi)** | Rentan & Terpencil | Capaian TKA terendah se-Jabar (37.59) & kemiskinan 9.82% |
| **7** | **Kab. Bandung Barat** | **65.28** | 🟠 **Prioritas 2 (Tinggi)** | Darurat Sarpras | Kemiskinan 9.87% & rasio guru padat (19.39) |
| **8** | **Kab. Purwakarta** | **64.61** | 🟠 **Prioritas 2 (Tinggi)** | Darurat Sarpras | Beban guru padat (20.25) & TKA rendah (39.41) |
| **9** | **Kab. Garut** | **62.96** | 🟠 **Prioritas 2 (Tinggi)** | Rentan & Terpencil | Kemiskinan 9.39% & TKA rendah (38.48) |
| **10** | **Kota Tasikmalaya** | **60.47** | 🟠 **Prioritas 2 (Tinggi)** | Darurat Sarpras | Kemiskinan perkotaan tertinggi (10.84%) |
| **11** | Kab. Kuningan | 58.94 | 🟡 Prioritas 3 (Menengah) | Darurat Sarpras | Kemiskinan 10.74% |
| **12** | Kab. Bandung | 58.89 | 🟡 Prioritas 3 (Menengah) | Wilayah Maju (Agregat) | Sekolah padat (387 murid) & rasio guru 20.11 |
| **13** | Kab. Bogor | 57.93 | 🟡 Prioritas 3 (Menengah) | Wilayah Maju (Agregat) | Rasio guru sangat padat (23.00) |
| **14** | Kab. Sumedang | 56.28 | 🟡 Prioritas 3 (Menengah) | Darurat Sarpras | Kemiskinan 8.81% |
| **15** | Kab. Tasikmalaya | 55.01 | 🟡 Prioritas 3 (Menengah) | Rentan & Terpencil | Kemiskinan 10.15% & TKA rendah (38.85) |
| **16** | Kab. Sukabumi | 52.77 | 🟡 Prioritas 3 (Menengah) | Rentan & Terpencil | TKA rendah (38.39) & tantangan geografis |
| **17** | Kab. Ciamis | 50.88 | 🟡 Prioritas 3 (Menengah) | Rentan & Terpencil | TKA di bawah rata-rata (39.11) |
| **18** | Kota Sukabumi | 50.27 | 🟡 Prioritas 3 (Menengah) | Wilayah Maju (Agregat) | Rerata moderat |
| **19** | Kab. Bekasi | 47.38 | 🟡 Prioritas 3 (Menengah) | Wilayah Maju (Agregat) | Rasio guru padat (20.38) |
| **20** | Kab. Pangandaran | 43.86 | 🟡 Prioritas 3 (Menengah) | Rentan & Terpencil | Sekolah kecil tersebar di pesisir (238 murid) |
| **21** | Kota Cirebon | 42.03 | 🟡 Prioritas 3 (Menengah) | Darurat Sarpras | Kemiskinan perkotaan 8.66% |
| **22** | Kota Cimahi | 40.91 | 🟡 Prioritas 3 (Menengah) | Wilayah Maju (Agregat) | Kepadatan murid tinggi (443 murid/sekolah) |
| **23** | Kota Bogor | 39.03 | 🟢 Prioritas 4 (Mandiri) | Wilayah Maju (Agregat) | TKA tinggi (43.73), sarana mapan |
| **24** | Kota Banjar | 38.57 | 🟢 Prioritas 4 (Mandiri) | Rentan & Terpencil | Rasio guru sangat longgar (14.16), kemiskinan 5.73% |
| **25** | Kota Bandung | 29.75 | 🟢 Prioritas 4 (Mandiri) | Wilayah Maju (Agregat) | TKA unggul (44.33), kemiskinan rendah (3.78%) |
| **26** | Kota Bekasi | 25.34 | 🟢 Prioritas 4 (Mandiri) | Wilayah Maju (Agregat) | TKA unggul (44.71), kemiskinan rendah (3.96%) |
| **27** | Kota Depok | 20.04 | 🟢 Prioritas 4 (Mandiri) | Wilayah Maju (Agregat) | Kemiskinan terendah (2.31%), TKA tinggi (43.35) |

* **Output Grafik Visualisasi**: [`clustering/statistik_grafik/grafik_eda/ranking_prioritas_intervensi.png`](file:///clustering/statistik_grafik/grafik_eda/ranking_prioritas_intervensi.png).

---

## 11. Rekomendasi Kebijakan Berbasis Data (Policy Action)

Pemerintah Provinsi Jawa Barat dapat mengadopsi kerangka kerja intervensi berbasis **Klasterisasi (tipe intervensi)** dan **Indeks IPI (urutan alokasi anggaran)**:

```
+----------------------------------------------------------------------------------------------------+
|                                REKOMENDASI KEBIJAKAN PENDIDIKAN JABAR                              |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [ TIER 1: PRIORITAS 1 - MENDESAK (SKOR IPI >= 70) ]                                               |
|  - Target Wilayah: Karawang, Indramayu, Cirebon.                                                   |
|  - Alokasi Anggaran Darurat: Pembangunan Ruang Kelas Baru (RKB), rekrutmen guru PPPK prioritas,    |
|    dan intervensi BOS Afirmatif pemenuhan gizi/transportasi bagi siswa keluarga miskin.            |
|                                                                                                    |
|  [ TIER 2: PRIORITAS 2 - TINGGI (SKOR IPI 60 - 69.9) ]                                             |
|  - Target Wilayah: Subang, Majalengka, Cianjur, Bandung Barat, Purwakarta, Garut, Kota Tasik.      |
|  - Intervensi Terarah: Pelatihan guru untuk pemulihan capaian literasi-numerasi (Cianjur/Garut)   |
|    dan redistribusi guru antarsekolah untuk mengurai ketimpangan beban mengajar.                   |
|                                                                                                    |
|  [ TIER 3: PRIORITAS 3 - MENENGAH (SKOR IPI 40 - 59.9) ]                                           |
|  - Target Wilayah: Kuningan, Sumedang, Kab. Tasikmalaya, Sukabumi, Ciamis, Pangandaran, dll.       |
|  - Penguatan Rutin: Pemeliharaan sarpras berkala dan digitalisasi bahan ajar sekolah pedesaan.     |
|                                                                                                    |
|  [ TIER 4: PRIORITAS 4 - MANDIRI (SKOR IPI < 40) ]                                                 |
|  - Target Wilayah: Kota Bandung, Kota Bekasi, Kota Depok, Kota Bogor, Kota Banjar.                 |
|  - Program Pengimbasan (Sister-School): Menjadi pusat inkubasi inovasi dan pendamping bagi         |
|    sekolah-sekolah di kabupaten sekitarnya.                                                        |
+----------------------------------------------------------------------------------------------------+
```

---

## 12. Struktur Direktori Proyek

```text
klasterisasi-wilayah-tka/
├── README.md                                        # Dokumentasi komprehensif data mining
├── clustering/
│   ├── standarisasi.py                              # Skrip Z-score normalization
│   ├── evaluasi_k.py                                # Skrip evaluasi k (Elbow, Silhouette, DBI)
│   ├── kmeans_clustering.py                         # Skrip pemodelan K-Means & visualisasi klaster
│   ├── indeks_prioritas.py                          # Skrip penghitungan formula IPI & ranking daerah
│   └── statistik_grafik/
│       ├── korelasi.py                              # Skrip evaluasi matriks korelasi Pearson
│       ├── statistik.py                             # Skrip statistik deskriptif & histogram
│       ├── analisis_variabel_pendukung.py           # Skrip inferensial (ANOVA, Kruskal-Wallis, Regresi)
│       ├── matriks_korelasi.csv                     # Tabel koefisien korelasi antarvariabel
│       ├── pasangan_korelasi_tinggi.csv             # Daftar variabel dengan r >= 0.85
│       ├── evaluasi_optimal_k.csv                   # Tabel metrik Elbow, Silhouette, DBI (k=2 s/d 8)
│       ├── profil_klaster_smp.csv                   # Tabel rata-rata nilai asli tiap klaster
│       ├── korelasi_tka_variabel_pendukung.csv       # Tabel korelasi Pearson, Spearman, & slope regresi
│       ├── peringkat_prioritas_intervensi.csv       # Tabel ranking skor IPI 27 kabupaten/kota
│       └── grafik_eda/
│           ├── matriks_korelasi.png                 # Heatmap korelasi
│           ├── evaluasi_optimal_k.png               # Grafik evaluasi Elbow, Silhouette, DBI
│           ├── visualisasi_klaster_kmeans.png       # 4-panel grafik hasil klasterisasi K-Means
│           ├── heatmap_korelasi.png                 # Heatmap korelasi antar-fitur analisis pendukung
│           ├── scatter_regresi_tka.png              # Scatter plot + garis tren regresi linier TKA
│           ├── boxplot_tka_per_klaster.png          # Boxplot sebaran nilai TKA per klaster
│           └── ranking_prioritas_intervensi.png     # Diagram batang peringkat IPI (P1 s/d P4)
└── dataset/
    ├── jabar_smp_merged.csv                         # Dataset hasil integrasi seluruh variabel
    ├── fitur_kandidat_kmeans_smp.csv                # Dataset kandidat fitur sebelum standardisasi
    ├── fitur_kmeans_smp_standardisasi.csv           # Dataset fitur hasil standardisasi Z-Score
    ├── hasil_klaster_kmeans_smp.csv                 # Dataset final berlabel klaster k=3 dan k=4
    ├── indeks_prioritas_intervensi_smp.csv          # Dataset lengkap skor IPI dan kategori prioritas
    ├── kabkota_jabar_keyed.geojson                  # Poligon spasial 27 kab/kota Jawa Barat
    ├── qc_dan_data_dictionary_smp.xlsx              # Data dictionary & quality control
    ├── TKA/                                         # Data mentah scraping TKA SMP/MTs
    ├── infrastruktur & sdm/                         # Data Dapodik guru, murid, sekolah
    └── sosio ekonomi/                               # Data BPS kemiskinan kab/kota
```

---

## 13. Panduan Menjalankan Skrip

Pastikan dependensi Python telah terpasang:
```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

Jalankan tahapan analisis secara berurutan:
```bash
# 1. Melakukan standardisasi fitur dengan Z-Score:
python clustering/standarisasi.py

# 2. Menghitung metrik evaluasi penentuan nilai k optimal (Elbow, Silhouette, DBI):
python clustering/evaluasi_k.py

# 3. Menjalankan klasterisasi K-Means, profiling wilayah, dan visualisasi klaster:
python clustering/kmeans_clustering.py

# 4. Menjalankan analisis inferensial (korelasi, regresi linear, uji ANOVA & Kruskal-Wallis):
python clustering/statistik_grafik/analisis_variabel_pendukung.py

# 5. Menghitung formula pembobotan Indeks Prioritas Intervensi (IPI) & ranking 27 daerah:
python clustering/indeks_prioritas.py
```
Seluruh artefak hasil analisis akan diperbarui secara otomatis pada direktori `dataset/` dan `clustering/statistik_grafik/`.

