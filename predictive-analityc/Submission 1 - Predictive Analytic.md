# Laporan Proyek Machine Learning — Prediksi Risiko Stroke

**Nama:** AZHAR ZUHRO  
**Tanggal:** April 2026

---

## Domain Proyek

Stroke merupakan salah satu penyebab kematian dan kecacatan jangka panjang terbesar di dunia. Menurut World Health Organization (WHO), stroke menyebabkan sekitar **5,5 juta kematian setiap tahun** dan menjadi penyebab kecacatan permanen nomor satu secara global [1]. Di Indonesia, prevalensi stroke terus meningkat — berdasarkan data Riset Kesehatan Dasar (Riskesdas) 2018, angkanya mencapai **10,9 per mil penduduk**, naik signifikan dari 7,0 per mil pada 2013 [2].

Tantangan utama dalam penanganan stroke adalah sifatnya yang mendadak dan membutuhkan penanganan dalam **waktu emas (*golden hour*)**. Setiap menit keterlambatan penanganan berkorelasi langsung dengan kerusakan jaringan otak yang semakin luas dan ireversibel. Oleh karena itu, deteksi dini terhadap pasien berisiko tinggi menjadi sangat penting karena dapat mencegah kejadian stroke melalui intervensi medis seperti pengendalian hipertensi, manajemen diabetes, dan perubahan gaya hidup.

Pendekatan *machine learning* telah terbukti mampu mengidentifikasi pola risiko kompleks dari data klinis dan demografis pasien. Penelitian oleh Asadi et al. (2014) menunjukkan bahwa model prediksi berbasis ML mampu mengklasifikasikan risiko stroke dengan akurasi hingga 86% menggunakan data angiografi [3]. Penelitian lain oleh Dritsas & Trigka (2022) membandingkan berbagai algoritma ML untuk prediksi stroke dan menemukan bahwa model ensemble seperti Random Forest dan XGBoost memberikan performa terbaik pada data klinis yang imbalanced [4].

**Mengapa masalah ini harus diselesaikan?**

1. Stroke dapat dicegah melalui modifikasi faktor risiko jika dideteksi sejak dini — hingga 80% kejadian stroke dapat dihindari dengan penanganan faktor risiko yang tepat.
2. Sistem prediksi berbasis ML dapat membantu tenaga medis di daerah terpencil melakukan skrining awal tanpa memerlukan peralatan diagnostik canggih.
3. Pengurangan biaya layanan kesehatan melalui pencegahan yang lebih tepat sasaran dibandingkan penanganan pasca-kejadian stroke yang jauh lebih mahal.

### Referensi

[1] World Health Organization. (2023). *Stroke, Cerebrovascular accident*. WHO Health Topics. https://www.who.int/health-topics/stroke

[2] Kementerian Kesehatan RI. (2019). *Laporan Nasional Riskesdas 2018*. Badan Penelitian dan Pengembangan Kesehatan. https://labdata.litbang.kemkes.go.id/images/download/laporan/RKD/2018/Laporan_Nasional_RKD2018_FINAL.pdf

[3] Asadi, H., Dowling, R., Yan, B., & Mitchell, P. (2014). Machine learning for outcome prediction of acute ischemic stroke post intra-arterial therapy. *PLOS ONE*, 9(2), e88225. https://doi.org/10.1371/journal.pone.0088225

[4] Dritsas, E., & Trigka, M. (2022). Stroke risk prediction with machine learning techniques. *Sensors*, 22(13), 4670. https://doi.org/10.3390/s22134670

---

## Business Understanding

### Problem Statements

1. **Pernyataan Masalah 1:** Bagaimana cara membangun model machine learning yang mampu memprediksi risiko stroke pada pasien berdasarkan data demografis dan klinis seperti usia, hipertensi, penyakit jantung, kadar glukosa darah, dan BMI?

2. **Pernyataan Masalah 2:** Algoritma machine learning mana yang memberikan **Recall dan ROC-AUC tertinggi** untuk meminimalkan kasus pasien berisiko stroke yang tidak terdeteksi (*false negative*)?

3. **Pernyataan Masalah 3:** Faktor risiko klinis dan demografis apa yang paling berkontribusi terhadap prediksi stroke berdasarkan model terbaik?

### Goals

1. **Jawaban Masalah 1:** Membangun model klasifikasi biner yang memprediksi apakah seorang pasien berisiko stroke (1 = stroke, 0 = tidak), dengan target **Recall ≥ 0.80** dan **ROC-AUC ≥ 0.83** pada data uji.

2. **Jawaban Masalah 2:** Membandingkan Logistic Regression, Random Forest, dan XGBoost secara sistematis, lalu memilih model dengan Recall dan ROC-AUC tertinggi sebagai prioritas dalam konteks deteksi dini medis.

3. **Jawaban Masalah 3:** Mengekstrak dan menganalisis *feature importance* dari model terbaik untuk memberikan insight klinis yang dapat digunakan oleh tenaga kesehatan.

### Solution Statements

**Solusi 1 — Baseline Logistic Regression:**  
Menggunakan Logistic Regression dengan `class_weight='balanced'` sebagai baseline yang mudah diinterpretasikan. Model dievaluasi menggunakan Accuracy, Precision, Recall, F1-Score, dan ROC-AUC. Target terukur: Recall ≥ 0.80 dan ROC-AUC ≥ 0.83 pada data uji.

**Solusi 2 — Ensemble Methods + SMOTE + Hyperparameter Tuning:**  
Menggunakan Random Forest (dengan SMOTE untuk menangani imbalance) dan XGBoost (dengan `scale_pos_weight` sebagai native class weighting) sebagai model ensemble yang lebih powerful. Model terbaik dari kedua pendekatan dioptimalkan lebih lanjut menggunakan `GridSearchCV` dengan 5-fold stratified cross-validation. Target terukur yang sama: **Recall ≥ 0.80** dan **ROC-AUC ≥ 0.83**.

---

## Data Understanding

- **Sumber Data:** [Stroke Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Jumlah Data:** 5.110 baris × 12 kolom
- **Kondisi Data:** Terdapat **201 missing values** pada kolom `bmi` (3,93%). Tidak ditemukan baris duplikat. Dataset sangat tidak seimbang (*highly imbalanced*): hanya **4,87% pasien** mengalami stroke (249 kasus) versus 95,13% tidak stroke (4.861 kasus).

### Variabel-Variabel pada Dataset

| No | Variabel | Tipe | Deskripsi |
|----|----------|------|-----------|
| 1 | `id` | Integer | ID unik pasien — tidak relevan secara prediktif, akan di-drop |
| 2 | `gender` | Kategorikal | Jenis kelamin (Male, Female, Other) |
| 3 | `age` | Float | Usia pasien dalam tahun |
| 4 | `hypertension` | Binary | Riwayat hipertensi (1 = ya, 0 = tidak) |
| 5 | `heart_disease` | Binary | Riwayat penyakit jantung (1 = ya, 0 = tidak) |
| 6 | `ever_married` | Kategorikal | Status pernikahan (Yes / No) |
| 7 | `work_type` | Kategorikal | Jenis pekerjaan (Private, Self-employed, Govt_job, children, Never_worked) |
| 8 | `Residence_type` | Kategorikal | Tipe tempat tinggal (Urban / Rural) |
| 9 | `avg_glucose_level` | Float | Rata-rata kadar glukosa darah (mg/dL) |
| 10 | `bmi` | Float | Indeks Massa Tubuh (kg/m²) — **201 missing values** |
| 11 | `smoking_status` | Kategorikal | Status merokok (formerly smoked, never smoked, smokes, Unknown) |
| 12 | `stroke` | Binary | **Target** — 1 = stroke, 0 = tidak stroke |

### Exploratory Data Analysis (EDA)

#### 1. Distribusi Target — Sangat Imbalanced

Dari 5.110 sampel, hanya 249 (4,87%) adalah kasus stroke. Ini merupakan tantangan utama yang harus ditangani secara eksplisit karena model naif yang selalu memprediksi "tidak stroke" pun bisa meraih akurasi 95% — metrik yang menyesatkan.

```python
print(df['stroke'].value_counts())
# 0    4861  (95.13%)
# 1     249  ( 4.87%)
```

#### 2. Distribusi Usia — Faktor Paling Dominan

Analisis KDE dan boxplot usia berdasarkan status stroke menunjukkan perbedaan yang sangat mencolok:

- **Rata-rata usia pasien tidak stroke:** 42,0 tahun
- **Rata-rata usia pasien stroke:** 67,7 tahun
- **Selisih:** 25,7 tahun — sangat signifikan secara klinis

```python
print(f"Rata-rata usia pasien TIDAK stroke : {df[df['stroke']==0]['age'].mean():.1f} tahun")
print(f"Rata-rata usia pasien STROKE       : {df[df['stroke']==1]['age'].mean():.1f} tahun")
# Output: 42.0 vs 67.7 tahun
```

#### 3. Fitur Kategorikal vs Target

Dari analisis crosstab proporsi stroke per kategori, ditemukan bahwa pasien dengan **hipertensi** dan **penyakit jantung** memiliki proporsi stroke yang jauh lebih tinggi. Proporsi stroke juga lebih tinggi pada pasien yang **pernah menikah** (berkorelasi kuat dengan usia). **Status merokok** menunjukkan bahwa pasien yang "formerly smoked" memiliki proporsi stroke tertinggi.

#### 4. Distribusi Fitur Numerik

Pasien stroke rata-rata memiliki kadar **`avg_glucose_level`** lebih tinggi, mengindikasikan hubungan erat antara diabetes dan risiko stroke. Median BMI keseluruhan dataset = **28,10** (overweight ringan) yang digunakan sebagai nilai imputasi.

#### 5. Matriks Korelasi

Hasil heatmap korelasi menunjukkan bahwa `age` memiliki korelasi positif tertinggi dengan target `stroke`, diikuti oleh `ever_married`, `hypertension`, `heart_disease`, dan `avg_glucose_level`.

---

## Data Preparation

Tahapan data preparation dilakukan secara berurutan. Urutan ini penting karena setiap tahap bergantung pada hasil tahap sebelumnya.

### 1. Drop Kolom `id`

```python
df = df.drop(columns=['id'])
# Hasil: shape (5110, 11)
```

**Alasan:** Kolom `id` adalah pengenal administratif yang tidak memiliki nilai prediktif. Menyertakannya dalam model dapat menyebabkan *data leakage* atau overfitting terhadap ID yang tidak bermakna.

### 2. Hapus Baris Gender 'Other'

```python
df = df[df['gender'] != 'Other']
# Hasil: shape (5109, 11) — 1 baris terhapus
```

**Alasan:** Hanya terdapat 1 baris dengan nilai 'Other' — terlalu sedikit untuk membentuk kategori tersendiri yang representatif. Mempertahankan kategori ini berpotensi menyebabkan error saat encoding dan mengganggu generalisasi model.

### 3. Imputasi Missing Values pada BMI

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
df['bmi'] = imputer.fit_transform(df[['bmi']])
# Nilai median BMI yang digunakan: 28.10
```

**Alasan:** Sebanyak 201 baris (3,93%) memiliki nilai BMI kosong. Dipilih strategi **median** (bukan mean) karena lebih robust terhadap outlier pada distribusi BMI yang umumnya memiliki ekor kanan yang panjang. Baris tidak dihapus karena akan semakin mengurangi kelas minoritas (pasien stroke) yang sudah sangat sedikit.

### 4. Encoding Variabel Kategorikal

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Label Encoding: fitur binary
df['ever_married']   = le.fit_transform(df['ever_married'])    # Yes=1, No=0
df['Residence_type'] = le.fit_transform(df['Residence_type'])  # Urban=1, Rural=0

# One-Hot Encoding: fitur multi-kelas
df = pd.get_dummies(df, columns=['gender', 'work_type', 'smoking_status'], drop_first=True)
# Hasil: shape (5109, 16) — 15 fitur + 1 target
```

**Alasan:** Fitur dengan dua kategori biner cukup dengan Label Encoding. Fitur multi-kelas (`gender`, `work_type`, `smoking_status`) menggunakan One-Hot Encoding untuk menghindari asumsi ordinal yang salah. Parameter `drop_first=True` digunakan untuk menghindari multikolinearitas (*dummy variable trap*).

### 5. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Training: (4087, 15) | Stroke: 199 (4.9%)
# Test    : (1022, 15) | Stroke:  50 (4.9%)
```

**Alasan:** Rasio 80:20 memberikan data training yang cukup sekaligus data uji yang representatif. Parameter `stratify=y` memastikan proporsi kelas stroke yang sama (~4,9%) di kedua set, mencegah bias evaluasi akibat distribusi kelas yang tidak seimbang di data uji.

### 6. Feature Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
# Mean ≈ 0.0000, Std ≈ 1.0000 — standardisasi berhasil
```

**Alasan:** Fitur memiliki skala yang sangat berbeda — `age` (0–82 tahun), `avg_glucose_level` (55–272 mg/dL), dan `bmi` (10–97). Tanpa scaling, fitur berskala besar akan mendominasi model berbasis gradien. `fit` hanya dilakukan pada data training dan hasilnya di-`transform` ke test set untuk **mencegah data leakage**.

### 7. SMOTE untuk Penanganan Class Imbalance

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
# Sebelum: {0: 3888, 1: 199}
# Setelah : {0: 3888, 1: 3888} → total 7.776 sampel
```

**Alasan:** Rasio 95%:5% sangat ekstrem. SMOTE (*Synthetic Minority Over-sampling Technique*) menyeimbangkan kelas dengan membuat sampel sintetis kelas minoritas berdasarkan interpolasi k-nearest neighbors. Diterapkan **hanya pada training data** untuk mencegah data leakage ke test set. SMOTE hanya digunakan untuk Random Forest — XGBoost menggunakan `scale_pos_weight` sebagai alternatif native yang terbukti lebih efektif pada eksperimen ini.

---

## Modeling

### Model 1: Logistic Regression (Baseline)

```python
lr_model = LogisticRegression(
    class_weight='balanced', random_state=42, max_iter=1000
)
lr_model.fit(X_train_scaled, y_train)
```

Logistic Regression dipilih sebagai **baseline** karena kesederhanaannya dan kemampuan interpretasinya. Parameter `class_weight='balanced'` secara otomatis menyesuaikan bobot tiap kelas berdasarkan frekuensinya — kelas minoritas mendapatkan bobot ~19,54×.

**Kelebihan:** Cepat dilatih, mudah diinterpretasikan melalui koefisien, tidak memerlukan tuning yang kompleks, bekerja baik dengan `class_weight='balanced'` untuk dataset imbalanced.

**Kekurangan:** Mengasumsikan hubungan linier antara fitur dan log-odds target. Tidak efektif untuk menangkap interaksi fitur yang kompleks atau hubungan non-linier yang mungkin ada dalam data klinis.

### Model 2: Random Forest

```python
rf_model = RandomForestClassifier(
    n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1
)
rf_model.fit(X_train_res, y_train_res)  # dilatih pada data SMOTE
```

Random Forest adalah algoritma ensemble berbasis bagging yang melatih banyak decision tree secara paralel dan menggabungkan prediksinya. Model ini dilatih pada data yang sudah di-resample dengan SMOTE.

**Kelebihan:** Robust terhadap outlier, mampu menangkap hubungan non-linier dan interaksi fitur yang kompleks, menghasilkan feature importance, tidak sensitif terhadap scaling.

**Kekurangan:** Pada kasus ini, kombinasi SMOTE + Random Forest menghasilkan Recall yang sangat rendah (0,16). SMOTE menciptakan distribusi training yang terlalu jauh berbeda dari distribusi data nyata — RF overfit terhadap pola sintetis dan gagal menggeneralisasi ke distribusi test set asli.

### Model 3: XGBoost

```python
ratio = 19.54  # rasio kelas 0:1 aktual dari data training

xgb_model = XGBClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=6,
    scale_pos_weight=ratio, random_state=42, eval_metric='auc', n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)
```

XGBoost (*Extreme Gradient Boosting*) adalah algoritma ensemble berbasis boosting yang membangun tree secara sekuensial, di mana setiap tree baru belajar dari kesalahan tree sebelumnya. Parameter `scale_pos_weight=19.54` memberikan bobot lebih tinggi pada kelas minoritas secara native, tanpa perlu resampling.

**Kelebihan:** Regularisasi L1/L2 built-in yang mengurangi overfitting, mendukung penanganan imbalance native via `scale_pos_weight`, umumnya memberikan performa state-of-the-art pada data tabular.

**Kekurangan:** Memiliki banyak hyperparameter yang perlu disetel, sensitif terhadap learning rate, dan lebih sulit diinterpretasikan dibanding Logistic Regression.

### Hyperparameter Tuning — Model Terbaik: XGBoost

Setelah evaluasi awal menunjukkan XGBoost memiliki potensi terbaik, dilakukan optimasi lebih lanjut menggunakan GridSearchCV:

```python
param_grid = {
    'n_estimators'    : [100, 200, 300],
    'max_depth'       : [3, 5, 6],
    'learning_rate'   : [0.01, 0.05, 0.1],
    'subsample'       : [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(scale_pos_weight=ratio, random_state=42,
                            eval_metric='auc', n_jobs=-1),
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc', n_jobs=-1, verbose=1
)
# Fitting 5 folds × 108 candidates = 540 fits
```

**Best Parameters yang ditemukan:**

```
{'colsample_bytree': 1.0, 'learning_rate': 0.01, 'max_depth': 3,
 'n_estimators': 300, 'subsample': 0.8}
Best ROC-AUC (CV): 0.8347
```

**Mengapa XGBoost (Tuned) dipilih sebagai model final?**

XGBoost (Tuned) dipilih karena kombinasi ROC-AUC **0.8446** (tertinggi) dan Recall **0.8200** (tertinggi) — dua metrik paling krusial untuk deteksi dini stroke. Logistic Regression mendekati performa ini (Recall 0,80, ROC-AUC 0,8395) tetapi tidak melebihinya pada kedua metrik sekaligus. Random Forest gagal total dengan Recall hanya 0,16 karena overfitting terhadap distribusi SMOTE yang sintetis.

---

## Evaluation

### Metrik Evaluasi

Lima metrik digunakan untuk mengevaluasi model secara komprehensif:

**1. Accuracy**

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Proporsi prediksi benar dari keseluruhan data. **Kurang representatif** untuk dataset imbalanced 95%:5% — model yang selalu memprediksi "tidak stroke" pun bisa mencapai Accuracy 95%.

**2. Precision**

$$\text{Precision} = \frac{TP}{TP + FP}$$

Dari semua yang diprediksi stroke, berapa yang benar-benar stroke. Precision rendah berarti banyak *false alarm* (pasien sehat dirujuk untuk pemeriksaan yang tidak diperlukan).

**3. Recall (Sensitivity) — Metrik Prioritas**

$$\text{Recall} = \frac{TP}{TP + FN}$$

Dari semua pasien yang benar-benar stroke, berapa yang berhasil terdeteksi. **Metrik terpenting secara klinis** — *false negative* (stroke terlewat) dapat berakibat fatal karena pasien kehilangan kesempatan intervensi dini.

**4. F1-Score**

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

*Harmonic mean* dari Precision dan Recall. Berguna untuk mengevaluasi performa keseluruhan pada dataset imbalanced, namun F1 yang tinggi tidak selalu menjamin Recall yang cukup tinggi untuk konteks medis.

**5. ROC-AUC (Area Under the ROC Curve)**

$$\text{ROC-AUC} = \int_0^1 TPR(FPR)\, d(FPR)$$

Mengukur kemampuan diskriminasi model di berbagai threshold keputusan. ROC-AUC = 1,0 berarti klasifikasi sempurna; = 0,5 berarti setara prediksi acak. Sangat tepat untuk dataset imbalanced karena tidak dipengaruhi dominasi kelas mayoritas.

---

### Hasil Evaluasi

**Tabel Perbandingan Performa Model pada Data Uji:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 0.7387 | 0.1347 | 0.8000 | 0.2305 | 0.8395 |
| Random Forest | 0.9207 | 0.1702 | 0.1600 ❌ | 0.1649 | 0.7577 |
| XGBoost | 0.8894 | 0.1942 | 0.4000 | 0.2614 | 0.7902 |
| **XGBoost (Tuned)** | **0.6977** | **0.1202** | **0.8200 ✅** | **0.2097** | **0.8446** |

**Cross-Validation 5-Fold — XGBoost (Tuned):**

| Fold | ROC-AUC | Recall |
|:----:|:-------:|:------:|
| Fold 1 | 0.8401 | 0.8250 |
| Fold 2 | 0.8766 | 0.8750 |
| Fold 3 | 0.8489 | 0.8462 |
| Fold 4 | 0.7923 | 0.7750 |
| Fold 5 | 0.8156 | 0.8000 |
| **Mean** | **0.8347 ± 0.0288** | **0.8242 ± 0.0348** |

**Classification Report — XGBoost (Tuned):**

| Kelas | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| Tidak Stroke (0) | 0.99 | 0.69 | 0.81 | 972 |
| Stroke (1) | 0.12 | 0.82 | 0.21 | 50 |
| **Weighted avg** | **0.94** | **0.70** | **0.78** | **1022** |

---

### Analisis Hasil

**1. Model Terbaik: XGBoost (Tuned)**

Dengan ROC-AUC **0.8446** dan Recall **0.8200**, dari **50 kasus stroke nyata** di test set, model berhasil mendeteksi **41 kasus (82%)** — hanya 9 yang terlewat. Target Recall ≥ 0.80 dan ROC-AUC ≥ 0.83 keduanya tercapai.

**2. Confusion Matrix Aktual (dari data uji):**

|  | Prediksi: Tidak Stroke | Prediksi: Stroke |
|--|:---------------------:|:----------------:|
| **Aktual: Tidak Stroke (972)** | TN = 672 ✅ | FP = 300 ⚠️ |
| **Aktual: Stroke (50)** | FN = 9 ❌ | TP = 41 ✅ |

**3. Mengapa Accuracy 69.77% Bukan Masalah?**

Accuracy rendah adalah **konsekuensi yang disengaja dan benar** dari `scale_pos_weight=19.54`. Model sengaja dikorbankan akurasinya untuk mendapatkan Recall tinggi — strategi yang tepat secara klinis. Mengirim 300 pasien sehat untuk pemeriksaan lanjutan (FP) jauh lebih baik daripada membiarkan 9 pasien stroke tanpa peringatan (FN) yang dapat berakhir dengan kecacatan permanen atau kematian.

**4. Temuan Kritis: Random Forest Gagal Total (Recall 0.16)**

Random Forest yang dilatih pada data SMOTE hanya mampu mendeteksi 16% kasus stroke. Penyebabnya: SMOTE menciptakan distribusi training yang sangat berbeda dari test set nyata (50 dari 1.022 = 4,9%), sehingga RF overfit terhadap pola sintetis. **Pelajaran kritis:** untuk imbalance ekstrem (>90%:10%), native class weighting (`scale_pos_weight`) lebih unggul daripada resampling.

**5. Feature Importance Aktual — Top 5 (XGBoost Tuned):**

| Rank | Fitur | Importance Score | Insight Klinis |
|:----:|-------|:----------------:|----------------|
| 1 | `age` | **0.3309 (33,09%)** | Usia paling dominan; rata-rata stroke 67,7 tahun vs non-stroke 42,0 tahun |
| 2 | `smoking_status_never smoked` | 0.0950 (9,50%) | Status merokok sebagai pembeda risiko signifikan |
| 3 | `gender_Male` | 0.0666 (6,66%) | Pria memiliki risiko stroke lebih tinggi |
| 4 | `bmi` | 0.0657 (6,57%) | Obesitas sebagai faktor risiko kardiovaskular |
| 5 | `avg_glucose_level` | 0.0656 (6,56%) | Indikator diabetes yang erat hubungannya dengan stroke |

---

### Error Analysis — Mengapa Model Masih Salah Prediksi?

**1. False Positive (FP) — 300 kasus — "Sehat diprediksi Stroke"**

Berdasarkan profil rata-rata aktual dari data uji, pasien FP memiliki rata-rata usia **66,17 tahun**, glukosa **131,02 mg/dL**, BMI 30,95, dan 28% memiliki hipertensi. Profil demografis ini sangat mirip pasien stroke nyata sehingga model yang sangat sensitif (Recall 0,82) memang cenderung menghasilkan lebih banyak *false alarm*. Dampak: pasien menjalani pemeriksaan lanjutan yang tidak diperlukan, namun ini adalah **trade-off yang dapat diterima** untuk sistem skrining awal — bukan alat diagnosis final.

**2. False Negative (FN) — 9 kasus — "Stroke tidak terdeteksi"**

Profil aktual pasien FN sangat berbeda dari pasien stroke yang berhasil terdeteksi (TP): rata-rata usia FN hanya **39,44 tahun** dibandingkan TP yang **74,24 tahun**, dengan glukosa rata-rata jauh lebih rendah (83,02 mg/dL vs 149,02 mg/dL). Karena `age` mendominasi model (33,09%), pasien stroke yang lebih muda dengan glukosa normal dinilai berisiko rendah. Contoh nyata dari data uji: pasien berusia 32 tahun dengan glukosa 76,13 mg/dL hanya mendapatkan probabilitas stroke **0,03** — padahal benar-benar stroke.

Stroke pada kelompok usia muda kemungkinan dipicu oleh faktor yang tidak tersedia dalam dataset: kelainan pembekuan darah, diseksi arteri karotis, atau fibrilasi atrium yang belum terdiagnosis.

**Profil perbandingan aktual FP vs FN vs TP:**

| Fitur | False Positive (FP) | False Negative (FN) | True Positive (TP) |
|-------|:-------------------:|:-------------------:|:------------------:|
| Usia (tahun) | 66,17 | **39,44** | 74,24 |
| Glukosa (mg/dL) | 131,02 | **83,02** | 149,02 |
| BMI | 30,95 | 31,64 | 29,55 |
| Hipertensi | 28% | 0% | 29% |
| Penyakit Jantung | 9% | 0% | 27% |

**3. Keterbatasan Kritis Dataset**

Dataset tidak mencakup beberapa prediktor stroke yang terbukti secara klinis: kadar kolesterol LDL/HDL, riwayat TIA (*Transient Ischemic Attack*), fibrilasi atrium, penggunaan kontrasepsi hormonal, dan riwayat keluarga. Tanpa fitur-fitur ini, model secara struktural "buta" terhadap sebagian pasien stroke berisiko tinggi — khususnya kelompok usia muda.

**4. Kualitas Data: `smoking_status = Unknown` (~30% data)**

Meskipun status merokok masuk top-2 feature importance (9,50%), sekitar 30% data memiliki status merokok yang tidak diketahui. Model tidak dapat mengekstrak sinyal akurat dari kelompok ini, berpotensi melemahkan prediksi untuk pasien dengan status merokok tidak diketahui.

**5. Analisis Threshold Optimal**

Dengan threshold default 0,5, model menghasilkan Recall 0,82 dan Precision 0,12 (FP = 300, FN = 9). Dari analisis kurva Precision-Recall, ditemukan **threshold optimal = 0,605** yang menghasilkan Recall 0,80 dengan Precision yang lebih baik (0,1653) — mengurangi FP namun mengorbankan 1 kasus stroke tambahan (FN menjadi 10).

```
# Output aktual dari notebook:
# Threshold default (0.50): Recall=0.8200, Precision=0.1202, FP=300, FN=9
# Threshold optimal (0.605): Recall=0.8000, Precision=0.1653, FN=10
```

Untuk implementasi klinis: gunakan threshold lebih rendah dari 0,5 jika kapasitas pemeriksaan lanjutan mencukupi (memaksimalkan Recall), atau gunakan 0,605 jika ingin mengurangi beban rujukan sambil tetap memenuhi target Recall 0,80.

**Rekomendasi Pengembangan:**
- Menambahkan fitur klinis kritis: kolesterol LDL/HDL, riwayat TIA, dan fibrilasi atrium
- Mencoba **LightGBM** atau **CatBoost** sebagai alternatif XGBoost yang lebih efisien
- Menggunakan **ADASYN** (*Adaptive Synthetic Sampling*) sebagai alternatif SMOTE yang lebih adaptif
- Melakukan optimasi threshold berdasarkan kapasitas fasilitas medis yang akan mengimplementasikan sistem

---

### Kesimpulan

Model **XGBoost dengan hyperparameter tuning** berhasil dipilih sebagai solusi akhir dengan performa terbaik: ROC-AUC **0.8446** dan Recall **0.8200**. Dari 50 kasus stroke nyata di data uji, model berhasil mendeteksi **41 kasus (82%)** — hanya 9 yang terlewat. Kedua target metrik (Recall ≥ 0.80 dan ROC-AUC ≥ 0.83) berhasil dicapai. Cross-validation 5-fold mengkonfirmasi konsistensi model dengan mean ROC-AUC 0.8347 ± 0.0288 dan mean Recall 0.8242 ± 0.0348.

Error analysis mengungkap temuan kritis: Random Forest yang dilatih dengan SMOTE justru gagal total (Recall 0,16) karena overfitting terhadap distribusi sintetis. Ini membuktikan bahwa untuk imbalance ekstrem (95%:5%), teknik native weighting (`scale_pos_weight`) lebih efektif daripada resampling. Analisis profil FN menunjukkan pasien yang terlewat rata-rata berusia **39,44 tahun** — jauh lebih muda dari pasien stroke yang terdeteksi (74,24 tahun) — mengindikasikan kebutuhan fitur klinis tambahan untuk mendeteksi stroke pada kelompok usia muda. Feature importance mengkonfirmasi **usia (33,09%)** sebagai faktor risiko stroke paling dominan, konsisten dengan literatur medis internasional.

Model ini berpotensi diimplementasikan sebagai sistem skrining awal risiko stroke di fasilitas kesehatan primer, dengan catatan penting bahwa keputusan diagnosis akhir tetap berada pada penilaian dokter yang kompeten dan tidak dapat digantikan sepenuhnya oleh sistem otomatis.

