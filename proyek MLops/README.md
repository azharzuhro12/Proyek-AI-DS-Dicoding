# Pipeline MLOps Analisis Sentimen dengan TFX, Docker, Prometheus & Grafana

Proyek ini menerapkan alur **MLOps (Machine Learning Operations)** menggunakan **TensorFlow Extended (TFX)** untuk mengotomatisasi proses pelatihan, validasi, deployment, hingga monitoring model analisis sentimen.

---

## Masalah & Tujuan

### Masalah
- Perusahaan teknologi menerima ribuan tweet dari pelanggan setiap hari, mulai dari komentar positif hingga keluhan produk (seperti baterai boros, error setelah update, atau layar rusak). Memisahkan keluhan pelanggan secara manual memakan banyak waktu.
- Dari sisi teknis, model machine learning yang hanya dijalankan secara manual di notebook sulit dipakai langsung di server produksi karena proses validasi data, pelatihan ulang, dan pengecekan kualitas model belum otomatis.

### Tujuan
- Membuat model klasifikasi sentimen (**Bidirectional LSTM**) untuk membedakan tweet positif dan tweet keluhan pelanggan.
- Mengotomatisasi seluruh tahapan pipeline menggunakan **TFX**, mulai dari pembacaan data, validasi skema, preprocessing teks, tuning parameter, hingga evaluasi model.
- Menjalankan model yang sudah lolos evaluasi sebagai REST API menggunakan **TensorFlow Serving (Docker)**.
- Memantau kondisi server model secara berkala menggunakan **Prometheus** dan **Grafana**.

---

## Dataset

Dataset yang digunakan adalah [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/dineshpiyasamara/sentiment-analysis-dataset) (`data/sentiment_analysis.csv`) yang berisi **7.920 tweet** pelanggan terkait produk teknologi (Apple, Sony, Samsung, Android):
- **Label 0 (Positif):** 5.894 data
- **Label 1 (Negatif / Keluhan):** 2.026 data
- Data dibagi menjadi **80% data latih** dan **20% data evaluasi**.

---

## Alur Pipeline TFX

Pipeline diatur pada file `pipeline.py` dan `modules/components.py` dengan urutan berikut:

1. **CsvExampleGen**: Membaca file CSV dan membagi data menjadi data latih (80%) dan evaluasi (20%).
2. **StatisticsGen & SchemaGen**: Menghitung statistik data dan membuat skema tipe data otomatis.
3. **ExampleValidator**: Mengecek apakah ada data kosong atau format data yang menyimpang dari skema.
4. **Transform (`modules/transform.py`)**: Mengubah teks tweet menjadi huruf kecil (*lowercase*) menggunakan `tensorflow_transform`.
5. **Tuner (`modules/tuner.py`)**: Mencari kombinasi hyperparameter terbaik menggunakan `RandomSearch`.
6. **Trainer (`modules/trainer.py`)**: Melatih model **Bidirectional LSTM** dan menyimpan model ke format `SavedModel`.
7. **Evaluator**: Menguji performa model menggunakan **TFMA**. Model hanya diloloskan jika akurasinya melewati batas minimum dan lebih baik dari model sebelumnya.
8. **Pusher**: Menyimpan model yang sudah lolos evaluasi ke folder `serving_model_dir/`.

---

## Hasil Evaluasi Model

Model **Bidirectional LSTM** yang dihasilkan memiliki performa sebagai berikut pada tahap evaluasi:
- **Binary Accuracy:** 97,16% (`0.9716`)
- **Loss:** `0.0897`
- **AUC:** `0.7300` (73%)

---

## Struktur Folder

```text
proyek MLops/
├── config/
│   └── prometheus.config          # Konfigurasi metrik Prometheus untuk TF Serving
├── data/
│   └── sentiment_analysis.csv     # Dataset tweet sentimen (7.920 baris)
├── modules/
│   ├── components.py              # Susunan komponen pipeline TFX
│   ├── transform.py               # Kode preprocessing teks
│   ├── tuner.py                   # Kode tuning hyperparameter
│   └── trainer.py                 # Kode arsitektur model Bi-LSTM & training
├── monitoring/
│   ├── Dockerfile                 # Dockerfile untuk Prometheus
│   └── prometheus.yml             # Konfigurasi scraping metrik TF Serving
├── serving_model_dir/             # Folder hasil model yang siap di-deploy
├── Dockerfile                     # Dockerfile untuk TensorFlow Serving
├── pipeline.py                    # Konfigurasi utama jalannya pipeline TFX
├── notebook.ipynb                 # Notebook untuk menjalankan pipeline TFX
├── check_tf_serving.ipynb         # Notebook untuk cek status server model
└── test_notebook.ipynb            # Notebook untuk tes kirim request prediksi ke API
```

---

## Cara Menjalankan

### 1. Menjalankan Pipeline TFX
Buka dan jalankan semua sel pada `notebook.ipynb` untuk memproses data hingga menghasilkan model di folder `serving_model_dir/`.

### 2. Menjalankan Model dengan Docker (TensorFlow Serving)
```bash
docker build -t sentiment-tf-serving .
docker run -d -p 8500:8500 -p 8501:8501 --name tf-serving sentiment-tf-serving
```

Cek apakah model sudah aktif:
```bash
curl http://localhost:8501/v1/models/cc-model
```

### 3. Menguji Prediksi API
Jalankan `test_notebook.ipynb` untuk mengirim contoh teks tweet ke endpoint `http://localhost:8501/v1/models/cc-model:predict`.

### 4. Menjalankan Monitoring Prometheus
```bash
cd monitoring
docker build -t tf-serving-prometheus .
docker run -d -p 9090:9090 --name prometheus tf-serving-prometheus
```
