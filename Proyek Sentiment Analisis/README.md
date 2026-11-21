# Analisis Sentimen Ulasan Aplikasi Shopee (Google Play Store)

Proyek ini berisi analisis sentimen terhadap ulasan pengguna aplikasi **Shopee Indonesia** di Google Play Store. Tujuannya untuk mengelompokkan ribuan komentar pengguna menjadi tiga sentimen: **Positif**, **Negatif**, dan **Netral** menggunakan Machine Learning dan Deep Learning.

---

## Masalah & Tujuan

### Masalah
- Jumlah ulasan aplikasi di Play Store sangat banyak (puluhan ribu komentar), sehingga sulit direkap jika dibaca satu per satu secara manual.
- Rating bintang sering tidak sesuai dengan isi komentar. Misalnya pengguna memberi bintang 5 tapi isinya komplain pengiriman, atau sebaliknya.
- Komentar pengguna banyak memakai bahasa sehari-hari dan singkatan (*"gk"*, *"bgt"*, *"cod"*, *"lemot"*) sehingga perlu dirapikan dulu sebelum diolah komputer.

### Tujuan
- Mengambil data ulasan asli Shopee dari Google Play Store menggunakan `google-play-scraper`.
- Membersihkan teks ulasan dan mengubah kata-kata singkatan/gaul menjadi kata baku bahasa Indonesia.
- Melatih dan membandingkan 4 model (**LSTM**, **GRU**, **Logistic Regression**, dan **DNN**) untuk melihat model mana yang paling akurat dalam menebak sentimen.
- Menyimpan model yang sudah dilatih agar bisa langsung dipakai untuk menebak kalimat ulasan baru.

---

## Dataset

Data diambil dari ulasan aplikasi **Shopee (`com.shopee.id`)** di Google Play Store:
- **Jumlah awal hasil scraping:** 90.000 ulasan
- **Jumlah setelah dibersihkan (hapus data kosong & duplikat):** 58.300 ulasan

Pembagian label sentimen dihitung menggunakan kamus kata positif dan negatif bahasa Indonesia (*lexicon*):

| Sentimen | Jumlah Data | Persentase |
| :--- | :--- | :--- |
| Positif | 28.901 | 49,57% |
| Negatif | 25.730 | 44,13% |
| Netral | 3.669 | 6,29% |
| **Total** | **58.300** | **100%** |

---

## Tahapan Pengerjaan

1. **Scraping Data (`Scraping_Code.ipynb`)**
   Mengambil data ulasan dari Google Play Store dan menyimpannya ke file `ulasan_shoope.csv`.
2. **Pembersihan Teks (Preprocessing)**
   - Menghapus angka, tanda baca, link, mention, dan hashtag.
   - Mengubah semua huruf menjadi huruf kecil (*case folding*).
   - Mengubah kata singkatan/gaul menjadi kata baku.
   - Menghapus kata umum yang tidak berpengaruh ke sentimen (*stopwords* seperti *"yang"*, *"di"*, *"nya"*, *"sih"*).
3. **Pelabelan & Ekstraksi Fitur**
   Memberi label `positive`, `negative`, atau `neutral` pada teks, lalu mengubah teks menjadi bentuk angka menggunakan **Tokenizer (Word Embedding)** dan **TF-IDF**.
4. **Pelatihan Model & Evaluasi**
   Melatih 4 model dan membandingkan akurasinya pada data uji.

---

## Hasil Perbandingan Model

| Model | Fitur | Pembagian Data (Latih:Uji) | Akurasi Data Uji |
| :--- | :--- | :---: | :---: |
| **LSTM** | Word Embedding | 80 : 20 | **92,20%** |
| **Bidirectional GRU** | Word Embedding | 80 : 20 | **92,11%** |
| **Logistic Regression** | TF-IDF | 70 : 30 | **91,37%** |
| **DNN** | TF-IDF | 80 : 20 | **90,42%** |

Contoh hasil prediksi pada kalimat baru (`inference_Sentiment_Analysis.ipynb`):
- *"Saya sangat senang dengan produk ini!"* -> **POSITIF**
- *"Saya kecewa sama barangnya"* -> **NEGATIF**
- *"barang jelek gk sesuai"* -> **NEGATIF**
- *"banyak barangnya"* -> **NETRAL**

---

## Struktur File

```text
Proyek Sentiment Analisis/
├── data/
│   └── ulasan_shoope.csv                                  # Data hasil scraping ulasan Shopee
├── model/
│   ├── Model_best_logistic.pkl                            # File model Logistic Regression yang sudah dilatih
│   └── tfidf.pkl                                          # File TF-IDF vectorizer
├── Azhar_Sentiment_Analysis_using_Machine_Learning.ipynb  # Notebook utama (pembersihan data & pelatihan 4 model)
├── Scraping_Code.ipynb                                    # Notebook untuk scraping ulasan Play Store
├── inference_Sentiment_Analysis.ipynb                     # Notebook untuk tes prediksi kalimat baru
└── requirements.txt                                       # Daftar library Python
```

---

## Cara Menjalankan

1. **Install library yang dibutuhkan**
   ```bash
   pip install -r requirements.txt
   pip install google-play-scraper sastrawi nltk scikit-learn tensorflow wordcloud joblib
   ```
2. **Menjalankan pelatihan / eksplorasi data**
   Buka dan jalankan `Azhar_Sentiment_Analysis_using_Machine_Learning.ipynb` di Jupyter Notebook atau Google Colab.
3. **Mencoba prediksi kalimat baru**
   Buka `inference_Sentiment_Analysis.ipynb` untuk mengetes model yang sudah tersimpan di folder `model/`.
