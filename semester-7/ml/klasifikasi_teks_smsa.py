 # ==============================================================================
# 1. IMPORT LIBRARY
# ==============================================================================
import re
import string
import pandas as pd
import numpy as np

# Library untuk pemrosesan teks bahasa Indonesia
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Library Scikit-Learn untuk ML dan evaluasi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Download stopwords dari NLTK (jika belum pernah diunduh)
nltk.download('stopwords', quiet=True)


# ==============================================================================
# 2. MEMUAT DATASET (LOAD DATA)
# ==============================================================================
# CATATAN: Jika menggunakan file CSV sendiri, hilangkan komentar baris berikut:
# df = pd.read_csv('nama_file_dataset.csv')
# Pastikan nama kolom teks dan label disesuaikan.

# Di bawah ini adalah contoh data dummy ulasan bahasa Indonesia untuk demonstrasi
data_dummy = {
    'teks': [
        'Aplikasi ini sangat bagus, transaksinya cepat dan antarmukanya ramah pengguna!',
        'Sangat kecewa, aplikasi sering error saat login dan saldo tidak terupdate.',
        'Pelayanan customer service cukup baik, respon cepat dan sangat membantu.',
        'Aplikasi lambat sekali dan fiturnya membingungkan, tolong diperbaiki.',
        'Suka sekali dengan promo dan diskon yang diberikan, sangat bermanfaat!',
        'Aplikasi penipu, uang saya hilang dan tidak ada respon dari admin.',
        'Bagus sekali, sistem keamanan terjamin dan proses verifikasi mudah.',
        'Jelek banget, selalu mental saat buka menu pembayaran.',
        'Fitur transfer antar bank sangat lancar tanpa kendala sama sekali.',
        'Update terbaru justru bikin aplikasi makin lemot dan berat.'
    ],
    'label': [
        'positif', 'negatif', 'positif', 'negatif', 'positif',
        'negatif', 'positif', 'negatif', 'positif', 'negatif'
    ]
}

df = pd.DataFrame(data_dummy)
print(f"Total data: {len(df)} baris")
print(df.head())


# ==============================================================================
# 3. TAHAP PREPROCESSING TEKS BAHASA INDONESIA
# ==============================================================================
# Menyiapkan daftar stopword Bahasa Indonesia dari NLTK
list_stopwords = set(stopwords.words('indonesian'))

# Menyiapkan Stemmer Sastrawi
# CATATAN: Stemming memakan waktu cukup lama jika dataset berukuran > 5.000 data.
# Jika data sangat besar, proses stemming bisa dilewati/dinonaktifkan jika perlu.
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_indonesia(text, use_stemmer=False):
    """
    Fungsi untuk membersihkan dan menormalisasi teks bahasa Indonesia:
    1. Case folding (mengubah ke huruf kecil)
    2. Menghapus URL, username/mention, hashtag, angka, dan tanda baca
    3. Menghapus spasi berlebih
    4. Stopword removal (menghapus kata yang kurang bermakna)
    5. Stemming (opsional)
    """
    # 1. Case folding
    text = text.lower()

    # 2. Pembersihan teks dengan Regular Expression (Regex)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)     # Hapus URL/Link
    text = re.sub(r'[-+]?[0-9]+', '', text)               # Hapus angka
    text = re.sub(r'[@#]\S+', '', text)                  # Hapus mention (@) dan hashtag (#)
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    text = re.sub(r'\s+', ' ', text).strip()              # Hapus spasi berlebih

    # 3. Tokenisasi sederhana & Stopword Removal
    tokens = text.split()
    tokens_filtered = [word for word in tokens if word not in list_stopwords]

    # 4. Stemming (opsional)
    if use_stemmer:
        cleaned_text = stemmer.stem(' '.join(tokens_filtered))
    else:
        cleaned_text = ' '.join(tokens_filtered)

    return cleaned_text

# Terapkan preprocessing ke seluruh kolom teks
print("\n[INFO] Sedang melakukan preprocessing teks...")
df['teks_bersih'] = df['teks'].apply(lambda x: preprocess_indonesia(x, use_stemmer=False))
print("[INFO] Preprocessing selesai!")


# ==============================================================================
# 4. PEMBAGIAN DATASET (TRAIN-TEST SPLIT)
# ==============================================================================
X = df['teks_bersih']
y = df['label']

# Pembagian data: 80% Training dan 20% Testing
# stratify=y memastikan rasio kelas seimbang pada data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nJumlah Data Latih (Train): {len(X_train)}")
print(f"Jumlah Data Uji (Test)   : {len(X_test)}")


# ==============================================================================
# 5. EKSTRAKSI FITUR TF-IDF
# ==============================================================================
# max_features membatasi jumlah fitur teratas (misal 5000 kata penting)
# ngram_range=(1, 2) mengambil unigram (1 kata) dan bigram (pasangan 2 kata)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit dan transform pada data train, HANYA transform pada data test (mencegah data leakage)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"\nDimensi matriks TF-IDF Data Train: {X_train_tfidf.shape}")


# ==============================================================================
# 6. INISIALISASI DAN PELATIHAN 3 MODEL KLASIFIKASI
# ==============================================================================
models = {
    "Naive Bayes (Bayesian)": MultinomialNB(alpha=1.0),
    "SVM (Support Vector Machine)": SVC(kernel='linear', C=1.0, random_state=42),
    "Random Forest (Ensemble Tree)": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Dictionary untuk menyimpan rekap metrik hasil evaluasi
summary_results = []

print("\n" + "="*70)
print("HASIL EVALUASI MODEL KLASIFIKASI (CLASSIFICATION REPORT)")
print("="*70)

for model_name, model in models.items():
    # A. Pelatihan model
    model.fit(X_train_tfidf, y_train)

    # B. Prediksi pada data uji
    y_pred = model.predict(X_test_tfidf)

    # C. Cetak Classification Report detail per model
    print(f"\n>>> Model: {model_name}")
    print("-" * 55)
    print(classification_report(y_test, y_pred, zero_division=0))

    # D. Hitung metrik keseluruhan untuk tabel ringkasan (Weighted Average)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

    summary_results.append({
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4)
    })


# ==============================================================================
# 7. TABEL KOMPARASI PERFORMA MODEL (UNTUK LAPORAN/PAPER)
# ==============================================================================
df_comparison = pd.DataFrame(summary_results)

print("\n" + "="*70)
print("TABEL RINGKASAN PERBANDINGAN PERFORMA ALGORITMA")
print("="*70)
print(df_comparison.to_string(index=False))