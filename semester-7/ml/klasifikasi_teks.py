# ==============================================================================
# PROGRAM KLASIFIKASI TEKS BAHASA INDONESIA (VERSI REVISI KOMPREHENSIF)
# Penulis: Tim Riset NLP
# Target: Naskah Publikasi Jurnal Sinta 3 / Sinta 4
# Algoritma: Support Vector Machine (SVM), Random Forest (RF), Naive Bayes (MNB)
# Ekstraksi Fitur: TF-IDF (Unigram + Bigram)
# ==============================================================================

import os
import re
import string
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Library NLP Bahasa Indonesia
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

warnings.filterwarnings('ignore')

# Unduh resource NLTK jika belum tersedia
nltk.download('stopwords', quiet=True)

# ==============================================================================
# KONFIGURASI PARAMETER & DIREKTORI
# ==============================================================================
# Jika memiliki file dataset CSV sendiri, masukkan path-nya di sini:
# Contoh: DATASET_PATH = 'dataset_ulasan.csv'
DATASET_PATH = None

# Nama kolom pada dataset CSV
COL_TEXT = 'teks'
COL_LABEL = 'label'

# Rasio pembagian data (80% Train, 20% Test)
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Pengaturan Preprocessing
USE_STEMMING = False  # Set True jika ingin mengaktifkan stemming Sastrawi

# Direktori penyimpanan output
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. KAMUS NORMALISASI SLANG & SINGKATAN BAHASA INDONESIA
# ==============================================================================
SLANG_DICT = {
    'yg': 'yang', 'dgn': 'dengan', 'sdh': 'sudah', 'blm': 'belum',
    'bgt': 'banget', 'bgus': 'bagus', 'bgs': 'bagus', 'gak': 'tidak',
    'ga': 'tidak', 'gk': 'tidak', 'nggak': 'tidak', 'tdk': 'tidak',
    'tp': 'tapi', 'tpi': 'tapi', 'klo': 'kalau', 'kl': 'kalau',
    'sy': 'saya', 'aku': 'saya', 'gw': 'saya', 'gue': 'saya',
    'lu': 'kamu', 'lo': 'kamu', 'dr': 'dari', 'utk': 'untuk',
    'bkn': 'bukan', 'bgtu': 'begitu', 'apk': 'aplikasi', 'app': 'aplikasi',
    'lemot': 'lambat', 'lola': 'lambat', 'ancur': 'hancur', 'rusak': 'hancur',
    'parah': 'buruk', 'jelek': 'buruk', 'bintang': 'rating', 'mantap': 'bagus',
    'top': 'bagus', 'jos': 'bagus', 'puas': 'puas', 'kecewa': 'kecewa',
    'respon': 'tanggapan', 'slow': 'lambat', 'fast': 'cepat', 'error': 'rusak',
    'down': 'gangguan', 'tf': 'transfer', 'rek': 'rekening', 'admin': 'pengelola'
}

# ==============================================================================
# 2. PERSIAPAN DATASET
# ==============================================================================
def load_dataset():
    """Memuat dataset dari CSV atau menggunakan data sintetis komprehensif."""
    if DATASET_PATH and os.path.exists(DATASET_PATH):
        print(f"[INFO] Memuat dataset kustom dari: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        df = df[[COL_TEXT, COL_LABEL]].dropna()
    else:
        print("[INFO] Menggunakan dataset sampel representatif (60 ulasan simulasi)...")
        # Kosakata dengan sinyal sentimen kuat yang realistis
        positif_texts = [
            'Aplikasi ini sangat bagus, transaksi cepat dan antarmuka sangat ramah pengguna!',
            'Pelayanan sangat memuaskan, respon admin cepat dan sangat membantu.',
            'Suka sekali dengan promo dan diskon, aplikasi ini sangat bagus dan bermanfaat.',
            'Bagus sekali, sistem keamanan terjamin dan proses verifikasi sangat mudah dan cepat.',
            'Fitur transfer antar bank sangat lancar, cepat, dan mudah digunakan.',
            'Pengalaman pengguna sangat memuaskan, tampilan bagus dan transaksi selalu lancar.',
            'Fitur top up saldo bekerja sangat cepat, mudah, dan sangat memuaskan.',
            'Sangat merekomendasikan aplikasi ini, proses praktis, cepat, dan aman.',
            'Proses pendaftaran akun sangat mudah, cepat, dan tidak berbelit-belit.',
            'Tampilan modern, navigasi mudah, dan performa aplikasi sangat bagus dan stabil.',
            'Luar biasa cepat proses pencairan dana, layanan sangat memuaskan dan terpercaya.',
            'Customer service ramah, respon sangat cepat dan pelayanan sangat baik.',
            'Sangat membantu transaksi harian, aplikasi ringan, cepat, dan sangat bagus.',
            'Mudah digunakan, menu tertata rapi, dan sistem sangat memuaskan.',
            'Fitur keamanan sangat bagus, verifikasi sidik jari lancar dan aman.',
            'Aplikasi terbaik untuk pembayaran, selalu cepat dan tidak pernah ada kendala.',
            'Sangat puas dengan update baru, makin cepat, responsif, dan mudah.',
            'Bagus banget pelayanannya, keluhan langsung diselesaikan dengan cepat dan ramah.',
            'Aplikasi transaksi paling praktis, aman, dan sangat memuaskan penggunanya.',
            'Senang sekali menggunakan aplikasi ini, hemat waktu dan fiturnya sangat bagus.',
            'Top banget, transfer uang antar rekening selalu lancar dan sangat cepat.',
            'Keren sekali sistemnya, tampilan rapi, mudah dipelajari, dan performa bagus.',
            'Sangat bermanfaat untuk bisnis, transaksi aman, cepat, dan memuaskan.',
            'Pelayanan ramah, aplikasi ringan dan sangat bagus untuk kebutuhan harian.',
            'Respon customer service jempolan, sangat cepat, solutif, dan ramah.',
            'Aplikasi mantap, loading cepat, navigasi mudah, dan transaksi selalu berhasil.',
            'Proses mudah dan aman, sangat puas menggunakan layanan perbankan digital ini.',
            'Bagus dan terpercaya, setiap transaksi konfirmasinya sangat cepat.',
            'Aplikasi ini tidak mengecewakan, transfer saldo cepat dan pelayanan ramah.',
            'Pelayanan tidak buruk sama sekali, proses verifikasi akun cukup cepat dan aman.'
        ]
        negatif_texts = [
            'Sangat kecewa, aplikasi sering error saat login dan sistem lambat.',
            'Aplikasi lambat sekali dan fiturnya membingungkan, sangat mengecewakan.',
            'Aplikasi penipu, uang hilang dan tidak ada respon sama sekali, jelek.',
            'Jelek banget, selalu mental dan error saat buka menu pembayaran.',
            'Update terbaru justru bikin aplikasi makin lemot, berat, dan sering error.',
            'Pelayanan sangat mengecewakan, komplain berhari-hari tidak pernah direspon, buruk.',
            'Sistem pembayaran gagal terus dan transaksi error padahal koneksi lancar.',
            'Banyak bug pada versi ini, aplikasi sering keluar sendiri dan lambat.',
            'Kecewa berat, akun tiba-tiba terblokir dan layanan sangat buruk dan lambat.',
            'Aplikasi terburuk yang pernah saya pakai, sering gangguan dan sangat mengecewakan.',
            'Sering force close dan error saat transaksi, sistem sangat jelek.',
            'Biaya admin mahal dan aplikasi sering error, sangat mengecewakan pengguna.',
            'Server sering down dan lambat di jam sibuk, bikin frustrasi dan kecewa.',
            'Notifikasi transaksi sering lambat dan error, sangat buruk keamanannya.',
            'Fitur baru tidak ada gunanya, bikin aplikasi makin berat, lemot, dan rusak.',
            'Aplikasi tidak berguna, sering gagal transaksi dan admin tidak ada respon.',
            'Sangat kecewa dengan respon lambat customer service saat ada masalah saldo.',
            'Tampilan jelek dan navigasi membingungkan, sering macet dan error.',
            'Aplikasi parah, selalu gagal saat verifikasi data dan lambat sekali.',
            'Pelayanan sangat jelek, sistem sering rusak dan saldo terpotong tapi gagal.',
            'Aplikasi lemot dan sering crash, perbaikan sistem tidak ada hasilnya, buruk.',
            'Kecewa sekali, transaksi tertunda berjam-jam tanpa kejelasan dari sistem.',
            'Jangan pakai aplikasi ini, banyak bug, lambat, dan sangat merugikan.',
            'Sistem sangat buruk, loading lama sekali dan sering muncul pesan error.',
            'Kacau sekali update kali ini, malah sering mental dan koneksi selalu gagal.',
            'Customer care lambat sekali merespon, pelayanan paling buruk dan mengecewakan.',
            'Aplikasi jelek dan lambat, fitur transfer sering gangguan dan gagal terus.',
            'Luar biasa sekali aplikasinya, baru buka langsung menutup sendiri, sangat parah.',
            'Keren banget nih sistemnya, saldo kepotong tapi transaksi dibilang gagal.',
            'Aplikasi payah, bikin emosi karena selalu lemot dan gagal login.'
        ]
        sample_data = {
            'teks': positif_texts + negatif_texts,
            'label': ['positif'] * len(positif_texts) + ['negatif'] * len(negatif_texts)
        }
        df = pd.DataFrame(sample_data)

    print(f"[INFO] Total data: {len(df)} baris.")
    print(f"[INFO] Distribusi label:\n{df[COL_LABEL].value_counts()}")
    return df

# ==============================================================================
# 3. FUNGSI PRA-PEMROSESAN TEKS (TEXT PREPROCESSING)
# ==============================================================================
stop_words_id = set(stopwords.words('indonesian'))
if USE_STEMMING:
    stemmer = StemmerFactory().create_stemmer()

def preprocess_text(text):
    """
    Tahapan:
    1. Case Folding
    2. Pembersihan Regex (URL, mention, hashtag, angka, tanda baca)
    3. Normalisasi Kata Slang & Singkatan
    4. Stopword Removal
    5. Stemming (Opsional)
    """
    # 1. Case Folding
    text = str(text).lower()

    # 2. Pembersihan Regex
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)       # Hapus URL
    text = re.sub(r'[-+]?[0-9]+', ' ', text)                 # Hapus Angka
    text = re.sub(r'[@#]\S+', ' ', text)                    # Hapus Mention & Hashtag
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus Tanda Baca
    text = re.sub(r'\s+', ' ', text).strip()                # Normalisasi Spasi

    # 3. Normalisasi Slang & Stopword Removal
    words = text.split()
    normalized_words = [SLANG_DICT.get(w, w) for w in words]
    filtered_words = [w for w in normalized_words if w not in stop_words_id and len(w) > 2]

    # 4. Stemming (jika diaktifkan)
    if USE_STEMMING:
        cleaned_text = stemmer.stem(' '.join(filtered_words))
    else:
        cleaned_text = ' '.join(filtered_words)

    return cleaned_text

# ==============================================================================
# 4. EKSEKUSI UTAMA (MAIN PIPELINE)
# ==============================================================================
def main():
    print("\n" + "="*70)
    print("PIPELINE KLASIFIKASI TEKS BAHASA INDONESIA: SVM vs RF vs NAIVE BAYES")
    print("="*70)

    # 1. Load Data
    df = load_dataset()

    # 2. Preprocessing
    print("\n[1/5] Melakukan pra-pemrosesan teks (cleansing, slang, stopword)...")
    df['clean_text'] = df[COL_TEXT].apply(preprocess_text)
    
    # Filter kalimat kosong jika ada
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)

    # 3. Train-Test Split (Stratified 80:20)
    print(f"\n[2/5] Membagi dataset ({int((1-TEST_SIZE)*100)}% Train : {int(TEST_SIZE*100)}% Test, Stratified)...")
    X = df['clean_text']
    y = df[COL_LABEL]
    X_original = df[COL_TEXT]  # Disimpan untuk analisis kesalahan

    X_train, X_test, y_train, y_test, orig_train, orig_test = train_test_split(
        X, y, X_original, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      - Jumlah data latih : {len(X_train)}")
    print(f"      - Jumlah data uji   : {len(X_test)}")

    # 4. Ekstraksi Fitur TF-IDF (Unigram + Bigram)
    print("\n[3/5] Mengekstraksi fitur dengan TF-IDF (Unigram & Bigram)...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)  # Transform saja untuk cegah data leakage
    print(f"      - Dimensi matriks TF-IDF: {X_train_tfidf.shape[1]} fitur kosakata.")

    # 5. Definisi dan Pelatihan Model
    print("\n[4/5] Melatih dan mengevaluasi model klasifikasi...")
    models = {
        "Support Vector Machine": SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0)
    }

    metrics_summary = []
    predictions = {}
    class_labels = sorted(list(y.unique()))

    # Setup figure untuk visualisasi Confusion Matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (name, model) in enumerate(models.items()):
        # Pelatihan
        model.fit(X_train_tfidf, y_train)
        
        # Pengujian
        y_pred = model.predict(X_test_tfidf)
        predictions[name] = y_pred

        # Hitung metrik evaluasi
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        metrics_summary.append({
            'Algoritma': name,
            'Akurasi (%)': round(acc * 100, 2),
            'Presisi': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4)
        })

        # Cetak Classification Report
        print(f"\n--- [Classification Report: {name}] ---")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
            xticklabels=class_labels, yticklabels=class_labels, cbar=False
        )
        axes[idx].set_title(f"CM: {name}\nAkurasi: {acc*100:.1f}%", fontsize=11, fontweight='bold')
        axes[idx].set_xlabel("Prediksi Model")
        axes[idx].set_ylabel("Label Aktual")

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_komparasi.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Gambar Confusion Matrix disimpan di: {cm_path}")

    # ==============================================================================
    # 5. TABEL RINGKASAN HASIL & EKSPOR
    # ==============================================================================
    df_results = pd.DataFrame(metrics_summary).sort_values(by='Akurasi (%)', ascending=False)
    print("\n" + "="*70)
    print("TABEL PERBANDINGAN PERFORMA ALGORITMA (UNTUK BAB 3 PAPER)")
    print("="*70)
    print(df_results.to_string(index=False))

    csv_results_path = os.path.join(OUTPUT_DIR, "hasil_evaluasi_model.csv")
    df_results.to_csv(csv_results_path, index=False)
    print(f"\n[INFO] Tabel metrik disimpan di: {csv_results_path}")

    # Visualisasi Bar Chart Perbandingan Metrik
    plt.figure(figsize=(9, 5))
    bar_data = pd.melt(df_results, id_vars=['Algoritma'], value_vars=['Presisi', 'Recall', 'F1-Score'],
                       var_name='Metrik', value_name='Skor')
    sns.barplot(x='Algoritma', y='Skor', hue='Metrik', data=bar_data, palette='Set2')
    plt.title("Perbandingan Metrik Evaluasi Model Klasifikasi", fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.ylabel("Skor (Weighted Average)")
    plt.legend(loc='lower right')
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "grafik_komparasi_model.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[INFO] Grafik komparasi metrik disimpan di: {chart_path}")

    # ==============================================================================
    # 6. ANALISIS KESALAHAN OTOMATIS (ERROR ANALYSIS)
    # ==============================================================================
    print("\n" + "="*70)
    print("ANALISIS KESALAHAN (ERROR ANALYSIS) PADA MODEL TERBAIK (SVM)")
    print("="*70)

    best_model_name = df_results.iloc[0]['Algoritma']
    best_preds = predictions[best_model_name]

    df_errors = pd.DataFrame({
        'Teks_Asli': orig_test.values,
        'Teks_Clean': X_test.values,
        'Label_Aktual': y_test.values,
        'Prediksi_Model': best_preds
    })

    misclassified = df_errors[df_errors['Label_Aktual'] != df_errors['Prediksi_Model']]

    if len(misclassified) > 0:
        print(f"[INFO] Ditemukan {len(misclassified)} teks yang salah diprediksi oleh {best_model_name}:")
        for i, row in misclassified.iterrows():
            print(f"- Teks Asli     : \"{row['Teks_Asli']}\"")
            print(f"  Label Aktual  : {row['Label_Aktual']} --> Prediksi: {row['Prediksi_Model']}")
            print(f"  Teks Bersih   : \"{row['Teks_Clean']}\"\n")
    else:
        print(f"[INFO] Seluruh data uji berhasil diklasifikasikan dengan sempurna oleh {best_model_name}!")

    print("="*70)
    print("PROGRAM SELESAI DIJALANKAN DENGAN SUKSES!")
    print("="*70)

if __name__ == '__main__':
    main()
