import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Library NLP & ML
from google_play_scraper import reviews, Sort
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from wordcloud import WordCloud

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="Analisis Sentimen Google Play Store",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Analisis Sentimen Ulasan Aplikasi Play Store")
st.markdown("Aplikasi interaktif untuk Scraping, Pre-processing, Labelling, Ekstraksi Fitur, dan Klasifikasi Machine Learning.")

# --- SIDEBAR: NAVIGASI ---
sidebar_menu = st.sidebar.radio(
    "Pilih Menu:",
    [
        "1. Scraping Data", 
        "2. Pre-Processing", 
        "3. Labelling Sentimen", 
        "4. Ekstraksi Fitur & Visualisasi",
        "5. Klasifikasi Random Forest"
    ]
)

# --- FUNGSI PREPROCESSING TEKS ---
@st.cache_data
def preprocessing_text(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|\#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    kamus_normalisasi = {
        'gw': 'saya', 'gue': 'saya', 'lg': 'lagi', 'bgt': 'banget',
        'gk': 'tidak', 'ngggak': 'tidak', 'tdk': 'tidak', 'jg': 'juga',
        'dgn': 'dengan', 'utk': 'untuk', 'ml': 'mobile legends',
        'mlbb': 'mobile legends', 'bocil': 'anak kecil', 'nyel': 'menyesal',
        'karna': 'karena', 'tp': 'tapi', 'sm': 'sama'
    }
    words = text.split()
    words = [kamus_normalisasi.get(word, word) for word in words]
    return ' '.join(words)

# Lexicon untuk Sentimen
positive_lexicon = {
    'bagus', 'suka', 'senang', 'mantap', 'hebat', 'keren', 'puas', 'terbaik',
    'rekomendasi', 'ramah', 'cepat', 'mudah', 'seru', 'asik', 'lancar', 'juara', 'andal'
}

negative_lexicon = {
    'jelek', 'buruk', 'kecewa', 'lambat', 'kesal', 'rusak', 'mahal', 'sulit',
    'rugi', 'parah', 'benci', 'kapok', 'lag', 'bug', 'curang', 'toksik',
    'sampah', 'kalah', 'dark', 'sistem', 'tai', 'kontol', 'memek', 'goblok', 'anjing', 'tolol'
}

def classify_sentiment(text):
    words = str(text).lower().split()
    pos_score = sum(1 for word in words if word in positive_lexicon)
    neg_score = sum(1 for word in words if word in negative_lexicon)
    score = pos_score - neg_score
    return 'Positif' if score >= 0 else 'Negatif'


# MENU 1: SCRAPING DATA
if sidebar_menu == "1. Scraping Data":
    st.header("1. Scraping Ulasan Aplikasi")
    st.write("Ambil ulasan secara real-time dari Google Play Store.")

    col1, col2 = st.columns(2)
    with col1:
        app_package = st.text_input("App Package Name:", value="com.mobile.legends")
        target_count = st.number_input("Target Jumlah Data:", min_value=10, max_value=10000, value=200, step=100)
    with col2:
        start_year = st.number_input("Tahun Mulai:", min_value=2015, max_value=2026, value=2024)
        end_year = st.number_input("Tahun Selesai:", min_value=2015, max_value=2026, value=2026)

    if st.button("Mulai Scraping"):
        with st.spinner("Sedang mengambil data ulasan..."):
            all_reviews = []
            continuation_token = None
            fetched_count = 0
            batch_size = 200
            start_time = time.time()
            progress_bar = st.progress(0)

            while fetched_count < target_count:
                try:
                    result, continuation_token = reviews(
                        app_package,
                        lang='id',
                        country='id',
                        sort=Sort.NEWEST,
                        count=batch_size,
                        continuation_token=continuation_token
                    )

                    if not result:
                        break

                    filtered_batch = []
                    stop_scraping = False

                    for r in result:
                        review_date = r.get('at')
                        if review_date:
                            review_year = review_date.year
                            if start_year <= review_year <= end_year:
                                filtered_batch.append({
                                    'reviewId': r.get('reviewId'),
                                    'userName': r.get('userName'),
                                    'score': r.get('score'),
                                    'thumbsUp': r.get('thumbsUpCount'),
                                    'review': r.get('content'),
                                    'at': review_date,
                                    'appVersion': r.get('reviewCreatedVersion')
                                })
                            elif review_year < start_year:
                                stop_scraping = True
                                break

                    all_reviews.extend(filtered_batch)
                    fetched_count = len(all_reviews)
                    progress = min(fetched_count / target_count, 1.0)
                    progress_bar.progress(progress)

                    if stop_scraping or not continuation_token:
                        break
                    time.sleep(1)

                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
                    break

            if all_reviews:
                df_scraped = pd.DataFrame(all_reviews[:target_count])
                st.session_state['df_raw'] = df_scraped
                elapsed_time = round(time.time() - start_time, 2)
                st.success(f"Scraping selesai! Total ulasan: {len(df_scraped)} (Waktu: {elapsed_time} detik)")
                st.dataframe(df_scraped.head(10))

                csv = df_scraped.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("Download Data Hasil Scraping (CSV)", csv, f"playstore_reviews_{app_package}.csv", "text/csv")
            else:
                st.warning("Tidak ada data yang berhasil dikumpulkan.")


# MENU 2: PRE-PROCESSING
elif sidebar_menu == "2. Pre-Processing":
    st.header("2. Pre-Processing Teks")
    
    uploaded_file = st.file_uploader("Unggah file CSV hasil scraping (opsional jika sudah scraping di Menu 1)", type=["csv"])
    if uploaded_file is not None:
        st.session_state['df_raw'] = pd.read_csv(uploaded_file)

    if 'df_raw' in st.session_state:
        df = st.session_state['df_raw'].copy()
        st.write("### Data Awal:")
        st.dataframe(df.head())

        if st.button("Jalankan Pre-Processing"):
            with st.spinner("Memproses teks (Cleaning, Stopwords, & Stemming Sastrawi)... Ini mungkin memakan waktu beberapa saat."):
                df['clean_review'] = df['review'].apply(preprocessing_text)
                
                stop_factory = StopWordRemoverFactory()
                stopword = stop_factory.create_stop_word_remover()
                df['clean_review'] = df['clean_review'].apply(lambda x: stopword.remove(x))

                stem_factory = StemmerFactory()
                stemmer = stem_factory.create_stemmer()
                df['clean_review'] = df['clean_review'].apply(lambda x: stemmer.stem(x))

                st.session_state['df_processed'] = df
                st.success("Pre-processing selesai!")
                
                st.write("### Contoh Hasil Perbandingan Teks:")
                for i in range(min(3, len(df))):
                    st.text(f"[{i+1}] Asli  : {df['review'].iloc[i]}")
                    st.text(f"    Clean : {df['clean_review'].iloc[i]}\n")
                
                csv_proc = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button("Download Data Preprocessed (CSV)", csv_proc, "processed_data.csv", "text/csv")
    else:
        st.info("Silakan lakukan scraping di Menu 1 terlebih dahulu atau unggah file CSV hasil scraping.")


# MENU 3: LABELLING SENTIMEN
elif sidebar_menu == "3. Labelling Sentimen":
    st.header("3. Labelling Sentimen Berdasarkan Lexicon")
    
    uploaded_file = st.file_uploader("Unggah file CSV processed data (opsional)", type=["csv"])
    if uploaded_file is not None:
        st.session_state['df_processed'] = pd.read_csv(uploaded_file)

    if 'df_processed' in st.session_state:
        df = st.session_state['df_processed'].copy()
        df['clean_review'] = df['clean_review'].fillna(df['review']).fillna('')

        if st.button("Proses Labelling"):
            df['Sentimen'] = df['clean_review'].apply(classify_sentiment)
            st.session_state['df_labeled'] = df
            st.success("Labelling selesai!")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("### Hasil Labelling Data:")
                st.dataframe(df[['clean_review', 'Sentimen']].head(10))
            
            with col2:
                sentiment_counts = df['Sentimen'].value_counts()
                st.write("### Jumlah Distribusi:")
                st.write(sentiment_counts)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.set_theme(style='whitegrid')
            colors = ['#2ecc71' if label == 'Positif' else '#e74c3c' for label in sentiment_counts.index]
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors, ax=ax)
            ax.set_title('Distribusi Analisis Sentimen', fontsize=12, fontweight='bold')
            ax.set_xlabel('Kategori Sentimen')
            ax.set_ylabel('Jumlah Ulasan')
            
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height):,}', (p.get_x() + p.get_width() / 2.0, height),
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            st.pyplot(fig)

            csv_labeled = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("Download Data Hasil Labelling (CSV)", csv_labeled, "hasil_labelling.csv", "text/csv")
    else:
        st.info("Silakan lakukan pre-processing di Menu 2 terlebih dahulu atau unggah file CSV yang telah diproses.")


# MENU 4: EKSTRAKSI FITUR & VISUALISASI
elif sidebar_menu == "4. Ekstraksi Fitur & Visualisasi":
    st.header("4. Ekstraksi Fitur (TF-IDF) & Visualisasi Kata")

    uploaded_file = st.file_uploader("Unggah file CSV hasil labelling (opsional)", type=["csv"])
    if uploaded_file is not None:
        st.session_state['df_labeled'] = pd.read_csv(uploaded_file)

    if 'df_labeled' in st.session_state:
        df = st.session_state['df_labeled'].copy()
        df['clean_review'] = df['clean_review'].fillna('').astype(str)

        # 1. TF-IDF Extraction
        st.subheader("A. Ekstraksi Fitur TF-IDF")
        vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.85)
        X_tfidf = vectorizer.fit_transform(df['clean_review'])
        feature_names = vectorizer.get_feature_names_out()

        df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=feature_names)
        if 'Sentimen' in df.columns:
            df_tfidf['Sentimen_Target'] = df['Sentimen'].values

        st.write("Preview Matriks TF-IDF:")
        st.dataframe(df_tfidf.head())

        csv_tfidf = df_tfidf.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("Download Matriks TF-IDF (CSV)", csv_tfidf, "fitur_tfidf.csv", "text/csv")

        st.markdown("---")

        # 2. Visualisasi Kata Terbanyak
        st.subheader("B. 10 Kata Paling Sering Muncul")
        all_words = ' '.join(df['clean_review']).split()
        top_words = Counter(all_words).most_common(10)
        df_top_words = pd.DataFrame(top_words, columns=['Kata', 'Frekuensi'])

        fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
        sns.barplot(x='Frekuensi', y='Kata', data=df_top_words, palette='viridis', ax=ax_bar)
        ax_bar.set_title('10 Kata Paling Sering Muncul', fontsize=12, fontweight='bold')
        
        for p in ax_bar.patches:
            width = p.get_width()
            ax_bar.annotate(f'{int(width):,}', (width + 5, p.get_y() + p.get_height() / 2.0),
                            ha='left', va='center', fontsize=10, fontweight='bold')
        
        st.pyplot(fig_bar)

        st.markdown("---")

        # 3. Visualisasi WordCloud
        st.subheader("C. Visualisasi Word Cloud")
        text_corpus = ' '.join(df['clean_review'])
        if text_corpus.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=150).generate(text_corpus)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis('off')
            ax_wc.set_title('Word Cloud Ulasan', fontsize=14, fontweight='bold')
            st.pyplot(fig_wc)
        else:
            st.warning("Teks kosong, tidak dapat membuat WordCloud.")
    else:
        st.info("Silakan lakukan labelling di Menu 3 terlebih dahulu atau unggah file CSV yang sudah di-label.")


# MENU 5: KLASIFIKASI RANDOM FOREST
elif sidebar_menu == "5. Klasifikasi Random Forest":
    st.header("5. Pemodelan & Klasifikasi Sentimen (Random Forest)")

    uploaded_file = st.file_uploader("Unggah file CSV hasil labelling (opsional)", type=["csv"])
    if uploaded_file is not None:
        st.session_state['df_labeled'] = pd.read_csv(uploaded_file)

    if 'df_labeled' in st.session_state:
        df = st.session_state['df_labeled'].copy()
        
        if 'Sentimen' not in df.columns:
            st.error("Kolom 'Sentimen' tidak ditemukan pada dataset. Lakukan labelling terlebih dahulu di Menu 3.")
        else:
            df['clean_review'] = df['clean_review'].fillna('').astype(str)
            
            # Pengaturan parameter pelatihan
            col1, col2 = st.columns(2)
            with col1:
                test_size_ratio = st.slider("Rasio Data Uji (Test Size):", 0.1, 0.5, 0.2, 0.05)
            with col2:
                n_estimators = st.number_input("Jumlah Pohon (n_estimators):", min_value=10, max_value=5000, value=100, step=10)

            if st.button("Latih Model Random Forest"):
                with st.spinner("Melatih model Random Forest..."):
                    # 1. Ekstraksi Fitur TF-IDF
                    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.85)
                    X = vectorizer.fit_transform(df['clean_review'])
                    y = df['Sentimen']

                    # 2. Split Data Training & Testing
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)

                    # 3. Training Model Random Forest
                    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    rf_model.fit(X_train, y_train)

                    # 4. Prediksi & Evaluasi
                    y_pred = rf_model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)

                    # Simpan ke session_state untuk prediksi interaktif & persistensi hasil
                    st.session_state['rf_model'] = rf_model
                    st.session_state['tfidf_vectorizer'] = vectorizer
                    st.session_state['rf_acc'] = acc
                    st.session_state['rf_report'] = pd.DataFrame(report_dict).transpose()
                    st.session_state['rf_cm'] = cm
                    st.session_state['rf_classes'] = rf_model.classes_
                    st.session_state['rf_trained'] = True

                    st.success("Model Random Forest berhasil dilatih!")

            # --- Tampilkan Output Evaluasi & Klasifikasi ---
            if st.session_state.get('rf_trained', False):
                st.markdown("---")
                st.subheader("📊 Output Klasifikasi & Evaluasi Model")

                acc_percent = st.session_state['rf_acc'] * 100

                # Tampilkan Akurasi dalam Persentase
                col_acc1, col_acc2 = st.columns([1, 2])
                with col_acc1:
                    st.metric(label="Akurasi Model (Accuracy)", value=f"{acc_percent:.2f}%")
                with col_acc2:
                    st.info(f"Model Random Forest berhasil dilatih dengan tingkat akurasi sebesar **{acc_percent:.2f}%**.")

                # Classification Report & Confusion Matrix
                col_rep, col_cm = st.columns([1, 1])
                with col_rep:
                    st.subheader("Classification Report")
                    st.dataframe(st.session_state['rf_report'])
                with col_cm:
                    st.subheader("Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
                    sns.heatmap(
                        st.session_state['rf_cm'], 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues', 
                        xticklabels=st.session_state['rf_classes'], 
                        yticklabels=st.session_state['rf_classes'], 
                        ax=ax_cm
                    )
                    ax_cm.set_xlabel('Prediksi')
                    ax_cm.set_ylabel('Aktual')
                    ax_cm.set_title('Confusion Matrix Random Forest')
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

                # --- Visualisasi Word Cloud (Tumpukan Kata) ---
                st.markdown("---")
                st.subheader("☁️ Word Cloud (Tumpukan Kata)")
                
                tab_wc1, tab_wc2, tab_wc3 = st.tabs(["Semua Sentimen", "Sentimen Positif", "Sentimen Negatif"])

                with tab_wc1:
                    text_all = ' '.join(df['clean_review'].dropna().astype(str))
                    if text_all.strip():
                        wc_all = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=150).generate(text_all)
                        fig_wc_all, ax_wc_all = plt.subplots(figsize=(10, 5))
                        ax_wc_all.imshow(wc_all, interpolation='bilinear')
                        ax_wc_all.axis('off')
                        ax_wc_all.set_title('Word Cloud - Semua Ulasan', fontsize=14, fontweight='bold')
                        st.pyplot(fig_wc_all)
                        plt.close(fig_wc_all)
                    else:
                        st.warning("Tidak ada teks untuk membuat Word Cloud.")

                with tab_wc2:
                    df_pos = df[df['Sentimen'] == 'Positif']
                    text_pos = ' '.join(df_pos['clean_review'].dropna().astype(str)) if not df_pos.empty else ''
                    if text_pos.strip():
                        wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens', max_words=150).generate(text_pos)
                        fig_wc_pos, ax_wc_pos = plt.subplots(figsize=(10, 5))
                        ax_wc_pos.imshow(wc_pos, interpolation='bilinear')
                        ax_wc_pos.axis('off')
                        ax_wc_pos.set_title('Word Cloud - Sentimen Positif', fontsize=14, fontweight='bold')
                        st.pyplot(fig_wc_pos)
                        plt.close(fig_wc_pos)
                    else:
                        st.warning("Tidak ada data ulasan dengan sentimen Positif.")

                with tab_wc3:
                    df_neg = df[df['Sentimen'] == 'Negatif']
                    text_neg = ' '.join(df_neg['clean_review'].dropna().astype(str)) if not df_neg.empty else ''
                    if text_neg.strip():
                        wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds', max_words=150).generate(text_neg)
                        fig_wc_neg, ax_wc_neg = plt.subplots(figsize=(10, 5))
                        ax_wc_neg.imshow(wc_neg, interpolation='bilinear')
                        ax_wc_neg.axis('off')
                        ax_wc_neg.set_title('Word Cloud - Sentimen Negatif', fontsize=14, fontweight='bold')
                        st.pyplot(fig_wc_neg)
                        plt.close(fig_wc_neg)
                    else:
                        st.warning("Tidak ada data ulasan dengan sentimen Negatif.")

            # --- Uji Prediksi Teks Baru ---
            st.markdown("---")
            st.subheader("🔍 Uji Coba Prediksi Ulasan Baru")
            user_input = st.text_area("Masukkan teks ulasan aplikasi:")
            
            if st.button("Prediksi Sentimen"):
                if 'rf_model' in st.session_state and 'tfidf_vectorizer' in st.session_state:
                    if user_input.strip() == "":
                        st.warning("Masukkan teks terlebih dahulu.")
                    else:
                        # Preprocessing teks baru
                        clean_input = preprocessing_text(user_input)
                        stop_factory = StopWordRemoverFactory()
                        clean_input = stop_factory.create_stop_word_remover().remove(clean_input)
                        clean_input = StemmerFactory().create_stemmer().stem(clean_input)

                        # Vektorisasi & Prediksi
                        input_vector = st.session_state['tfidf_vectorizer'].transform([clean_input])
                        prediction = st.session_state['rf_model'].predict(input_vector)[0]
                        probability = st.session_state['rf_model'].predict_proba(input_vector).max()

                        if prediction == 'Positif':
                            st.success(f"Hasil Prediksi: **{prediction}** (Confidence: {probability * 100:.2f}%)")
                        else:
                            st.error(f"Hasil Prediksi: **{prediction}** (Confidence: {probability * 100:.2f}%)")
                else:
                    st.warning("Silakan latih model Random Forest terlebih dahulu di atas sebelum melakukan prediksi.")
    else:
        st.info("Silakan lakukan labelling di Menu 3 terlebih dahulu atau unggah file CSV hasil labelling.")