#library
from requests import options
import streamlit as st
from google_play_scraper import Sort, reviews
import pandas as pd
import datetime as time

#HEADER
st.set_page_config(page_title="**SCRAPPER**", layout = "wide")
st.title("GOOGLE PLAY SCRAPPER REGION INDONESIA CUYS 🔥")
st.write("aplikasi scrapper ini diperuntukan sebagai tempat untuk menguji penelitian skripsi")
st.write("MOCH. IHSANUL HADI")

#MENU SIDEBAR INPUT
st.sidebar.header("Setting Scraper")
sidebar_menu = st.sidebar.radio(
    "Pilih Menu:",
    [
        "1. Scraping Data", 
        "2. soon", 
        "3. soon", 
        "4. soon",
        "5. soon"
    ]
)
if sidebar_menu == "1. Scraping Data":
    st.header("1. Scraping Ulasan Aplikasi")
    st.write("Ambil ulasan secara real-time dari Google Play Store ya cuyss 🔥.")

    col1, col2 = st.columns(2)
    with col1:
        app_package = st.text_input("App Package Name:", value="com.mobile.legends")
        target_count = st.number_input("Target Jumlah Data:", min_value=1, max_value=10000, value=1, step=100)
    with col2:
        start_year = st.number_input("Tahun Mulai:", min_value=2015, max_value=2026, value=2015)
        end_year = st.number_input("Tahun Selesai:", min_value=2015, max_value=2026, value=2026)

    if st.button("Mulai Scraping"):
        with st.spinner("Sedang mengambil data ulasan cuyss🔥..."):
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
                st.warning("Nooo!! gaada datanyaa.....😭😭😭")