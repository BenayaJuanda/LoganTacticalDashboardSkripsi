import streamlit as st
from utils.common import guard_login
from pathlib import Path
import pandas as pd
import json
from utils.ui import render_header, sidebar_brand

sidebar_brand()
render_header("About", "Aplikasi & Model Info")

guard_login()

st.markdown("## ℹ️ Tentang Aplikasi")

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Aplikasi")
    st.markdown("""
**Logan Tactical Dashboard** — aplikasi internal untuk kelola data penjualan, visualisasi tren, dan prediksi berbasis Stacked LSTM.
- Upload data real (CSV/XLSX)
- Dashboard KPI & tren bulanan
- Prediksi per produk (3/6/12 bulan)
""")
with c2:
    st.subheader("Versi")
    st.metric("App", "v1.0.0")
    st.metric("Streamlit", st.__version__)

st.divider()

c3, c4 = st.columns(2)

with c3:
    st.subheader("Model")
    model_path = Path("models/best_model_fixed.h5")
    scaler_path = Path("models/scaler_bundle_LOG.pkl")
    st.write(f"Model: {'✅' if model_path.exists() else '❌'} **models/best_model_fixed.h5**")
    st.write(f"Scaler: {'✅' if scaler_path.exists() else '❌'} **models/scaler_bundle_LOG.pkl**")

    metrics_path = Path("reports/metrics.json")
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            a, b, c = st.columns(3)
            a.metric("MAE", m.get("MAE", "-"))
            b.metric("RMSE", m.get("RMSE", "-"))
            c.metric("MAPE", f"{m.get('MAPE', '-') }%")
        except Exception as e:
            st.warning(f"Gagal membaca reports/metrics.json: {e}")
    else:
        st.info("reports/metrics.json belum tersedia.")

with c4:
    st.subheader("Data")
    cleaned = Path("data/cleaned.parquet")
    sample_csv = Path("data/sample_sales.csv")
    st.write(f"Data dibersihkan: {'✅' if cleaned.exists() else '❌'} **data/cleaned.parquet**")
    st.write(f"Contoh CSV: {'✅' if sample_csv.exists() else '❌'} **data/sample_sales.csv**")
    if cleaned.exists():
        try:
            df_info = pd.read_parquet(cleaned)
            st.caption(f"{len(df_info)} baris × {len(df_info.columns)} kolom")
            tmp = df_info.head(10).reset_index(drop=True)
            tmp.index = tmp.index + 1
            st.dataframe(tmp)
        except Exception as e:
            st.warning(f"Gagal membaca cleaned.parquet: {e}")

st.divider()

st.subheader("Cara Kerja Stacked LSTM dalam Platform ini")
st.markdown("""
**Stacked LSTM** adalah LSTM dengan beberapa lapisan bertumpuk untuk menangkap pola deret waktu yang lebih kompleks.

**Alur di aplikasi ini:**
1) Data transaksi harian diubah ke agregasi **bulanan per produk**.  
2) Dibuat fitur: **lag** penjualan, **moving average (MA)**, **sin/cos bulan** (musiman), serta **flag Promo (A/B/C/D)** dan **Holiday (1–4)**.  
3) Fitur dinormalisasi memakai **scaler** yang sama seperti saat training.  
4) Model **Stacked LSTM** memproses urutan fitur dan memprediksi **jumlah terjual** bulan berikutnya.  
5) Prediksi multi-bulan dilakukan **autoregressive**: output bulan t dipakai sebagai input t+1.  
6) Di halaman **Prediksi**, kamu bisa simulasi **Promo/Holiday** masa depan; flag fitur disuntikkan ke langkah prediksi sehingga memengaruhi output.

**Interpretasi cepat:**
- Akurasi bergantung pada konsistensi pola musiman/promosi di historis.  
- Metrik seperti **MAE/ RMSE/ MAPE** pada `reports/metrics.json` menunjukkan tingkat error model.  
- Fitur skenario membantu **uji strategi** (stok & campaign) sebelum eksekusi nyata.
""")

st.divider()

st.subheader("About Creator")
st.markdown("""
- Creator: Benaya Juanda (535220266)  
- Email: benayajuandaaa@gmail.com  
- Catatan: Aplikasi ini untuk kebutuhan internal & pengujian akademik skripsi **"Prediksi Penjualan Produk Logan Tactical Store Menggunakan Algoritma Stacked LSTM"**.
""")
