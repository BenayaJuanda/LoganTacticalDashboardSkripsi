import streamlit as st
import pandas as pd
from utils.common import guard_login, load_df
from utils.model_infer import predict_with_lstm_for_product
from utils.ui import render_header, sidebar_brand

sidebar_brand()
render_header("Prediksi Penjualan", "Baseline vs Scenario Simulation")

guard_login()
st.title("📈 Prediksi Penjualan Per Item")

df = load_df()
if df is None:
    st.info("Belum ada data. Upload dataset di halaman **Data Penjualan** terlebih dahulu.")
    st.stop()

produk_list = sorted(df["Nama Produk"].dropna().unique().tolist())

col1, col2 = st.columns([2,1])
with col1:
    produk = st.selectbox("Pilih Produk", options=produk_list)
with col2:
    horizon_label = st.selectbox("Horizon Prediksi", options=["3 bulan", "6 bulan", "12 bulan"])
    horizon = int(horizon_label.split()[0])

promo_options = {
    "Tidak Ada Promo": None,
    "A – Promotion (10% BB Buying)": "A",
    "B – Promotion (15% BB Buying & 10% Unit Buying)": "B",
    "C – Promotion (17% BB Buying)": "C",
    "D – Promotion (15% BB & Unit Buying)": "D",
}

holiday_options = {
    "Tidak Ada Holiday": None,
    "1 – New Year": "1",
    "2 – Idul Fitri": "2",
    "3 – Kemerdekaan": "3",
    "4 – Christmas": "4",
}

col_s1, col_s2 = st.columns(2)
with col_s1:
    promo_label = st.selectbox("Skenario Promo", list(promo_options.keys()))
    promo_choice = promo_options[promo_label]

with col_s2:
    holi_label = st.selectbox("Skenario Holiday", list(holiday_options.keys()))
    holi_choice = holiday_options[holi_label]

if st.button("🚀 Generate Prediksi", type="primary"):
    try:
        sub = df[df["Nama Produk"] == produk].copy()
        sub["Tanggal"] = pd.to_datetime(sub["Tanggal"], errors="coerce")
        sub = sub.dropna(subset=["Tanggal"]).sort_values("Tanggal")
        daily = sub.groupby("Tanggal")["Jumlah Terjual"].sum()
        monthly = daily.resample("MS").sum()

        last_month = monthly.index.max() if len(monthly) else pd.Timestamp.today().normalize()
        future_index = pd.date_range((last_month + pd.offsets.MonthBegin(1)), periods=horizon, freq="MS")

        yhat_base = predict_with_lstm_for_product(df, produk, horizon)
        promo_param = promo_choice
        holi_param = int(holi_choice) if holi_choice is not None else None
        yhat_scn = predict_with_lstm_for_product(df, produk, horizon, promo_code=promo_param, holi_code=holi_param)

        pred_df = pd.DataFrame({
            "Periode": future_index,
            "Baseline": yhat_base,
            "Skenario": yhat_scn
        })
        pred_df["Label"] = pred_df["Periode"].dt.strftime("%Y-%m")

        st.success(f"Prediksi {produk} untuk {horizon} bulan (Baseline vs Skenario)")
        tbl = pred_df[["Label","Baseline","Skenario"]].copy().reset_index(drop=True)
        tbl.index = tbl.index + 1
        st.dataframe(tbl)

        tab1, tab2 = st.tabs(["📊 Actual 12 Bulan Terakhir", "🔮 Baseline vs Skenario"])
        with tab1:
            if len(monthly) == 0:
                st.info("Data aktual belum tersedia.")
            else:
                last12 = monthly.tail(12).copy()
                last12_df = last12.reset_index()
                last12_df["Label"] = last12_df["Tanggal"].dt.strftime("%Y-%m")
                chart_a = pd.DataFrame({
                    "Periode": last12_df["Label"],
                    "Actual": last12_df["Jumlah Terjual"].astype(int)
                })
                st.line_chart(chart_a.set_index("Periode"))
        with tab2:
            chart_f = pred_df.set_index("Label")[["Baseline","Skenario"]]
            st.line_chart(chart_f)
    except Exception as e:
        st.error(f"Gagal membuat prediksi: {e}")
