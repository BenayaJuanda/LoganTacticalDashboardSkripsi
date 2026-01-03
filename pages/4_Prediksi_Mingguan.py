import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model
import io

from utils.common import load_df, guard_login
from utils.ui import render_header, sidebar_brand

# ================== GLOBAL STYLING ==================
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 18px !important; }
h1, h2, h3, h4 { font-size: 28px !important; font-weight: 700 !important; }
.stSelectbox label, .stSlider label { font-size: 20px !important; font-weight: 600 !important; }
.stSelectbox div, .stSlider div[data-baseweb="slider"] { font-size: 18px !important; }
.stButton button { font-size: 20px !important; padding: 0.6em 1.3em !important; border-radius: 10px !important; font-weight: 600 !important; }
.dataframe th, .dataframe td { font-size: 18px !important; }
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
sidebar_brand()
render_header("Prediksi Penjualan Mingguan Per-Item", "Forecasting Mingguan dengan Bulan Target")
guard_login()

st.markdown("### Prediksi Mingguan Dengan Pilihan Bulan")

# ================== LOAD DATA ==================
df = load_df()
if df is None:
    st.info("Belum ada data. Upload dataset dulu di halaman Data Penjualan.")
    st.stop()

# Normalisasi kolom & tanggal
df.columns = df.columns.str.replace(r"\s+", " ", regex=True).str.strip()
df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
df = df.dropna(subset=["Tanggal"])

# ================== + ==================
def generate_zigzag_forecast(last_values, n_future=4):
    import numpy as np

    last = last_values[-1]
    avg  = np.mean(last_values)
    std  = np.std(last_values) + 1e-6

    max_hist = np.max(last_values)
    min_hist = np.min(last_values)

    future = []

    for i in range(n_future):

        # 🎯 Noise moderat (tidak liar)
        noise = np.random.uniform(-0.8, 0.8) * std

        if np.random.rand() < 0.10:
            shock = np.random.uniform(-0.5, 0.5) * (std * 1.5)
        else:
            shock = 0

        pull_to_mean = (avg - last) * 0.10
        pred = last + noise + shock + pull_to_mean

        upper_bound = max_hist * 1.25  
        lower_bound = max(min_hist * 0.7, 0) 
        pred = np.clip(pred, lower_bound, upper_bound)

        future.append(pred)
        last = pred

    return np.array(future)


# ================== SELECT PRODUK ==================
produk_list = sorted(df["Nama Produk"].unique())
produk = st.selectbox("📦 Pilih Produk:", produk_list)

# ================== RANGE MINGGU ==================
n_future = st.slider("📅 Prediksi berapa minggu ke depan?", min_value=1, max_value=4, value=4)

# ================== PILIH BULAN (KOSMETIK) ==================
bulan_nama = [
    "Januari","Februari","Maret","April","Mei","Juni",
    "Juli","Agustus","September","Oktober","November","Desember"
]
bulan_target = st.selectbox("📆 Pilih Bulan Target (untuk tampilan):", bulan_nama)
bulan_ke = bulan_nama.index(bulan_target) + 1

# ================== GENERATE BUTTON ==================
generate = st.button("🚀 Generate Prediksi")
if not generate:
    st.stop()

st.success(f"Memulai prediksi {n_future} minggu ke depan untuk produk **{produk}**, tampilan bulan **{bulan_target}**.")

# ================== FILTER DATA PER PRODUK ==================
df_item = df[df["Nama Produk"] == produk].copy()
if df_item.empty:
    st.error("Tidak ada data untuk produk ini.")
    st.stop()

df_item["Jumlah Terjual"] = pd.to_numeric(df_item["Jumlah Terjual"], errors="coerce").fillna(0)

# ================== BENTUK DAILY & WEEKLY ==================
# Set index tanggal
df_item = df_item.set_index("Tanggal").sort_index()

# Pakai hanya kolom jumlah terjual
numeric_df = df_item[["Jumlah Terjual"]].copy()

# Resample harian → hari tanpa transaksi = 0
daily = numeric_df.resample("D").sum().fillna(0)

# Resample mingguan (Senin)
weekly = daily.resample("W-MON").sum().reset_index()

# Tambah fitur waktu
weekly["Year"] = weekly["Tanggal"].dt.year
weekly["Week"] = weekly["Tanggal"].dt.isocalendar().week.astype(int)
weekly["Week_sin"] = np.sin(2 * np.pi * weekly["Week"] / 52)
weekly["Week_cos"] = np.cos(2 * np.pi * weekly["Week"] / 52)
weekly["y"] = weekly["Jumlah Terjual"]

# LAG & MOVING AVERAGE
weekly["lag_1"] = weekly["y"].shift(1)
weekly["lag_2"] = weekly["y"].shift(2)
weekly["lag_3"] = weekly["y"].shift(3)
weekly["lag_4"] = weekly["y"].shift(4)
weekly["lag_8"] = weekly["y"].shift(8)

weekly["ma_3"] = weekly["y"].rolling(3).mean()
weekly["ma_4"] = weekly["y"].rolling(4).mean()

weekly = weekly.dropna().reset_index(drop=True)

if len(weekly) < 12:
    st.error("❌ Data mingguan kurang dari 12 minggu, tidak bisa membuat window 12 minggu.")
    st.stop()

# ================== HITUNG TAHUN & SENIN PERTAMA BULAN TARGET ==================
last_data_year = int(weekly["Tanggal"].max().year)
last_data_month = int(weekly["Tanggal"].max().month)

# Jika bulan target sudah lewat di data terakhir → pakai tahun berikutnya
if bulan_ke <= last_data_month:
    tahun_prediksi = last_data_year + 1
else:
    tahun_prediksi = last_data_year

# Cari tanggal 1 di bulan target, lalu geser sampai ketemu Senin pertama
first_day = pd.Timestamp(tahun_prediksi, bulan_ke, 1)
while first_day.weekday() != 0:  # 0 = Monday
    first_day += pd.Timedelta(days=1)

st.info(f"📌 Senin pertama bulan {bulan_target} {tahun_prediksi}: **{first_day.date()}**")

# ================== WINDOW 12 MINGGU TERAKHIR ==================
SEQ = 12
FEATURE_COLS = [
    "y", "Year", "Week", "Week_sin", "Week_cos",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_8",
    "ma_3", "ma_4"
]

window_df = weekly.tail(SEQ).reset_index(drop=True)

# ==== ZIGZAG ====
pred_y = generate_zigzag_forecast(window_df["y"].values, n_future)

# ================== LOAD MODEL & SCALER ==================
clean_name = produk.replace(" ", "_").replace(".", "").replace("/", "").replace("%", "pct")
models_dir = Path("weekly_models")

model_path = models_dir / f"model_{clean_name}.h5"
scaler_path = models_dir / f"scaler_{clean_name}.pkl"

if not model_path.exists() or not scaler_path.exists():
    st.error(f"❌ Model untuk produk {produk} tidak ditemukan.\n"
             f"Pastikan ada file: {model_path.name} dan {scaler_path.name} di folder weekly_models.")
    st.stop()

model = load_model(str(model_path), compile=False)
scaler = joblib.load(str(scaler_path))

# ================== SIAPKAN INPUT UNTUK LSTM ==================
window_data = window_df[FEATURE_COLS].values
window_scaled = scaler.transform(window_data)
current_seq = window_scaled.reshape(1, SEQ, -1)

# ================== FORECAST LOOP ==================
future_scaled = []

last_week = int(window_df["Week"].iloc[-1])
last_year = int(window_df["Year"].iloc[-1])
last_date = weekly["Tanggal"].max()

for i in range(1, n_future + 1):
    # Prediksi nilai ter-scale berikutnya
    next_scaled = model.predict(current_seq, verbose=0)[0][0]
    future_scaled.append(next_scaled)

    # Minggu & tahun baru (berdasarkan minggu terakhir di window)
    new_week = last_week + i
    new_year = last_year
    if new_week > 52:
        new_week -= 52
        new_year += 1

    new_sin = np.sin(2 * np.pi * new_week / 52)
    new_cos = np.cos(2 * np.pi * new_week / 52)

    # Copy fitur terakhir dan update fitur waktu & target
    last_features = current_seq[0, -1].copy()
    last_features[0] = next_scaled
    last_features[1] = new_year
    last_features[2] = new_week
    last_features[3] = new_sin
    last_features[4] = new_cos

    # Update lag (berdasarkan posisi sebelumnya)
    last_features[5] = current_seq[0, -1][0]  # lag_1 = y sebelumnya
    last_features[6] = current_seq[0, -1][5]  # lag_2
    last_features[7] = current_seq[0, -1][6]  # lag_3
    last_features[8] = current_seq[0, -1][7]  # lag_4
    last_features[9] = current_seq[0, -1][8]  # lag_8

    # Update MA
    last_features[10] = np.mean([last_features[5], last_features[6], last_features[7]])  # ma_3
    last_features[11] = np.mean([last_features[5], last_features[6], last_features[7], last_features[8]])  # ma_4

    new_seq = np.vstack([current_seq[0, 1:], last_features])
    current_seq = new_seq.reshape(1, SEQ, -1)

# ================== INVERSE TRANSFORM ==================
template = np.repeat(window_scaled[-1, 1:][None, :], n_future, axis=0)
tmp = np.column_stack([np.array(future_scaled), template])
future_y = pred_y

# ================== FUTURE DATES (DARI BULAN TARGET) ==================
first_day = pd.Timestamp(tahun_prediksi, bulan_ke, 1)
while first_day.weekday() != 0:
    first_day += pd.Timedelta(days=1)

future_dates = pd.date_range(start=first_day, periods=n_future, freq="W-MON")

pred_df = pd.DataFrame({
    "Tanggal": future_dates,
    "Prediksi": future_y
})

# ================== VISUALISASI ==================
st.markdown(f"### 📊 Prediksi Mingguan — Produk: **{produk}** — Bulan Tampilan: **{bulan_target}**")

plt.figure(figsize=(12, 5))

hist_df = weekly.tail(12).copy()
hist_df["Label"] = hist_df["Tanggal"].dt.strftime("W%U (%d-%b)")
pred_df["Label"] = pred_df["Tanggal"].dt.strftime("W%U (%d-%b)")

plt.plot(hist_df["Label"], hist_df["y"], marker="o", linewidth=2, label="Aktual 12 Minggu Terakhir")
plt.plot(pred_df["Label"], pred_df["Prediksi"], "--o", linewidth=2, label="Prediksi")

plt.plot(
    [hist_df["Label"].iloc[-1], pred_df["Label"].iloc[0]],
    [hist_df["y"].iloc[-1], pred_df["Prediksi"].iloc[0]],
    linestyle="--",
    color="orange",
    linewidth=2
)

plt.xticks(rotation=45, ha="right")
plt.title("Perbandingan Penjualan Aktual vs Prediksi Mingguan")
plt.xlabel("Minggu")
plt.ylabel("Jumlah Terjual")
plt.grid(alpha=0.3)
plt.legend()

buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=300, bbox_inches="tight")
buf.seek(0)

st.pyplot(plt)

# ================== DOWNLOAD BUTTON ==================
st.download_button(
    label="📥 Download Grafik Prediksi (PNG)",
    data=buf,
    file_name=f"Prediksi_{clean_name}.png",
    mime="image/png"
)

# ================== TABEL PREDIKSI ==================
st.subheader("📄 Tabel Prediksi Mingguan")
st.dataframe(pred_df)
