import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from utils.common import guard_login, load_df, set_df, clear_data
from utils.ui import render_header, sidebar_brand

sidebar_brand()
render_header("Data Penjualan", "Upload, Mapping, dan Validasi")

guard_login()
st.markdown("## 📦 Data Penjualan") 

REQUIRED = ["Tanggal", "ID Produk", "Nama Produk", "Brand", "Kategori", "Harga", "Jumlah Terjual", "Keuntungan per unit", "Keuntungan total"]

def parse_tanggal(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return pd.to_datetime(series, errors="coerce", unit="D", origin="1899-12-30")
    s = series.astype(str).str.strip().str.replace("/", "-", regex=False)
    dt = pd.to_datetime(s, format="%d-%m-%y", errors="coerce")
    if dt.isna().mean() > 0.2:
        dt = pd.to_datetime(s, format="%d-%m-%Y", errors="coerce")
    if dt.isna().mean() > 0.2:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return dt

def to_int_series(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biu":
        return s.astype(int)
    s2 = (
        s.astype(str)
         .str.replace(r"[^\d\-\.,]", "", regex=True)
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(s2, errors="coerce").fillna(0).astype(int)

def infer_kategori_from_nama(nm: str) -> str:
    if not isinstance(nm, str):
        return "Airsoft Gun"
    nm_low = nm.lower()
    if any(k in nm_low for k in ["bb", "peluru", "ammo", "magazine", "mag", "gas", "co2"]):
        return "Aksesori"
    if "operator" in nm_low:
        return "Aksesori"
    return "Airsoft Gun"

def read_any(uploaded, sheet=None, header_row=0):
    name = (uploaded.name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded, engine="openpyxl", sheet_name=sheet, header=header_row)
    else:
        try:
            return pd.read_csv(uploaded)
        except UnicodeDecodeError:
            uploaded.seek(0)
            return pd.read_csv(uploaded, encoding="latin-1")

st.markdown("### Upload Dataset (CSV/Excel)")
uploaded = st.file_uploader("Unggah file .csv / .xlsx / .xls", type=["csv", "xlsx", "xls"])

df_preview = None
sheet = None

if uploaded is not None:
    name = uploaded.name.lower()
    is_excel = name.endswith(".xlsx") or name.endswith(".xls")
    if is_excel:
        try:
            xl = pd.ExcelFile(uploaded)
            sheets = xl.sheet_names
            st.info(", ".join(sheets))
            sheet = st.selectbox("Pilih sheet", options=sheets, index=0, key="sheet_choice")
            uploaded.seek(0)
        except Exception as e:
            st.error(f"Gagal membaca sheet: {e}")
            uploaded.seek(0)

    try:
        df_raw = read_any(uploaded, sheet=sheet, header_row=0)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        df_raw = None

    if df_raw is not None and not df_raw.empty:
        if df_raw.shape[1] < 10:
            st.error("Jumlah kolom kurang dari 10. Dibutuhkan kolom A..J (A=Kode Barang, B=Nama, C=Harga, D, E, F=Jumlah, G=Tanggal dd-mm-yy, H=Brand, I=Promotion, J=Holiday).")
        else:
            col_A = df_raw.iloc[:, 0]
            col_B = df_raw.iloc[:, 1]
            col_C = df_raw.iloc[:, 2]
            col_D = df_raw.iloc[:, 3]
            col_E = df_raw.iloc[:, 4]
            col_F = df_raw.iloc[:, 5]
            col_G = df_raw.iloc[:, 6]
            col_H = df_raw.iloc[:, 7]
            col_I = df_raw.iloc[:, 8]
            col_J = df_raw.iloc[:, 9]

            out = pd.DataFrame({
                "ID Produk": to_int_series(col_A),
                "Nama Produk": col_B.astype(str).str.strip(),
                "Harga": to_int_series(col_C),
                "Keuntungan per unit": to_int_series(col_D),
                "Keuntungan total": to_int_series(col_E),
                "Jumlah Terjual": to_int_series(col_F),
                "Brand": col_H.astype(str).str.strip(),
                "Promotion": col_I.astype(str).str.strip(),
                "Holiday": col_J
            })

            out["Tanggal"] = parse_tanggal(col_G)
            out["Kategori"] = out["Nama Produk"].apply(infer_kategori_from_nama)

            before = len(out)
            out = out.dropna(subset=["Tanggal"])
            if before - len(out) > 0:
                st.warning(f"{before - len(out)} baris dibuang karena tanggal tidak valid (harus dd-mm-yy).")

            out = out[REQUIRED + ["Promotion", "Holiday"]]
            set_df(out)
            try:
                Path("data").mkdir(exist_ok=True)
                out.to_parquet("data/cleaned.parquet", index=False)
            except Exception:
                pass
            st.success("Dataset real berhasil dimuat.")
            df_preview = out

st.markdown("### Data Penjualan Historis")
df_session = load_df()

if df_session is None and df_preview is None:
    st.warning("Belum ada data. Unggah file real kamu di atas.")
else:
    df_show = df_preview if df_preview is not None else df_session
    tmp = df_show.head(500).reset_index(drop=True)
    tmp.index = tmp.index + 1
    st.dataframe(tmp)
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("🔄 Refresh Data (Clear)"):
            clear_data()
            st.success("Data dihapus dari sesi.")
    with c2:
        st.download_button(
            "⬇️ Export Data (CSV)",
            data=df_show.to_csv(index=False).encode("utf-8"),
            file_name="data_penjualan_clean.csv",
            mime="text/csv"
        )
