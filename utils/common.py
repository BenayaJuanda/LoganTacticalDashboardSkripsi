
import streamlit as st
import pandas as pd

SESSION_KEYS = {
    "logged_in": False,
    "df": None,
    "metrics": {},
}

def ensure_session_keys():
    for k, default in SESSION_KEYS.items():
        if k not in st.session_state:
            st.session_state[k] = default

def guard_login():
    ensure_session_keys()
    if not st.session_state.get("logged_in", False):
        st.warning("Silakan login terlebih dahulu di halaman utama.")
        st.stop()

def load_df() -> pd.DataFrame | None:
    return st.session_state.get("df")

def set_df(df: pd.DataFrame | None):
    st.session_state["df"] = df

def clear_data():
    st.session_state["df"] = None
    st.session_state["metrics"] = {}

def compute_basic_metrics(df: pd.DataFrame):
    # Expect columns: Tanggal, Harga, Jumlah Terjual
    if df is None or df.empty: 
        return {}
    tmp = df.copy()
    tmp["Tanggal"] = pd.to_datetime(tmp["Tanggal"], errors="coerce")
    tmp = tmp.dropna(subset=["Tanggal"])
    tmp["Revenue"] = tmp["Harga"].astype(float) * tmp["Jumlah Terjual"].astype(float)
    monthly = tmp.groupby(tmp["Tanggal"].dt.to_period("M")).agg({
        "Jumlah Terjual": "sum",
        "Revenue": "sum"
    }).sort_index()
    monthly.index = monthly.index.to_timestamp()
    total_year_sales = monthly["Jumlah Terjual"].sum()
    total_year_revenue = monthly["Revenue"].sum()
    # Placeholder accuracy (ubah setelah integrasi model)
    acc = 0.88 if len(monthly) >= 6 else 0.85
    return {
        "monthly": monthly,
        "pred_sales_year": int(total_year_sales * 1.05),  # naive uplift
        "pred_profit_year": int(total_year_revenue * 0.18),  # asumsi margin 18%
        "model_accuracy": acc
    }
