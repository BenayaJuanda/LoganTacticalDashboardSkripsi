import streamlit as st
from pathlib import Path
from utils.common import ensure_session_keys
from utils.ui import render_header, sidebar_brand

st.set_page_config(page_title="Logan Tactical — Streamlit", page_icon="🛡️", layout="wide")

ensure_session_keys()
sidebar_brand()
render_header("Logan Tactical Dashboard", "Login & Access Control")

st.title("🔐 Login — Logan Tactical Dashboard")

with st.form("login_form", clear_on_submit=False):
    u = st.text_input("Username", value="", autocomplete="username")
    p = st.text_input("Password", value="", type="password", autocomplete="current-password")
    submitted = st.form_submit_button("Login")
    if submitted:
        if u == "admin" and p == "admin123":
            st.session_state["logged_in"] = True
            st.success("Login berhasil. Buka menu di sidebar (Dashboard / Data Penjualan / Prediksi).")
        else:
            st.error("Username atau password salah.")

st.caption("Silahkan masukkan username dan password anda dengan benar.")
