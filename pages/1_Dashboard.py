import streamlit as st
import re, unicodedata
import altair as alt
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path
from utils.ui import export_chart_as_png
from utils.common import guard_login, load_df
from utils.model_infer import predict_with_lstm_for_product
from utils.ui import render_header, sidebar_brand, render_kpi_cards

sidebar_brand()
render_header("Logan Tactical Dashboard", "Sales Forecasting & Insights Platform")

st.cache_data.clear()

guard_login()
st.markdown("## 📊 Dashboard prediksi dalam satu tahun")

df = load_df()
if df is None:
    st.info("Belum ada data. Silakan upload dataset di halaman **Data Penjualan**.")
    st.stop()

df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
df = df.dropna(subset=["Tanggal"])
df["Jumlah Terjual"] = pd.to_numeric(df["Jumlah Terjual"], errors="coerce").fillna(0).astype(int)

def _coerce_money(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^\d,.\-]", "", regex=True)
    s = s.apply(lambda x: x.replace(".", "").replace(",", ".") if ("," in x and "." in x) else x)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

if "Harga" in df.columns:
    df["Harga"] = _coerce_money(df["Harga"]).fillna(0.0)

def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).lower()
    s = re.sub(r"[^\w]+", "", s)
    s = (s
         .replace("keuntungan", "untung")
         .replace("profit", "untung")
         .replace("laba", "untung")
         .replace("pendapatan", "revenue"))
    return s

norm_cols = {orig: _norm_name(orig) for orig in df.columns}

def _find_col(require_all: list[str], forbid_any: list[str] = None) -> str | None:
    forbid_any = forbid_any or []
    for orig, n in norm_cols.items():
        if all(k in n for k in require_all) and all(k not in n for k in forbid_any):
            return orig
    return None

col_profit_unit  = _find_col(["untung", "unit"], forbid_any=["total"])
col_profit_total = _find_col(["untung", "total"])

if col_profit_unit is None:
    for name in ["Keuntungan/Unit", "Keuntungan Unit", "Profit/Unit", "Keuntungan per unit",
                 "Keuntungan_per_unit", "Profit per unit", "Profit_per_unit"]:
        if name in df.columns:
            col_profit_unit = name
            break

if col_profit_total is None:
    for name in ["KeuntunganTotal", "Keuntungan total", "Keuntungan_total",
                 "Total Keuntungan", "Total_Keuntungan", "TotalProfit", "ProfitTotal"]:
        if name in df.columns:
            col_profit_total = name
            break

df["_profit_unit"]  = pd.NA
df["_profit_total"] = pd.NA

if col_profit_unit:
    df["_profit_unit"] = _coerce_money(df[col_profit_unit])

if col_profit_total:
    df["_profit_total"] = _coerce_money(df[col_profit_total])

if (df["_profit_unit"].isna().all() or df["_profit_unit"].fillna(0).eq(0).all()) and col_profit_total:
    qty_col = None
    for cand in ["Jumlah Terjual", "Jumlah", "Qty", "Quantity", "Kuantitas"]:
        if cand in df.columns:
            qty_col = cand
            break
    if qty_col:
        qty = pd.to_numeric(df[qty_col], errors="coerce")
        tot = pd.to_numeric(df["_profit_total"], errors="coerce")
        df["_profit_unit"] = tot / qty.replace(0, pd.NA)

if (df["_profit_unit"].isna().all() or df["_profit_unit"].fillna(0).eq(0).all()) and df["_profit_total"].notna().any():
    tot_profit_hist = pd.to_numeric(df["_profit_total"], errors="coerce").fillna(0).sum()
    tot_units_hist  = pd.to_numeric(df["Jumlah Terjual"], errors="coerce").fillna(0).sum()
    df["_profit_unit"] = float(tot_profit_hist / tot_units_hist) if tot_units_hist > 0 else 0.0

df["_profit_unit"]  = pd.to_numeric(df["_profit_unit"], errors="coerce")
df["_profit_total"] = pd.to_numeric(df["_profit_total"], errors="coerce")

produk_list = sorted(df["Nama Produk"].dropna().unique().tolist())

@st.cache_data(show_spinner=False)
def read_metrics_json():
    p = Path("reports/metrics.json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

@st.cache_data(show_spinner=True, ttl=300)
def compute_kpi(df_in: pd.DataFrame, products, horizon: int = 12):
    total_units_pred = 0
    total_profit_pred = 0.0
    for prod in products:
        sub = df_in[df_in["Nama Produk"] == prod]
        if sub.empty:
            continue
        try:
            yhat = predict_with_lstm_for_product(df_in, prod, horizon)
        except Exception:
            continue
        units_pred = int(sum(yhat))
        total_units_pred += units_pred
        avg_profit = pd.to_numeric(sub["_profit_unit"], errors="coerce")
        avg_profit = float(avg_profit.dropna().median()) if avg_profit.notna().any() else None
        if (avg_profit is None) or (avg_profit == 0):
            tot_profit_hist = pd.to_numeric(sub["_profit_total"], errors="coerce").dropna().sum()
            tot_units_hist  = pd.to_numeric(sub["Jumlah Terjual"], errors="coerce").dropna().sum()
            if tot_units_hist > 0 and tot_profit_hist > 0:
                avg_profit = float(tot_profit_hist / tot_units_hist)
        if avg_profit is None or pd.isna(avg_profit):
            avg_profit = 0.0
        total_profit_pred += units_pred * avg_profit
    return int(total_units_pred), int(round(total_profit_pred))

@st.cache_data(show_spinner=False)
def build_monthly_agg(df_in: pd.DataFrame):
    daily_sales = df_in.groupby("Tanggal", as_index=True)["Jumlah Terjual"].sum()
    monthly_sales = daily_sales.resample("MS").sum()
    if "Harga" in df_in.columns:
        df_in["Harga"] = pd.to_numeric(df_in["Harga"], errors="coerce").fillna(0.0)
        df_in["Revenue_item"] = df_in["Harga"] * df_in["Jumlah Terjual"]
        daily_rev = df_in.groupby("Tanggal", as_index=True)["Revenue_item"].sum()
        monthly_rev = daily_rev.resample("MS").sum()
    else:
        monthly_rev = pd.Series(dtype=float)
    monthly = pd.DataFrame({"Jumlah Terjual": monthly_sales})
    if not monthly_rev.empty:
        monthly["Revenue"] = monthly_rev
    return monthly

with st.spinner("Menghitung KPI dari model & data..."):
    pred_units_12m, pred_profit_12m = compute_kpi(df, produk_list, horizon=12)

metrics = read_metrics_json()
mape = metrics.get("MAPE", None)
smape = metrics.get("sMAPE", None)
wape = metrics.get("WAPE", None)

acc_label = "—"
acc_help = "Akurasi estimatif berbasis MAPE (informal)."
try:
    if mape is not None:
        mape_f = float(mape)
        acc_label = f"{(100.0 - mape_f):.1f}%"
        parts = []
        parts.append(f"MAPE {mape_f:.2f}%")
        if smape is not None:
            parts.append(f"sMAPE {float(smape):.2f}%")
        if wape is not None:
            parts.append(f"WAPE {float(wape):.2f}%")
        acc_help = " | ".join(parts)
except Exception:
    pass

c1, c2, c3 = st.columns(3)
c1.metric("Prediksi Penjualan / Tahun", f"{pred_units_12m:,.0f}")
c2.metric("Akurasi (estimatif)", acc_label, help=acc_help)
c3.metric("Prediksi Keuntungan / Tahun", f"Rp {pred_profit_12m:,.0f}")

st.subheader("📦 Ringkasan Penjualan per Produk (12 Bulan Terakhir)")

df_last12 = df.copy()
df_last12["Tanggal"] = pd.to_datetime(df_last12["Tanggal"], errors="coerce")
cutoff = df_last12["Tanggal"].max() - pd.DateOffset(months=12)
df_last12 = df_last12[df_last12["Tanggal"] >= cutoff]

summary = (
    df_last12.groupby("Nama Produk")["Jumlah Terjual"]
    .agg(["sum", "mean"])
    .rename(columns={
        "sum": "Total 12 Bulan",
        "mean": "Rata-rata / Bulan"
    })
    .sort_values("Total 12 Bulan", ascending=False)
)

summary_view = summary.reset_index()
summary_view.index = summary_view.index + 1

st.dataframe(summary_view)

monthly = build_monthly_agg(df) 

def _aggregate_pred_monthly(df_in: pd.DataFrame, products, horizon: int = 12) -> pd.Series:
    agg = {}
    for prod in products:
        sub = df_in[df_in["Nama Produk"] == prod].copy()
        if sub.empty:
            continue
        sub["Tanggal"] = pd.to_datetime(sub["Tanggal"], errors="coerce")
        sub = sub.dropna(subset=["Tanggal"]).sort_values("Tanggal")
        if sub.empty:
            continue
        daily = sub.groupby("Tanggal")["Jumlah Terjual"].sum()
        hist_m = daily.resample("MS").sum()
        last_month = hist_m.index.max() if len(hist_m) else pd.Timestamp.today().normalize()
        future_idx = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        try:
            yhat = predict_with_lstm_for_product(df_in, prod, horizon)
        except Exception:
            continue
        for i, ts in enumerate(future_idx):
            agg[ts] = agg.get(ts, 0) + int(yhat[i])
    return pd.Series(agg).sort_index()

def _event_label(ts: pd.Timestamp) -> str | None:
    m = int(ts.month)
    if m == 1:  return "Tahun Baru"
    if m == 4:  return "Idul Fitri"
    if m == 8:  return "HUT RI"
    if m == 12: return "Natal"
    return None

pred_series_all = _aggregate_pred_monthly(df, produk_list, horizon=12)
hist_last12 = None
if "Jumlah Terjual" in monthly.columns and not monthly.empty:
    hist_last12 = monthly["Jumlah Terjual"].tail(12)

summary_lines = []
if pred_series_all is not None and len(pred_series_all):
    top_pred = pred_series_all.nlargest(3)
    best_ts = top_pred.index[0]
    best_evt = _event_label(best_ts)
    summary_lines.append(f"Periode prediksi tertinggi: **{best_ts.strftime('%B %Y')}** ≈ **{int(top_pred.iloc[0]):,}** unit" + (f" — terkait **{best_evt}**" if best_evt else ""))
    if len(top_pred) > 1:
        second_ts = top_pred.index[1]
        summary_lines.append(f"Kedua tertinggi: **{second_ts.strftime('%B %Y')}** ≈ **{int(top_pred.iloc[1]):,}** unit")
    if len(top_pred) > 2:
        third_ts = top_pred.index[2]
        summary_lines.append(f"Ketiga tertinggi: **{third_ts.strftime('%B %Y')}** ≈ **{int(top_pred.iloc[2]):,}** unit")
    if hist_last12 is not None and len(hist_last12):
        htop = hist_last12.nlargest(1)
        hts = htop.index[0]
        summary_lines.append(f"Historis 12 bulan terakhir tertinggi: **{hts.strftime('%B %Y')}** ≈ **{int(htop.iloc[0]):,}** unit")
        if hts.month == best_ts.month:
            summary_lines.append("Polanya konsisten: bulan puncak historis selaras dengan bulan puncak prediksi.")
        else:
            summary_lines.append("Bulan puncak historis berbeda dengan prediksi; potensi pergeseran permintaan.")
else:
    summary_lines.append("Belum ada rangkuman karena prediksi bulanan gabungan belum tersedia.")

st.subheader("📌 Catatan Prediksi")
for line in summary_lines:
    st.markdown(f"- {line}")

with st.expander("Lihat tabel prediksi bulanan (gabungan semua produk)"):
    pred_df_view = pred_series_all.reset_index()
    pred_df_view.columns = ["Periode", "Prediksi Total"]
    pred_df_view["Label"] = pred_df_view["Periode"].dt.strftime("%Y-%m")
    tmpv = pred_df_view[["Label", "Prediksi Total"]].copy()
    tmpv = tmpv.reset_index(drop=True)
    tmpv.index = tmpv.index + 1
    st.dataframe(tmpv)
    st.line_chart(pred_df_view.set_index("Label")["Prediksi Total"])

st.subheader("Tren Penjualan (Aktual)")
if not monthly.empty and "Jumlah Terjual" in monthly.columns:
    tmp = monthly.copy()
    tmp.index = tmp.index.strftime("%Y-%m")
    st.line_chart(tmp["Jumlah Terjual"])
else:
    st.write("Belum ada agregasi bulanan.")

import matplotlib.ticker as ticker

fig, ax = plt.subplots(figsize=(8, 3))
idx = pd.to_datetime(monthly.index)
ax.plot(idx, monthly["Jumlah Terjual"], marker="o", color="blue")
labels = idx.strftime("%Y-%m")
ax.set_xticks(idx)
ax.set_xticklabels(labels, rotation=45, ha="right")

ax.set_title("Tren Penjualan Bulanan")
ax.set_xlabel("Periode (YYYY-MM)")
ax.set_ylabel("Jumlah Terjual")
ax.grid(True, alpha=0.3)

buf = export_chart_as_png(fig)
st.download_button(
    "⬇️ Download Grafik Penjualan (PNG)",
    data=buf,
    file_name="tren_penjualan_bulanan.png",
    mime="image/png",
)

st.subheader("Tren Pendapatan (Aktual)")
if not monthly.empty and "Revenue" in monthly.columns:
    tmp2 = monthly.copy()
    tmp2.index = tmp2.index.strftime("%Y-%m")
    st.line_chart(tmp2["Revenue"])
else:
    st.write("Belum ada kolom `Harga`, sehingga revenue belum bisa dihitung.")

if not monthly.empty and "Revenue" in monthly.columns:
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    idx = pd.to_datetime(monthly.index)
    ax2.plot(idx, monthly["Revenue"], marker="o", color="green")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x):,}".replace(",", ".")))
    labels = idx.strftime("%Y-%m")
    ax2.set_xticks(idx)
    ax2.set_xticklabels(labels, rotation=45, ha="right")

    ax2.set_title("Tren Pendapatan Bulanan")
    ax2.set_xlabel("Periode (YYYY-MM)")
    ax2.set_ylabel("Revenue (Rp)")
    ax2.grid(True, alpha=0.3)

    png_file2 = export_chart_as_png(fig2)
    st.download_button(
        label="⬇️ Download Grafik Revenue (PNG)",
        data=png_file2,
        file_name="tren_revenue.png",
        mime="image/png"
)
    
st.subheader("📅 Tren Penjualan Harian (Actual)")

df_daily = df.groupby("Tanggal")["Jumlah Terjual"].sum().reset_index()
df_daily.columns = ["Tanggal", "Jumlah Terjual"]

last_date = df_daily["Tanggal"].max()
cutoff_daily = last_date - pd.Timedelta(days=365)
df_daily = df_daily[df_daily["Tanggal"] >= cutoff_daily]

if not df_daily.empty:
    chart = (
        alt.Chart(df_daily)
        .mark_line(point=True)
        .encode(
            x=alt.X('Tanggal:T', title='Tanggal', axis=alt.Axis(format='%Y-%m-%d')),
            y=alt.Y('Jumlah Terjual:Q', title='Jumlah Terjual'),
            tooltip=['Tanggal:T', 'Jumlah Terjual:Q']
        )
        .properties(
            width=900,
            height=350,
            title="Tren Penjualan Harian (1 Tahun Terakhir)"
        )
        .interactive()  
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.write("Belum ada data harian yang mencukupi.")

with st.expander("🔎 Debug Profit Columns"):
    st.write("Detected _profit_unit values:", df["_profit_unit"].notna().sum())
    st.write("Detected _profit_total values:", df["_profit_total"].notna().sum())

    st.write("col_profit_unit:", col_profit_unit)
    st.write("col_profit_total:", col_profit_total)

    st.write("sum(_profit_total):", float(pd.to_numeric(df["_profit_total"], errors="coerce").fillna(0).sum()))
    st.write("sum(qty):", int(pd.to_numeric(df["Jumlah Terjual"], errors="coerce").fillna(0).sum()))

    sampel = df[["Nama Produk","Harga","Jumlah Terjual","_profit_unit","_profit_total"]].head(15)
    sampel = sampel.reset_index(drop=True)
    sampel.index = sampel.index + 1
    st.dataframe(sampel)
    st.write("Kolom df:", list(df.columns))
    st.write("Contoh 5 baris:")
    st.dataframe(df.head(5).reset_index(drop=True).assign(_idx=lambda d: d.index+1).set_index("_idx"))


