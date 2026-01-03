# Streamlit Skripsi—Logan Tactical Dashboard (Starter)

## Cara jalanin
```bash
pip install -r requirements.txt
streamlit run app.py
```
Login demo: **admin / admin123** (untuk keperluan uji fungsi saja).

## Struktur
- `app.py` — halaman login & redirect
- `pages/1_Dashboard.py` — KPI & tren
- `pages/2_Data Penjualan.py` — upload & tabel historis (+ tombol Refresh)
- `pages/3_Prediksi Penjualan.py` — prediksi per produk (3/6/12 bulan)
- `data/sample_sales.csv` — contoh data agar langsung bisa dicoba
- `models/` — tempat meletakkan model LSTM (opsional)
- `utils/common.py` — helper untuk session/data
- `utils/model_stub.py` — stub prediksi (ganti dengan LSTM Anda)
