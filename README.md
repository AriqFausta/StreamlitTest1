# Prediksi Konsumsi Energi Peralatan Rumah Tangga (Streamlit)

Aplikasi Streamlit dari notebook "TUGAS UTS PPD": membandingkan Linear Regression,
Decision Tree, Random Forest, dan Gradient Boosting pada dataset Appliances Energy
Prediction (UCI), lalu memprediksi konsumsi energi (Wh per 10 menit) dari input pengguna.

## Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Upload isi folder ini ke repository GitHub (app.py di root).
2. Buka https://share.streamlit.io, pilih repo, set Main file path = `app.py`, lalu Deploy.

## Catatan
- Dataset dimuat dari URL GitHub; jika gagal, unggah CSV lewat sidebar.
- Model dilatih saat aplikasi pertama kali dibuka (di-cache setelahnya).
- Memakai 27 fitur (semua kolom numerik kecuali Appliances & date), tanpa seleksi fitur Boruta.
