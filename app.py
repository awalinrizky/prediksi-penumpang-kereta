import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
from prophet.serialize import model_from_json

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Aplikasi Prediksi Penumpang Kereta",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Judul Aplikasi
st.title("🚂 Aplikasi Peramalan Volume Penumpang Kereta Api")
st.markdown("""
Aplikasi ini digunakan untuk memprediksi jumlah penumpang kereta api di masa depan 
menggunakan 3 algoritma AI sekaligus: **Prophet**, **ARIMA**, dan **Holt-Winters**.
""")
st.write("---")

# 3. Definisikan Kategori Moda Transportasi (Harus sama dengan yang di Colab)
kategori_opsi = {
    "Kereta Bandara": "Kereta_Bandara",
    "Non Jabodetabek (Jawa)": "Non_Jabodetabek_Jawa",
    "Non Jawa (Sumatera + Sulawesi)": "Non_Jawa_Sumatera_plus_Sulawesi",
    "Kereta cepat (Whoosh)": "Kereta_cepat_Whoosh",
    "MRT": "MRT",
    "Jawa (Jabodetabek+Non Jabodetabek)": "Jawa_JabodetabekplusNon_Jabodetabek",
    "LRT": "LRT"
}

# 4. Sidebar untuk Input Pengguna
st.sidebar.header("🎛️ Panel Kontrol")

# Pilihan Kategori Kereta
kategori_terpilih = st.sidebar.selectbox(
    "Pilih Moda Transportasi Kereta:",
    options=list(kategori_opsi.keys())
)

# Pilihan Jumlah Bulan ke Depan
jumlah_bulan = st.sidebar.slider(
    "Periode Prediksi (Bulan ke Depan):",
    min_value=1,
    max_value=12,
    value=3
)

file_suffix = kategori_opsi[kategori_terpilih]
folder_model = "saved_models"

# 5. Fungsi untuk Memuat Model (Caching agar aplikasi cepat saat dipindah-pindah menu)
@st.cache_resource
def load_all_models(suffix):
    prophet_model = None
    arima_model = None
    hw_model = None
    
    # Path file masing-masing model
    path_prophet = os.path.join(folder_model, f"prophet_{suffix}.json")
    path_arima = os.path.join(folder_model, f"arima_{suffix}.pkl")
    path_hw = os.path.join(folder_model, f"hw_{suffix}.pkl")
    
    # Load Prophet
    if os.path.exists(path_prophet):
        with open(path_prophet, 'r') as f:
            prophet_model = model_from_json(json.load(f))
            
    # Load ARIMA
    if os.path.exists(path_arima):
        with open(path_arima, 'rb') as f:
            arima_model = pickle.load(f)
            
    # Load Holt-Winters
    if os.path.exists(path_hw):
        with open(path_hw, 'rb') as f:
            hw_model = pickle.load(f)
            
    return prophet_model, arima_model, hw_model

# Eksekusi pemuatan model
model_prophet, model_arima, model_hw = load_all_models(file_suffix)

# 6. Proses Prediksi / Forecasting
if st.sidebar.button("📊 Jalankan Prediksi", type="primary"):
    
    with st.spinner("Sedang menghitung prediksi dari ketiga model AI..."):
        
        # Tempat menyimpan hasil
        hasil_prediksi = {}
        dates_df = None
        
        # --- PROPHET PREDICTION ---
        if model_prophet is not None:
            future = model_prophet.make_future_dataframe(periods=jumlah_bulan, freq='MS')
            forecast = model_prophet.predict(future)
            # Ambil n-bulan terakhir hasil forecast murni masa depan
            future_forecast = forecast.tail(jumlah_bulan)
            hasil_prediksi['Prophet'] = future_forecast['yhat'].values
            dates_df = future_forecast['ds'].dt.strftime('%Y-%m-%d').values
        
        # --- ARIMA PREDICTION ---
        if model_arima is not None:
            try:
                # Memanggil fungsi forecast bawaan statsmodels
                pred_arima = model_arima.forecast(steps=jumlah_bulan)
                # Jika output berupa pandas series, ambil array values-nya
                hasil_prediksi['ARIMA'] = pred_arima.values if hasattr(pred_arima, 'values') else pred_arima
            except Exception as e:
                hasil_prediksi['ARIMA'] = [np.nan] * jumlah_bulan
        
        # --- HOLT-WINTERS PREDICTION ---
        if model_hw is not None:
            try:
                pred_hw = model_hw.forecast(steps=jumlah_bulan)
                hasil_prediksi['Holt-Winters'] = pred_hw.values if hasattr(pred_hw, 'values') else pred_hw
            except Exception as e:
                hasil_prediksi['Holt-Winters'] = [np.nan] * jumlah_bulan

        # Jika tanggal gagal digenerate dari prophet, buat tanggal manual alternatif
        if dates_df is None:
            dates_df = [f"Bulan ke-{i+1}" for i in range(jumlah_bulan)]
            
        # 7. Tampilkan Output di Layout Utama
        st.subheader(f"📈 Hasil Analisis & Prediksi untuk: {kategori_terpilih}")
        
        # Membuat DataFrame Ringkasan Hasil
        df_hasil = pd.DataFrame({'Tanggal / Periode': dates_df})
        for nama_model, nilai in hasil_prediksi.items():
            df_hasil[nama_model] = nilai
            # Konversi tipe data ke integer untuk pembulatan jumlah manusia/penumpang
            df_hasil[nama_model] = df_hasil[nama_model].apply(lambda x: int(round(x)) if not np.isnan(x) else "Gagal")
            
        # Tampilkan Metrik Ringkasan (Mengambil bulan pertama prediksi sebagai sampel)
        col1, col2, col3 = st.columns(3)
        with col1:
            val_p = df_hasil['Prophet'].iloc[0] if 'Prophet' in df_hasil.columns else "N/A"
            st.metric(label="Prediksi Prophet (Bulan Depan)", value=f"{val_p:,}" if isinstance(val_p, int) else val_p)
        with col2:
            val_a = df_hasil['ARIMA'].iloc[0] if 'ARIMA' in df_hasil.columns else "N/A"
            st.metric(label="Prediksi ARIMA (Bulan Depan)", value=f"{val_a:,}" if isinstance(val_a, int) else val_a)
        with col3:
            val_h = df_hasil['Holt-Winters'].iloc[0] if 'Holt-Winters' in df_hasil.columns else "N/A"
            st.metric(label="Prediksi Holt-Winters (Bulan Depan)", value=f"{val_h:,}" if isinstance(val_h, int) else val_h)

        st.write(" ")
        
        # Pembagian Tab Tampilan: Tabel Data vs Grafik
        tab1, tab2 = st.tabs(["📋 Tabel Data Prediksi", "📊 Grafik Perbandingan"])
        
        with tab1:
            st.dataframe(df_hasil, use_container_width=True)
            st.caption("*Catatan: Angka di atas merupakan hasil pembulatan ke satuan terdekat karena menyatakan jumlah orang.")
            
        with tab2:
            # Membuat visualisasi perbandingan menggunakan matplotlib
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot line untuk tiap model yang sukses diprediksi
            for nama_model in hasil_prediksi.keys():
                # Ambil data murni (bukan string 'Gagal') untuk diplot
                y_plot = [float(x) if isinstance(x, (int, float)) else np.nan for x in hasil_prediksi[nama_model]]
                ax.plot(dates_df, y_plot, marker='o', label=f"Model {nama_model}", linewidth=2)
                
            ax.set_title(f"Adu Prediksi Masa Depan ({jumlah_bulan} Bulan) - {kategori_terpilih}", fontweight='bold')
            ax.set_ylabel("Estimasi Volume Penumpang")
            ax.set_xlabel("Periode")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Tampilkan objek grafik matplotlib ke dalam streamlit
            st.pyplot(fig)

else:
    # Tampilan awal saat tombol belum ditekan
    st.info("💡 Silakan pilih parameter di panel sebelah kiri, lalu klik tombol **Jalankan Prediksi** untuk melihat hasil analisis AI.")