import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Muat model dan scaler
try:
    with open('models/knn_model.pkl', 'rb') as model_file:
        knn_model = pickle.load(model_file)
    with open('models/scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Pastikan 'knn_model.pkl' dan 'scaler.pkl' berada di direktori yang benar.")
    st.stop()

# Judul aplikasi
st.title("Aplikasi Prediksi Gender")
st.write("Masukkan data berikut untuk memprediksi gender:")

# Input numerik
forehead_width = st.number_input("Lebar dahi (cm):", min_value=0.0, step=0.1, format="%.2f")
forehead_height = st.number_input("Tinggi dahi (cm):", min_value=0.0, step=0.1, format="%.2f")

# Input kategori
distance_nose_to_lip_long = st.selectbox("Jarak hidung ke bibir panjang:", ["Pendek", "Panjang"])
lips_thin = st.selectbox("Bibir tipis:", ["Ya", "Tidak"])
long_hair = st.selectbox("Rambut panjang:", ["Ya", "Tidak"])
nose_long = st.selectbox("Hidung panjang:", ["Ya", "Tidak"])
nose_wide = st.selectbox("Hidung lebar:", ["Ya", "Tidak"])

# Konversi input kategori ke numerik
distance_nose_to_lip_long = 1 if distance_nose_to_lip_long == "Panjang" else 0
lips_thin = 1 if lips_thin == "Ya" else 0
long_hair = 1 if long_hair == "Ya" else 0
nose_long = 1 if nose_long == "Ya" else 0
nose_wide = 1 if nose_wide == "Ya" else 0

# Prediksi
if st.button("Prediksi"):
    try:
        # Preprocessing input
        input_data = np.array([[forehead_width, forehead_height, distance_nose_to_lip_long,
                                lips_thin, long_hair, nose_long, nose_wide]])
        processed_data = scaler.transform(input_data)

        # Prediksi dengan model
        prediction = knn_model.predict(processed_data)

        # Tampilkan hasil
        gender = "Laki-laki" if prediction[0] == 1 else "Perempuan"
        st.success(f"Hasil prediksi: {gender}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
