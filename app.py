import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Muat model dan scaler
with open('models/knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)
with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Judul aplikasi
st.title("Aplikasi Prediksi Gender")
st.write("Masukkan data berikut untuk memprediksi gender:")

# Input dari pengguna untuk semua fitur
forehead_width = st.number_input("Lebar dahi (cm):", min_value=0.0, step=0.1, format="%.2f")
forehead_height = st.number_input("Tinggi dahi (cm):", min_value=0.0, step=0.1, format="%.2f")
distance_nose_to_lip_long = st.number_input("Jarak hidung ke bibir panjang (cm):", min_value=0.0, step=0.1, format="%.2f")
lips_thin = st.number_input("Ketebalan bibir (cm):", min_value=0.0, step=0.1, format="%.2f")
long_hair = st.number_input("Panjang rambut (cm):", min_value=0.0, step=0.1, format="%.2f")
nose_long = st.number_input("Panjang hidung (cm):", min_value=0.0, step=0.1, format="%.2f")
nose_wide = st.number_input("Lebar hidung (cm):", min_value=0.0, step=0.1, format="%.2f")

# Prediksi
if st.button("Prediksi"):
    try:
        # Buat input data baru dalam format DataFrame
        input_data = pd.DataFrame([[
            forehead_width, forehead_height, distance_nose_to_lip_long, 
            lips_thin, long_hair, nose_long, nose_wide
        ]], columns=[
            'forehead_width_cm', 'forehead_height_cm', 'distance_nose_to_lip_long',
            'lips_thin', 'long_hair', 'nose_long', 'nose_wide'
        ])

        # Preprocessing input
        processed_data = scaler.transform(input_data)

        # Prediksi dengan model
        prediction = knn_model.predict(processed_data)

        # Tampilkan hasil
        gender = "Laki-laki" if prediction[0] == 1 else "Perempuan"
        st.success(f"Hasil prediksi: {gender}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
