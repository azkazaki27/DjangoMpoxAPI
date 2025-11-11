from django.http import JsonResponse
from django.conf import settings
from django.http import HttpResponse
import pandas as pd
import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
from django.views.decorators.csrf import csrf_exempt
import joblib
from django.http import JsonResponse
from django.conf import settings
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from skimage.feature import graycomatrix, graycoprops

# --- Pemuatan Model dan Scaler (Tidak Berubah) ---
model_path = os.path.join(settings.BASE_DIR, 'deteksi', 'data', 'model_monkeypox_RF.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'deteksi', 'data', 'scaler_fix.pkl')

with open(model_path, 'rb') as f:
    model = joblib.load(f)

with open(scaler_path, 'rb') as f:
    scaler = joblib.load(f)

# --- Pemuatan VGG19 (Tidak Berubah) ---
base_model = VGG19(weights='imagenet', include_top=True)
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def homepage(request):
    return HttpResponse("Selamat datang di API Skripsi!")

def index(request):
    return HttpResponse("Halaman utama Django API Skripsi")

# --- Fungsi GLCM (Tidak Berubah) ---
def extract_glcm_features(image_gray):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    # Normalisasi tipe data (sudah benar)
    image_gray_normalized = image_gray.astype(np.uint8)
    glcm = graycomatrix(image_gray_normalized, distances=distances, angles=angles, symmetric=True, normed=True)
    features = []
    props = ['contrast', 'energy', 'homogeneity', 'correlation']
    for prop in props:
        vals = graycoprops(glcm, prop)[0]
        features.extend(vals)
    return features 

# --- MODIFIKASI FUNGSI VGG19 ---
# Sekarang menerima gambar yang SUDAH di-resize
def extract_vgg19_features(image_rgb_resized):
    # HAPUS RESIZING: image_resized = cv2.resize(image_rgb, (224, 224))
    image_array = img_to_array(image_rgb_resized)
    image_preprocessed = preprocess_input(np.expand_dims(image_array, axis=0))
    # Nonaktifkan progress bar (verbose=0) agar log API bersih
    features = vgg_model.predict(image_preprocessed, verbose=0)
    return features.flatten()

@csrf_exempt
def predict(request):
    try:
        if request.method != 'POST':
            return JsonResponse({'error': 'Gunakan metode POST'}, status=400)

        if 'image' not in request.FILES:
            return JsonResponse({'error': 'File tidak ditemukan'}, status=400)

        # Baca gambar dari request
        file = request.FILES['image']
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image_rgb is None:
            return JsonResponse({'error': 'Gagal membaca gambar. Pastikan format file benar.'}, status=400)
        print(f"Gambar berhasil dibaca. Shape Asli: {image_rgb.shape}")

        # --- MODIFIKASI ALUR PREPROCESSING ---
        
        # 1. RESIZE DULU (ke 224x224)
        # Ini adalah langkah preprocessing yang harus sama dengan saat pelatihan
        image_resized_224 = cv2.resize(image_rgb, (224, 224))
        print(f"Gambar di-resize ke: {image_resized_224.shape}")

        # 2. PERSIAPAN GRAYSCALE (Menggunakan Metode LAB)
        # Gunakan metode yang sama dengan pelatihan Anda (Lab)
        image_lab = cv2.cvtColor(image_resized_224, cv2.COLOR_BGR2Lab)
        image_gray_for_glcm = image_lab[:, :, 0] # Ambil saluran L (Luminosity)

        # 3. EKSTRAKSI FITUR
        # Ekstrak GLCM dari gambar grayscale 224x224
        glcm_feat = extract_glcm_features(image_gray_for_glcm)
        print(f"Fitur GLCM diekstraksi. Jumlah fitur GLCM: {len(glcm_feat)}")

        # Ekstrak VGG19 dari gambar RGB 224x224
        vgg_feat = extract_vgg19_features(image_resized_224)
        print(f"Fitur VGG19 diekstraksi. Jumlah fitur VGG19: {len(vgg_feat)}")

        # --- AKHIR MODIFIKASI ---

        # 4. Penggabungan fitur
        combined_feat = np.concatenate((glcm_feat, vgg_feat)).reshape(1, -1)
        print(f"Fitur digabungkan. Total fitur gabungan: {combined_feat.shape[1]}")

        # 5. Scaling
        scaled_feat = scaler.transform(combined_feat)
        print(f"Fitur telah di-scale.")

        # 6. Prediksi
        hasil = model.predict(scaled_feat)
        print(f"Prediksi model: {hasil[0]}") 

        return JsonResponse({'prediksi': int(hasil[0])})

    except Exception as e:
        print(f"Terjadi error dalam fungsi predict: {e}")
        return JsonResponse({'error': str(e)}, status=500)