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

model_path = os.path.join(settings.BASE_DIR, 'deteksi', 'data', 'contoh_model_monkeypox.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'deteksi', 'data', 'scaler_fix.pkl')

with open(model_path, 'rb') as f:
    model = joblib.load(f)

with open(scaler_path, 'rb') as f:
    scaler = joblib.load(f)

def homepage(request):
    return HttpResponse("Selamat datang di API Skripsi!")

def index(request):
    return HttpResponse("Halaman utama Django API Skripsi")

# Load VGG19 dan hilangkan layer klasifikasinya
base_model = VGG19(weights='imagenet', include_top=True)
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def extract_glcm_features(image_gray):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(image_gray, distances=distances, angles=angles, symmetric=True, normed=True)
    features = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
    for prop in props:
        vals = graycoprops(glcm, prop)[0]
        features.extend(vals)
    return features  # 16 fitur

def extract_vgg19_features(image_rgb):
    image_resized = cv2.resize(image_rgb, (224, 224))
    image_array = img_to_array(image_resized)
    image_preprocessed = preprocess_input(np.expand_dims(image_array, axis=0))
    features = vgg_model.predict(image_preprocessed)
    return features.flatten()  # 4096 fitur

@csrf_exempt
def predict(request):
    try:
        if request.method != 'POST':
            return JsonResponse({'error': 'Gunakan metode POST'}, status=400)

        if 'image' not in request.FILES:
            return JsonResponse({'error': 'File   tidak ditemukan'}, status=400)

        # Baca gambar dari request
        file = request.FILES['image']
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Tambahkan print untuk memastikan gambar terbaca
        if image_rgb is None:
            return JsonResponse({'error': 'Gagal membaca gambar. Pastikan format file benar.'}, status=400)
        print(f"Gambar berhasil dibaca. Shape: {image_rgb.shape}")

        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Ekstraksi fitur
        glcm_feat = extract_glcm_features(image_gray)
        print(f"Fitur GLCM diekstraksi. Jumlah fitur GLCM: {len(glcm_feat)}")
        print(f"Contoh fitur GLCM pertama: {glcm_feat[:5]}...") # Menampilkan 5 fitur pertama

        vgg_feat = extract_vgg19_features(image_rgb)
        print(f"Fitur VGG19 diekstraksi. Jumlah fitur VGG19: {len(vgg_feat)}")
        print(f"Contoh fitur VGG19 pertama: {vgg_feat[:5]}...") # Menampilkan 5 fitur pertama

        # Penggabungan fitur
        combined_feat = np.concatenate((glcm_feat, vgg_feat)).reshape(1, -1)
        print(f"Fitur digabungkan. Total fitur gabungan: {combined_feat.shape[1]}")

        # Scaling
        scaled_feat = scaler.transform(combined_feat)
        print(f"Fitur telah di-scale.")
        print(f"Contoh fitur ter-scale pertama: {scaled_feat[0][:5]}...") # Menampilkan 5 fitur pertama setelah scaling

        # Prediksi
        hasil = model.predict(scaled_feat)
        print(f"Prediksi model: {hasil[0]}") # Nilai mentah dari prediksi

        return JsonResponse({'prediksi': int(hasil[0])})

    except Exception as e:
        print(f"Terjadi error dalam fungsi predict: {e}")
        return JsonResponse({'error': str(e)}, status=500)