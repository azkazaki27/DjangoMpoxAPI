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
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Model
from keras.preprocessing.image import img_to_array
from skimage.feature import graycomatrix, graycoprops

# Muat model & scaler
model_path = os.path.join(settings.BASE_DIR, 'deteksi', 'data', 'Model_skenario_18.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'deteksi', 'data', 'scaler_skenario_18.pkl')

with open(model_path, 'rb') as f:
    model = joblib.load(f)

with open(scaler_path, 'rb') as f:
    scaler = joblib.load(f)

# Muat VGG19
base_model = VGG19(weights='imagenet', include_top=True)
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

def homepage(request):
    return HttpResponse("Selamat datang di API Skripsi!")

def index(request):
    return HttpResponse("Halaman utama Django API Skripsi")

# Fungsi validasi warna kulit
def is_skin_image(image_bgr, threshold_percent=20.0):
    """
    Cek apakah gambar didominasi warna kulit manusia.
    Menggunakan ruang warna HSV. Threshold 20% artinya minimal 20% piksel harus warna kulit.
    """
    try:
        # Konversi ke HSV
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Range warna kulit manusia (General)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Masking
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Hitung persentase
        total_pixels = mask.size
        skin_pixels = np.count_nonzero(mask)
        percentage = (skin_pixels / total_pixels) * 100

        print(f"[Skin Check] Persentase warna kulit: {percentage:.2f}%")
        
        # Return True jika di atas ambang batas
        return percentage >= threshold_percent
    except:
        return True # Fail-safe jika error, loloskan saja

# Fungsi ekstraksi fitur
def extract_glcm_features(image_gray):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    image_gray_normalized = image_gray.astype(np.uint8)
    glcm = graycomatrix(image_gray_normalized, distances=distances, angles=angles, symmetric=True, normed=True)
    features = []
    props = ['contrast', 'energy', 'homogeneity', 'correlation']
    for prop in props:
        vals = graycoprops(glcm, prop)[0]
        features.extend(vals)
    return features 

def extract_vgg19_features(image_rgb_resized):
    image_array = img_to_array(image_rgb_resized)
    image_preprocessed = preprocess_input(np.expand_dims(image_array, axis=0))
    features = vgg_model.predict(image_preprocessed, verbose=0)
    return features.flatten()

# Fungsi utama
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
            return JsonResponse({'error': 'Gagal membaca gambar.'}, status=400)
        
        print(f"Gambar berhasil dibaca. Shape Asli: {image_rgb.shape}")

        # Preprocessing resize ke 224x224
        image_resized_224 = cv2.resize(image_rgb, (224, 224))
        print(f"Gambar di-resize ke: {image_resized_224.shape}")

        # Deteksi persentasi kulit
        if not is_skin_image(image_resized_224):
            # Jika bukan kulit, langsung kembalikan respon penolakan
            print("Gambar ditolak: Bukan citra kulit.")
            return JsonResponse({
                'prediksi': -1, 
                'pesan': 'Tidak bisa mengenali kulit atau objek tidak dikenal'
            })
        
        # Lanjut ke proses utama jika lolos deteksi kulit
        # Preprocessing grayscaling
        image_lab = cv2.cvtColor(image_resized_224, cv2.COLOR_BGR2Lab)
        image_gray_for_glcm = image_lab[:, :, 0]

        # Ekstraksi Fitur GLCM
        glcm_feat = extract_glcm_features(image_gray_for_glcm)
        print(f"Fitur GLCM diekstraksi: {len(glcm_feat)}")

        # Ekstrak Fitur VGG19 
        vgg_feat = extract_vgg19_features(image_resized_224)
        print(f"Fitur VGG19 diekstraksi: {len(vgg_feat)}")

        # Penggabungan fitur
        combined_feat = np.concatenate((glcm_feat, vgg_feat)).reshape(1, -1)
        
        # Scaling
        scaled_feat = scaler.transform(combined_feat)
        print(f"Fitur telah di-scale.")

        # Prediksi RF
        hasil = model.predict(scaled_feat)
        prediksi_kelas = int(hasil[0])
        print(f"Prediksi model: {prediksi_kelas}") 

        # Mapping nama kelas untuk pesan (Opsional, agar lebih jelas di Android)
        class_names = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal']
        pesan_hasil = class_names[prediksi_kelas]

        return JsonResponse({
            'prediksi': prediksi_kelas,
            'pesan': pesan_hasil
        })

    except Exception as e:
        print(f"Terjadi error dalam fungsi predict: {e}")
        return JsonResponse({'error': str(e)}, status=500)