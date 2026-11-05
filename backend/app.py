
import os
import io
import base64
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# ==================================
# 1. MEMUAT MODEL RETINOVA
# ==================================


# ==================================
# 1. MEMUAT MODEL RETINOVA
# ==================================

HF_REPO_ID = "adtsyy/retinova-model"
HF_TOKEN = os.environ.get("HF_TOKEN")

LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_retinova_glf.keras"
)

if os.path.exists(LOCAL_MODEL_PATH):
    MODEL_PATH = LOCAL_MODEL_PATH
    print("Menggunakan model lokal RETINOVA...")

else:
    print("Mengunduh model dari Hugging Face...")

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN belum diatur pada environment."
        )

    from huggingface_hub import hf_hub_download

    MODEL_PATH = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="best_retinova_glf.keras",
        token=HF_TOKEN
    )

print("Memuat model RETINOVA...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

print("Model RETINOVA berhasil dimuat!")



# ==================================
# 2. MENYIAPKAN MODEL GRAD-CAM
# ==================================

backbone = model.get_layer("mobilenetv2_1.00_224")

feature_map_global = backbone._inbound_nodes[0].output_tensors[0]

model_gradcam = tf.keras.Model(
    inputs=model.inputs,
    outputs=[feature_map_global, model.output]
)

print("Model Grad-CAM berhasil disiapkan!")


# ==================================
# 3. FUNGSI PREPROCESSING
# ==================================

def preprocessing_gambar(image_array):

    gambar_global = tf.image.resize_with_pad(
        image_array, 224, 224
    )

    gambar_local = tf.image.crop_to_bounding_box(
        gambar_global, 32, 32, 160, 160
    )

    gambar_local = tf.image.resize(
        gambar_local, (224, 224)
    )

    gambar_global = tf.cast(gambar_global, tf.float32)
    gambar_local = tf.cast(gambar_local, tf.float32)

    gambar_global = tf.keras.applications.mobilenet_v2.preprocess_input(
        gambar_global
    )

    gambar_local = tf.keras.applications.mobilenet_v2.preprocess_input(
        gambar_local
    )

    gambar_global = tf.expand_dims(gambar_global, axis=0)
    gambar_local = tf.expand_dims(gambar_local, axis=0)

    return gambar_global, gambar_local


# ==================================
# 4. MEMBUAT GRAD-CAM
# ==================================

def membuat_gradcam(gambar_global, gambar_local, kelas):

    with tf.GradientTape() as tape:

        feature_maps, prediksi = model_gradcam(
            [gambar_global, gambar_local],
            training=False
        )

        skor_kelas = prediksi[:, kelas]

    gradients = tape.gradient(
        skor_kelas,
        feature_maps
    )

    bobot = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    feature_maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        feature_maps * bobot,
        axis=-1
    )

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )

    return heatmap.numpy()


# ==================================
# 5. MEMBUAT OVERLAY GRAD-CAM
# ==================================

def membuat_overlay(image_array, heatmap):

    # Mengubah heatmap menjadi ukuran gambar model
    heatmap_image = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_image = heatmap_image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    heatmap_array = np.array(
        heatmap_image
    ).astype(np.float32) / 255.0

    # Membuat colormap sederhana
    r = np.clip(1.5 - np.abs(4 * heatmap_array - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * heatmap_array - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * heatmap_array - 1), 0, 1)

    warna_heatmap = np.stack(
        [r, g, b],
        axis=-1
    )

    warna_heatmap = np.uint8(
        warna_heatmap * 255
    )

    # Mengubah gambar asli ke ukuran 224 x 224
    gambar_asli = tf.image.resize_with_pad(
        image_array, 224, 224
    )

    gambar_asli = np.uint8(
        tf.clip_by_value(gambar_asli, 0, 255).numpy()
    )

    # Menggabungkan gambar dan heatmap
    overlay = (
        0.6 * gambar_asli.astype(np.float32)
        + 0.4 * warna_heatmap.astype(np.float32)
    )

    overlay = np.uint8(
        np.clip(overlay, 0, 255)
    )

    return overlay


# ==================================
# 6. KONVERSI GAMBAR KE BASE64
# ==================================

def gambar_ke_base64(gambar_array):

    gambar = Image.fromarray(
        gambar_array.astype(np.uint8)
    )

    buffer = io.BytesIO()

    gambar.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


# ==================================
# 7. HALAMAN UTAMA API
# ==================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "RETINOVA API is running!"
    })


# ==================================
# 8. PREDIKSI DAN GRAD-CAM
# ==================================

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Gambar belum diunggah."
        }), 400

    file = request.files["file"]

    try:

        # Membaca gambar
        image = Image.open(file.stream).convert("RGB")
        image_array = np.array(image)

        # Preprocessing
        gambar_global, gambar_local = preprocessing_gambar(
            image_array
        )

        # Prediksi
        hasil = model.predict(
            [gambar_global, gambar_local],
            verbose=0
        )

        kelas = int(np.argmax(hasil[0]))

        # Membuat Grad-CAM berdasarkan kelas prediksi
        heatmap = membuat_gradcam(
            gambar_global,
            gambar_local,
            kelas
        )

        # Membuat overlay
        overlay = membuat_overlay(
            image_array,
            heatmap
        )

        # Mengubah overlay menjadi Base64
        gradcam_base64 = gambar_ke_base64(
            overlay
        )

        return jsonify({
            "status": "success",
            "prediction": NAMA_KELAS[kelas],
            "gradcam": gradcam_base64
        })

    except Exception as e:

        print("Error:", str(e))

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ==================================
# 9. MENJALANKAN SERVER
# ==================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
