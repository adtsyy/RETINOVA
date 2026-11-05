
import tensorflow as tf
import numpy as np

# Layer pengganti Lambda yang menerima parameter mask
class IdentityWithMask(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        return tf.identity(inputs)

    def get_config(self):
        config = super().get_config()
        return config


print("Memuat model asli RETINOVA...")

model = tf.keras.models.load_model(
    "best_retinova_glf.keras",
    compile=False,
    safe_mode=False
)

# Mengganti Lambda dengan IdentityWithMask
def replace_lambda(layer):

    if isinstance(layer, tf.keras.layers.Lambda):

        print("Mengganti Lambda:", layer.name)

        return IdentityWithMask(
            name=layer.name,
            trainable=layer.trainable,
            dtype=layer.dtype_policy
        )

    return layer.__class__.from_config(
        layer.get_config()
    )


print("Membuat salinan model...")

new_model = tf.keras.models.clone_model(
    model,
    clone_function=replace_lambda,
    recursive=True
)

# Menyalin bobot
new_model.set_weights(model.get_weights())

print("Menguji prediksi model...")

dummy_global = np.random.rand(
    1, 224, 224, 3
).astype("float32")

dummy_local = np.random.rand(
    1, 224, 224, 3
).astype("float32")

pred_original = model.predict(
    [dummy_global, dummy_local],
    verbose=0
)

pred_new = new_model.predict(
    [dummy_global, dummy_local],
    verbose=0
)

difference = np.max(
    np.abs(pred_original - pred_new)
)

print("Selisih maksimum:", difference)

if difference < 1e-5:

    new_model.save("retinova_tfjs_ready.keras")

    print("BERHASIL! Model siap untuk tahap ekspor.")

else:

    print("PERINGATAN: Prediksi belum identik.")

print("Mengekspor model ke TensorFlow SavedModel...")

new_model.export("retinova_savedmodel")

print("BERHASIL! SavedModel berhasil dibuat.")