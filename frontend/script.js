
// ==========================================
// RETINOVA - JAVASCRIPT
// AI RETINAL SCREENING SYSTEM
// ==========================================


// ==========================================
// 1. MENGAMBIL ELEMEN HTML
// ==========================================

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");

const analyzeButton = document.getElementById("analyzeButton");
const resetButton = document.getElementById("resetButton");

const predictionElement = document.getElementById("prediction");

const triagePriority = document.getElementById("triagePriority");
const triageDescription = document.getElementById("triageDescription");
const triageContent = document.querySelector(".triage-content");

const originalImage = document.getElementById("originalImage");
const gradcamImage = document.getElementById("gradcamImage");


// ==========================================
// 2. KONFIGURASI
// ==========================================

const API_URL = "http://127.0.0.1:5000/predict";

let selectedImage = null;


// ==========================================
// 3. PREVIEW GAMBAR RETINA
// ==========================================

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) {
        return;
    }

    // Validasi format gambar
    if (!["image/jpeg", "image/png"].includes(file.type)) {

        alert("Format gambar harus JPG, JPEG, atau PNG.");

        imageInput.value = "";
        selectedImage = null;

        imagePreview.innerHTML = "";

        resetButton.disabled = true;

        return;
    }

    // Menyimpan gambar
    selectedImage = file;

    // Menampilkan preview
    const reader = new FileReader();

    reader.onload = function (event) {

        imagePreview.innerHTML = `
            <img
                src="${event.target.result}"
                alt="Preview gambar retina"
            >
        `;

    };

    reader.readAsDataURL(file);

    // Mengaktifkan tombol reset
    resetButton.disabled = false;

});


// ==========================================
// 4. KATEGORI TRIAGE PROTOTIPE
// ==========================================

function tampilkanTriage(kelas) {

    const kategori = {

        "No DR": {

            priority: "Pemantauan",

            description:
                "Model tidak mendeteksi tanda retinopati diabetik pada gambar. Hasil tetap perlu dikonfirmasi melalui pemeriksaan profesional.",

            css: "routine"

        },

        "Mild NPDR": {

            priority: "Evaluasi Terjadwal",

            description:
                "Model mengidentifikasi kategori Mild NPDR. Pertimbangkan evaluasi lebih lanjut oleh tenaga medis profesional.",

            css: "scheduled"

        },

        "Moderate NPDR": {

            priority: "Evaluasi Prioritas",

            description:
                "Model mengidentifikasi kategori Moderate NPDR. Diperlukan evaluasi lebih lanjut oleh tenaga medis profesional.",

            css: "high"

        },

        "Severe NPDR": {

            priority: "Evaluasi Segera",

            description:
                "Model mengidentifikasi kategori Severe NPDR. Hasil ini perlu segera ditindaklanjuti melalui pemeriksaan tenaga medis profesional.",

            css: "urgent"

        },

        "Proliferative DR": {

            priority: "Evaluasi Segera",

            description:
                "Model mengidentifikasi kategori Proliferative DR. Hasil ini perlu segera ditindaklanjuti melalui pemeriksaan tenaga medis profesional.",

            css: "urgent"

        }

    };

    const hasil = kategori[kelas];

    // Jika kategori tidak ditemukan
    if (!hasil) {

        triagePriority.textContent = "Tidak tersedia";

        triageDescription.textContent =
            "Kategori hasil tidak dikenali.";

        return;

    }

    // Menampilkan prioritas
    triagePriority.textContent = hasil.priority;

    triageDescription.textContent = hasil.description;

    // Mengatur warna triage
    triageContent.classList.remove(
        "routine",
        "scheduled",
        "high",
        "urgent"
    );

    triageContent.classList.add(hasil.css);

}


// ==========================================
// 5. ANALISIS GAMBAR DENGAN AI
// ==========================================

analyzeButton.addEventListener("click", async function () {

    // Memeriksa gambar
    if (!selectedImage) {

        alert("Silakan pilih gambar retina terlebih dahulu.");

        return;

    }

    // Mengubah status tombol
    analyzeButton.disabled = true;

    analyzeButton.textContent = "Analyzing...";

    resetButton.disabled = true;

    predictionElement.textContent =
        "Sedang menganalisis gambar...";

    try {

        // Menyiapkan data gambar
        const formData = new FormData();

        formData.append("file", selectedImage);


        // Mengirim gambar ke backend Flask
        const response = await fetch(API_URL, {

            method: "POST",

            body: formData

        });


        // Membaca respons backend
        const result = await response.json();


        // Memeriksa respons
        if (!response.ok) {

            throw new Error(
                result.message || "Analisis gambar gagal."
            );

        }


        // ==================================
        // 5A. MENAMPILKAN HASIL PREDIKSI
        // ==================================

        predictionElement.textContent =
            result.prediction;


        // ==================================
        // 5B. MENAMPILKAN TRIAGE
        // ==================================

        tampilkanTriage(result.prediction);


        // ==================================
        // 5C. MENAMPILKAN GAMBAR ASLI
        // ==================================

        const reader = new FileReader();

        reader.onload = function (event) {

            originalImage.src = event.target.result;

            originalImage.style.display = "block";

        };

        reader.readAsDataURL(selectedImage);


        // ==================================
        // 5D. MENAMPILKAN GRAD-CAM
        // ==================================

        if (result.gradcam) {

            gradcamImage.src =
                "data:image/png;base64," + result.gradcam;

            gradcamImage.style.display = "block";

        } else {

            gradcamImage.removeAttribute("src");

            gradcamImage.style.display = "none";

            console.warn("Grad-CAM tidak tersedia.");

        }


        // Menampilkan informasi di console
        console.log("Prediksi:", result.prediction);

        console.log("Grad-CAM tersedia:", !!result.gradcam);


    } catch (error) {

        // ==================================
        // 5E. PENANGANAN ERROR
        // ==================================

        console.error("Error:", error);

        predictionElement.textContent =
            "Analisis gagal.";

        alert(
            "Terjadi kesalahan saat menganalisis gambar: " +
            error.message
        );

    } finally {

        // Mengembalikan tombol
        analyzeButton.disabled = false;

        analyzeButton.textContent = "Analisis Gambar";

        resetButton.disabled = !selectedImage;

    }

});


// ==========================================
// 6. RESET / ANALISIS ULANG
// ==========================================

resetButton.addEventListener("click", function () {

    // Menghapus gambar terpilih
    selectedImage = null;

    imageInput.value = "";


    // Menghapus preview gambar
    imagePreview.innerHTML = "";


    // Mengembalikan hasil prediksi
    predictionElement.textContent =
        "Belum ada gambar yang dianalisis.";


    // Mengembalikan informasi triage
    triagePriority.textContent =
        "Belum tersedia";

    triageDescription.textContent =
        "Prioritas tindak lanjut akan ditampilkan setelah gambar retina selesai dianalisis.";


    // Menghapus warna triage
    triageContent.classList.remove(
        "routine",
        "scheduled",
        "high",
        "urgent"
    );


    // Menghapus gambar asli
    originalImage.removeAttribute("src");

    originalImage.style.display = "none";


    // Menghapus visualisasi Grad-CAM
    gradcamImage.removeAttribute("src");

    gradcamImage.style.display = "none";


    // Mengembalikan tombol analisis
    analyzeButton.disabled = false;

    analyzeButton.textContent = "Analisis Gambar";


    // Menonaktifkan tombol reset
    resetButton.disabled = true;


    console.log("RETINOVA berhasil di-reset.");

});


// ==========================================
// END OF RETINOVA JAVASCRIPT
// ==========================================
