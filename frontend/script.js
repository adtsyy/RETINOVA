
// ==========================================
// 1. MEMUAT MODEL AI RETINOVA
// ==========================================

let retinovaModel = null;
let retinovaFeatureModel = null;

let retinovaModelPromise = null;
let retinovaFeaturePromise = null;

async function loadRetinovaModel() {
    try {
        await tf.setBackend("cpu");
        await tf.ready();

        console.log("Backend TensorFlow.js:", tf.getBackend());
        console.log("Memuat model RETINOVA...");

        retinovaModel = await tf.loadGraphModel(
            "./retinova_tfjs/model.json"
        );

        console.log("Model RETINOVA berhasil dimuat!");
        console.log("Output model:", retinovaModel.outputs);

        return retinovaModel;

    } catch (error) {
        retinovaModel = null;
        console.error("Gagal memuat model RETINOVA:", error);
        throw error;
    }
}

async function loadRetinovaFeatureModel() {
    try {
        if (retinovaFeatureModel) {
            return retinovaFeatureModel;
        }

        if (retinovaFeaturePromise) {
            return retinovaFeaturePromise;
        }

        retinovaFeaturePromise = (async () => {
            console.log("Memuat feature extractor...");

            retinovaFeatureModel = await tf.loadGraphModel(
                "./retinova_feature_tfjs/model.json"
            );

            console.log("Feature extractor berhasil dimuat!");

            return retinovaFeatureModel;
        })();

        return await retinovaFeaturePromise;

    } catch (error) {
        retinovaFeatureModel = null;
        retinovaFeaturePromise = null;

        console.error("Gagal memuat feature extractor:", error);
        throw error;
    }
}

// Memulai pemuatan model utama
retinovaModelPromise = loadRetinovaModel();


// ==========================================
// 2. MENGAMBIL ELEMEN HTML
// ==========================================

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");

const analyzeButton = document.getElementById("analyzeButton");
const resetButton = document.getElementById("resetButton");

const patientNameInput = document.getElementById("patientName");

const predictionElement = document.getElementById("prediction");

const triagePriority = document.getElementById("triagePriority");
const triageDescription = document.getElementById("triageDescription");
const triageContent = document.querySelector(".triage-content");

const originalImage = document.getElementById("originalImage");
const gradcamImage = document.getElementById("gradcamImage");

const historyBody = document.getElementById("historyBody");
const searchPatient = document.getElementById("searchPatient");

const totalExaminations = document.getElementById("totalExaminations");
const totalPatients = document.getElementById("totalPatients");

const clearHistoryButton = document.getElementById("clearHistoryButton");

const priorityPatientList =
    document.getElementById("priorityPatientList");

const priorityTotalPatients =
    document.getElementById("priorityTotalPatients");

const priorityHighPatients =
    document.getElementById("priorityHighPatients");

const priorityRoutinePatients =
    document.getElementById("priorityRoutinePatients");


// ==========================================
// 3. KONFIGURASI
// ==========================================

const STORAGE_KEY = "retinova_examination_history";

let selectedImage = null;

const NAMA_KELAS = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR"
];


// ==========================================
// 4. PREPROCESSING GAMBAR RETINOVA
// ==========================================

async function preprocessingRetinova(file) {

    const image = await createImageBitmap(file);

    try {
        return tf.tidy(() => {

            const canvas = document.createElement("canvas");
            canvas.width = 224;
            canvas.height = 224;

            const ctx = canvas.getContext("2d");

            // Padding hitam
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, 224, 224);

            // Mempertahankan rasio gambar
            const scale = Math.min(
                224 / image.width,
                224 / image.height
            );

            const newWidth = image.width * scale;
            const newHeight = image.height * scale;

            const offsetX = (224 - newWidth) / 2;
            const offsetY = (224 - newHeight) / 2;

            ctx.drawImage(
                image,
                offsetX,
                offsetY,
                newWidth,
                newHeight
            );

            // Tensor global
            const globalImage = tf.browser.fromPixels(canvas);

            // Crop tengah 160 x 160
            const localCrop = tf.slice(
                globalImage,
                [32, 32, 0],
                [160, 160, 3]
            );

            // Resize lokal
            const localImage = tf.image.resizeBilinear(
                localCrop,
                [224, 224],
                false
            );

            // Normalisasi MobileNetV2
            const globalInput = globalImage
                .toFloat()
                .div(127.5)
                .sub(1)
                .expandDims(0);

            const localInput = localImage
                .toFloat()
                .div(127.5)
                .sub(1)
                .expandDims(0);

            return {
                global: globalInput,
                local: localInput
            };
        });

    } finally {
        image.close();
    }
}


// ==========================================
// 5. GENERATE ACTIVATION HEATMAP
// ==========================================

async function generateRetinovaActivationHeatmap(
    inputs,
    originalFile
) {

    let features = null;
    let resizedHeatmap = null;
    let bitmap = null;

    try {
        console.time("Waktu Activation Heatmap");

        // Memuat feature extractor
        await loadRetinovaFeatureModel();

        // Mengambil feature map dari kedua cabang
        features = retinovaFeatureModel.predict([
            inputs.global,
            inputs.local
        ]);

        // Menggunakan feature map global
        const globalFeature = features[0];

        // Membuat dan menormalisasi heatmap
        resizedHeatmap = tf.tidy(() => {

            const heatmap = tf.relu(globalFeature)
                .mean(-1)
                .squeeze();

            const min = heatmap.min();
            const max = heatmap.max();

            const normalized = heatmap
                .sub(min)
                .div(max.sub(min).add(1e-8));

            return tf.image.resizeBilinear(
                normalized.expandDims(-1),
                [224, 224]
            ).squeeze();
        });

        const values = await resizedHeatmap.data();

        // Membuat canvas heatmap
        const canvas = document.createElement("canvas");
        canvas.width = 224;
        canvas.height = 224;

        const ctx = canvas.getContext("2d");
        const imageData = ctx.createImageData(224, 224);

        // Pewarnaan JET
        for (let i = 0; i < values.length; i++) {

            const v = Math.max(0, Math.min(1, values[i]));

            const r = Math.max(
                0,
                Math.min(1, 1.5 - Math.abs(4 * v - 3))
            );

            const g = Math.max(
                0,
                Math.min(1, 1.5 - Math.abs(4 * v - 2))
            );

            const b = Math.max(
                0,
                Math.min(1, 1.5 - Math.abs(4 * v - 1))
            );

            const index = i * 4;

            imageData.data[index] = r * 255;
            imageData.data[index + 1] = g * 255;
            imageData.data[index + 2] = b * 255;

            imageData.data[index + 3] =
                Math.round(Math.max(0, v - 0.15) * 180);
        }

        ctx.putImageData(imageData, 0, 0);

        // Membuat overlay dengan gambar asli
        bitmap = await createImageBitmap(originalFile);

        const overlay = document.createElement("canvas");
        overlay.width = 224;
        overlay.height = 224;

        const overlayCtx = overlay.getContext("2d");

        overlayCtx.drawImage(
            bitmap,
            0,
            0,
            224,
            224
        );

        overlayCtx.drawImage(canvas, 0, 0);

        // Menampilkan hasil
        gradcamImage.src = overlay.toDataURL("image/png");
        gradcamImage.style.display = "block";

        console.timeEnd("Waktu Activation Heatmap");

    } catch (error) {

        console.error(
            "Gagal membuat Activation Heatmap:",
            error
        );

        gradcamImage.style.display = "none";

    } finally {

        if (features) {
            features.forEach(tensor => tensor.dispose());
        }

        if (resizedHeatmap) {
            resizedHeatmap.dispose();
        }

        if (bitmap) {
            bitmap.close();
        }
    }
}


// ==========================================
// 6. NAVIGASI TAB
// ==========================================

const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

tabButtons.forEach(button => {

    button.addEventListener("click", function () {

        const targetTab = this.dataset.tab;

        tabButtons.forEach(item => {
            item.classList.remove("active");
        });

        tabPanels.forEach(panel => {
            panel.classList.remove("active");
            panel.hidden = true;
        });

        this.classList.add("active");

        const selectedPanel = document.getElementById(targetTab);

        if (selectedPanel) {
            selectedPanel.hidden = false;
            selectedPanel.classList.add("active");
        }

        if (targetTab === "riwayat") {
            tampilkanRiwayat();
            perbaruiStatistik();
        }

        if (targetTab === "prioritas") {
            tampilkanDashboardPrioritas();
        }
    });
});


// ==========================================
// 7. MENGAMBIL RIWAYAT
// ==========================================

function ambilRiwayat() {

    try {
        const data = localStorage.getItem(STORAGE_KEY);
        const riwayat = data ? JSON.parse(data) : [];

        return Array.isArray(riwayat) ? riwayat : [];

    } catch (error) {
        console.error("Gagal membaca riwayat:", error);
        return [];
    }
}


// ==========================================
// 8. MENYIMPAN RIWAYAT
// ==========================================

function simpanRiwayat(data) {

    try {
        localStorage.setItem(
            STORAGE_KEY,
            JSON.stringify(data)
        );

        return true;

    } catch (error) {
        console.error("Gagal menyimpan riwayat:", error);

        alert(
            "Riwayat gagal disimpan. Penyimpanan browser mungkin penuh."
        );

        return false;
    }
}


// ==========================================
// 9. FORMAT TANGGAL
// ==========================================

function formatTanggal(tanggal) {

    return new Date(tanggal).toLocaleString("id-ID", {
        day: "2-digit",
        month: "long",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit"
    });
}


function formatHariPemeriksaan(tanggal) {

    const waktu = new Date(tanggal);
    const sekarang = new Date();

    const hariIni = new Date(
        sekarang.getFullYear(),
        sekarang.getMonth(),
        sekarang.getDate()
    );

    const hariPemeriksaan = new Date(
        waktu.getFullYear(),
        waktu.getMonth(),
        waktu.getDate()
    );

    const selisihHari = Math.round(
        (hariIni - hariPemeriksaan) / 86400000
    );

    if (selisihHari === 0) {
        return "Hari ini · " + formatTanggal(tanggal);
    }

    if (selisihHari === 1) {
        return "Kemarin · " + formatTanggal(tanggal);
    }

    if (selisihHari > 1) {
        return selisihHari +
            " hari yang lalu · " +
            formatTanggal(tanggal);
    }

    return "Tanggal mendatang · " + formatTanggal(tanggal);
}


// ==========================================
// 10. PREVIEW GAMBAR
// ==========================================

imageInput.addEventListener("change", function () {

    const file = imageInput.files[0];

    if (!file) return;

    if (!["image/jpeg", "image/png"].includes(file.type)) {

        alert("Format gambar harus JPG, JPEG, atau PNG.");

        imageInput.value = "";
        selectedImage = null;

        imagePreview.innerHTML = "";
        imagePreview.style.display = "none";

        analyzeButton.disabled = true;
        resetButton.disabled = true;

        return;
    }

    selectedImage = file;

    const reader = new FileReader();

    reader.onload = function (event) {

        imagePreview.innerHTML = "";

        const image = document.createElement("img");

        image.src = event.target.result;
        image.alt = "Preview gambar retina";

        imagePreview.appendChild(image);

        imagePreview.style.display = "block";
    };

    reader.readAsDataURL(file);

    analyzeButton.disabled = false;
    resetButton.disabled = false;
});


// ==========================================
// 11. TRIAGE PROTOTIPE
// ==========================================

function tampilkanTriage(kelas) {

    const kategori = {

        "No DR": {
            priority: "Pemantauan Rutin",
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

    if (!hasil) {

        triagePriority.textContent = "Tidak tersedia";

        triageDescription.textContent =
            "Kategori hasil tidak dikenali.";

        return;
    }

    triagePriority.textContent = hasil.priority;
    triageDescription.textContent = hasil.description;

    triageContent.classList.remove(
        "routine",
        "scheduled",
        "high",
        "urgent"
    );

    triageContent.classList.add(hasil.css);
}


// ==========================================
// 12. MENYIMPAN HASIL PEMERIKSAAN
// ==========================================

function tambahRiwayat(namaPasien, hasilPrediksi) {

    const riwayat = ambilRiwayat();

    const pemeriksaanBaru = {

        id:
            Date.now().toString() +
            Math.random().toString(36).slice(2, 8),

        nama: namaPasien.trim(),

        tanggal: new Date().toISOString(),

        hasil: hasilPrediksi
    };

    riwayat.unshift(pemeriksaanBaru);

    return simpanRiwayat(riwayat);
}


// ==========================================
// 13. KATEGORI PRIORITAS
// ==========================================

const PRIORITAS = {

    "Proliferative DR": {
        level: 1,
        label: "Sangat Tinggi",
        css: "priority-very-high"
    },

    "Severe NPDR": {
        level: 2,
        label: "Tinggi",
        css: "priority-high"
    },

    "Moderate NPDR": {
        level: 3,
        label: "Sedang",
        css: "priority-medium"
    },

    "Mild NPDR": {
        level: 4,
        label: "Rendah",
        css: "priority-low"
    },

    "No DR": {
        level: 5,
        label: "Pemantauan Rutin",
        css: "priority-routine"
    }
};


// ==========================================
// 14. MENGAMBIL HASIL TERBARU SETIAP PASIEN
// ==========================================

function ambilPasienTerbaru() {

    const riwayat = ambilRiwayat();
    const pasienTerbaru = new Map();

    riwayat.forEach(item => {

        if (!item.nama || !item.tanggal || !item.hasil) {
            return;
        }

        const namaNormal = item.nama.trim().toLowerCase();
        const tanggalBaru = new Date(item.tanggal).getTime();

        if (Number.isNaN(tanggalBaru)) return;

        const dataLama = pasienTerbaru.get(namaNormal);

        if (
            !dataLama ||
            tanggalBaru > new Date(dataLama.tanggal).getTime()
        ) {
            pasienTerbaru.set(namaNormal, item);
        }
    });

    return Array.from(pasienTerbaru.values());
}


// ==========================================
// 15. DASHBOARD PRIORITAS
// ==========================================

function tampilkanDashboardPrioritas() {

    const pasien = ambilPasienTerbaru();

    pasien.sort((a, b) => {

        const prioritasA = PRIORITAS[a.hasil]?.level ?? 99;
        const prioritasB = PRIORITAS[b.hasil]?.level ?? 99;

        if (prioritasA !== prioritasB) {
            return prioritasA - prioritasB;
        }

        return new Date(b.tanggal) - new Date(a.tanggal);
    });

    const jumlahTinggi = pasien.filter(item =>
        ["Proliferative DR", "Severe NPDR"].includes(item.hasil)
    ).length;

    const jumlahRutin = pasien.filter(item =>
        item.hasil === "No DR"
    ).length;

    priorityTotalPatients.textContent = pasien.length;
    priorityHighPatients.textContent = jumlahTinggi;
    priorityRoutinePatients.textContent = jumlahRutin;

    priorityPatientList.innerHTML = "";

    if (pasien.length === 0) {

        const kosong = document.createElement("p");

        kosong.className = "empty-priority";
        kosong.textContent = "Belum ada data pemeriksaan pasien.";

        priorityPatientList.appendChild(kosong);

        return;
    }

    pasien.forEach(item => {

        const kategori = PRIORITAS[item.hasil];

        const card = document.createElement("div");
        card.className = "priority-patient-card";

        const info = document.createElement("div");
        info.className = "priority-patient-info";

        const nama = document.createElement("h3");
        nama.textContent = item.nama;

        const hasil = document.createElement("p");
        hasil.textContent = "Hasil terbaru: " + item.hasil;

        const tanggal = document.createElement("small");
        tanggal.textContent = formatHariPemeriksaan(item.tanggal);

        info.appendChild(nama);
        info.appendChild(hasil);
        info.appendChild(tanggal);

        const badge = document.createElement("span");

        badge.className =
            "priority-badge " +
            (kategori?.css || "priority-unknown");

        badge.textContent =
            kategori?.label || "Tidak diketahui";

        card.appendChild(info);
        card.appendChild(badge);

        priorityPatientList.appendChild(card);
    });
}


// ==========================================
// 16. RIWAYAT PEMERIKSAAN
// ==========================================

function tampilkanRiwayat(data = ambilRiwayat()) {

    const keyword = searchPatient.value.trim().toLowerCase();

    const hasilFilter = data.filter(item =>
        (item.nama || "").toLowerCase().includes(keyword)
    );

    historyBody.innerHTML = "";

    if (hasilFilter.length === 0) {

        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 5;

        cell.textContent = keyword
            ? "Pasien tidak ditemukan."
            : "Belum ada riwayat pemeriksaan.";

        row.appendChild(cell);
        historyBody.appendChild(row);

        return;
    }

    hasilFilter.forEach((item, index) => {

        const row = document.createElement("tr");

        const nomor = document.createElement("td");
        nomor.textContent = index + 1;

        const nama = document.createElement("td");
        nama.textContent = item.nama;

        const tanggal = document.createElement("td");
        tanggal.textContent = formatTanggal(item.tanggal);

        const hasil = document.createElement("td");
        hasil.textContent = item.hasil;

        const aksi = document.createElement("td");

        const deleteButton = document.createElement("button");

        deleteButton.type = "button";
        deleteButton.textContent = "Hapus";
        deleteButton.className = "history-delete-button";

        deleteButton.addEventListener("click", function () {
            hapusRiwayat(item.id);
        });

        aksi.appendChild(deleteButton);

        row.appendChild(nomor);
        row.appendChild(nama);
        row.appendChild(tanggal);
        row.appendChild(hasil);
        row.appendChild(aksi);

        historyBody.appendChild(row);
    });
}


// ==========================================
// 17. STATISTIK RIWAYAT
// ==========================================

function perbaruiStatistik() {

    const riwayat = ambilRiwayat();

    const pasienUnik = new Set(
        riwayat.map(item =>
            (item.nama || "").trim().toLowerCase()
        )
    );

    totalExaminations.textContent = riwayat.length;
    totalPatients.textContent = pasienUnik.size;
}


// ==========================================
// 18. MENGHAPUS SATU RIWAYAT
// ==========================================

function hapusRiwayat(id) {

    const konfirmasi = confirm(
        "Apakah kamu yakin ingin menghapus riwayat pemeriksaan ini?"
    );

    if (!konfirmasi) return;

    const riwayat = ambilRiwayat();
    const dataBaru = riwayat.filter(item => item.id !== id);

    if (simpanRiwayat(dataBaru)) {

        tampilkanRiwayat();
        perbaruiStatistik();
        tampilkanDashboardPrioritas();
    }
}


// ==========================================
// 19. MENGHAPUS SEMUA RIWAYAT
// ==========================================

clearHistoryButton.addEventListener("click", function () {

    const riwayat = ambilRiwayat();

    if (riwayat.length === 0) {
        alert("Belum ada riwayat pemeriksaan.");
        return;
    }

    const konfirmasi = confirm(
        "Apakah kamu yakin ingin menghapus seluruh riwayat pemeriksaan?"
    );

    if (!konfirmasi) return;

    if (simpanRiwayat([])) {

        tampilkanRiwayat();
        perbaruiStatistik();
        tampilkanDashboardPrioritas();

        alert("Seluruh riwayat berhasil dihapus.");
    }
});


// ==========================================
// 20. PENCARIAN PASIEN
// ==========================================

searchPatient.addEventListener("input", function () {
    tampilkanRiwayat();
});


// ==========================================
// 21. ANALISIS GAMBAR DENGAN AI
// ==========================================

analyzeButton.addEventListener("click", async function () {

    const namaPasien = patientNameInput.value.trim();

    if (!namaPasien) {

        alert("Silakan masukkan nama atau kode pasien.");
        patientNameInput.focus();

        return;
    }

    if (!selectedImage) {

        alert("Silakan pilih gambar retina terlebih dahulu.");
        return;
    }

    analyzeButton.disabled = true;
    resetButton.disabled = true;

    analyzeButton.textContent = "Analyzing...";

    predictionElement.textContent =
        "Sedang mempersiapkan model AI...";

    let inputs = null;
    let output = null;

    try {

        // Menunggu model utama selesai dimuat
        if (!retinovaModel) {

            predictionElement.textContent =
                "Menunggu model AI selesai dimuat...";

            await retinovaModelPromise;
        }

        if (!retinovaModel) {
            throw new Error("Model RETINOVA gagal dimuat.");
        }

        // Preprocessing gambar
        predictionElement.textContent =
            "Sedang memproses gambar...";

        inputs = await preprocessingRetinova(selectedImage);

        // Prediksi model utama
        predictionElement.textContent =
            "Sedang menganalisis gambar...";

        output = retinovaModel.predict([
            inputs.global,
            inputs.local
        ]);

        const prediksi = await output.data();

        // Menentukan kelas dengan nilai tertinggi
        let indeksPrediksi = 0;

        for (let i = 1; i < prediksi.length; i++) {

            if (prediksi[i] > prediksi[indeksPrediksi]) {
                indeksPrediksi = i;
            }
        }

        const hasilPrediksi = NAMA_KELAS[indeksPrediksi];

        if (!hasilPrediksi) {
            throw new Error("Kelas hasil prediksi tidak dikenali.");
        }

        // Menampilkan hasil prediksi
        predictionElement.textContent = hasilPrediksi;

        // Menampilkan triage prototipe
        tampilkanTriage(hasilPrediksi);

        // Menampilkan gambar retina asli
        const reader = new FileReader();

        reader.onload = function (event) {

            originalImage.src = event.target.result;
            originalImage.style.display = "block";
        };

        reader.readAsDataURL(selectedImage);

        // Membuat Activation Heatmap
        predictionElement.textContent =
            hasilPrediksi + " — Membuat Activation Heatmap...";

        await generateRetinovaActivationHeatmap(inputs, selectedImage );

        // Mengembalikan hasil prediksi
        predictionElement.textContent = hasilPrediksi;

        // Menyimpan riwayat
        const berhasilDisimpan = tambahRiwayat(
            namaPasien,
            hasilPrediksi
        );

        if (berhasilDisimpan) {

            tampilkanRiwayat();
            perbaruiStatistik();
            tampilkanDashboardPrioritas();

            alert("Pemeriksaan berhasil disimpan!");

        } else {

            alert("Hasil berhasil dianalisis, tetapi riwayat gagal disimpan.");
        }

        console.log("Nama pasien:", namaPasien);
        console.log("Hasil prediksi:", hasilPrediksi);

    } catch (error) {

        console.error("Terjadi kesalahan analisis:", error);

        predictionElement.textContent =
            "Analisis gagal. Silakan coba kembali.";

        alert(
            "Terjadi kesalahan saat menganalisis gambar. Periksa Console browser untuk detailnya."
        );

    } finally {

        // Membersihkan tensor
        if (inputs) {

            if (inputs.global) {
                inputs.global.dispose();
            }

            if (inputs.local) {
                inputs.local.dispose();
            }
        }

        if (output) {
            output.dispose();
        }

        analyzeButton.disabled = !selectedImage;
        analyzeButton.textContent = "Analisis Gambar";

        resetButton.disabled = !selectedImage;
    }
});


// ==========================================
// 22. RESET / ANALISIS ULANG
// ==========================================

resetButton.addEventListener("click", function () {

    // Reset identitas pasien
    patientNameInput.value = "";

    // Reset file gambar
    selectedImage = null;
    imageInput.value = "";

    // Hapus preview gambar
    imagePreview.innerHTML = "";
    imagePreview.style.display = "none";

    // Reset hasil prediksi
    predictionElement.textContent =
        "Belum ada gambar yang dianalisis.";

    // Reset triage
    triagePriority.textContent = "Belum tersedia";

    triageDescription.textContent =
        "Prioritas tindak lanjut akan ditampilkan setelah gambar retina selesai dianalisis.";

    triageContent.classList.remove(
        "routine",
        "scheduled",
        "high",
        "urgent"
    );

    // Hapus gambar retina asli
    originalImage.removeAttribute("src");
    originalImage.style.display = "none";

    // Hapus visualisasi heatmap
    gradcamImage.removeAttribute("src");
    gradcamImage.style.display = "none";

    // Reset tombol
    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analisis Gambar";

    resetButton.disabled = true;
});


// ==========================================
// 23. INISIALISASI TAMPILAN
// ==========================================

tampilkanRiwayat();
perbaruiStatistik();
tampilkanDashboardPrioritas();

console.log("RETINOVA berhasil diinisialisasi!");
