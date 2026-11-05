
// ==========================================
// RETINOVA - AI RETINAL SCREENING SYSTEM
// TAB: PEMERIKSAAN, RIWAYAT, PRIORITAS
// ==========================================


// ==========================================
// 1. ELEMEN HTML
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
// 2. KONFIGURASI
// ==========================================

const API_URL = "http://127.0.0.1:5000/predict";

const STORAGE_KEY = "retinova_examination_history";

let selectedImage = null;


// ==========================================
// 3. NAVIGASI TAB
// ==========================================

const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");

tabButtons.forEach(button => {

    button.addEventListener("click", function () {

        const targetTab = this.dataset.tab;

        // Menghapus status aktif semua tombol
        tabButtons.forEach(item => {
            item.classList.remove("active");
        });

        // Menonaktifkan semua panel
        tabPanels.forEach(panel => {
            panel.classList.remove("active");
            panel.hidden = true;
        });

        // Mengaktifkan tombol yang dipilih
        this.classList.add("active");

        // Menampilkan panel yang dipilih
        const selectedPanel = document.getElementById(targetTab);

        if (selectedPanel) {
            selectedPanel.hidden = false;
            selectedPanel.classList.add("active");
        }

        // Perbarui dashboard saat membuka tab
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
// 4. MENGAMBIL RIWAYAT
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
// 5. MENYIMPAN RIWAYAT
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
// 6. FORMAT TANGGAL
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
        return selisihHari + " hari yang lalu · " + formatTanggal(tanggal);
    }

    return "Tanggal mendatang · " + formatTanggal(tanggal);

}


// ==========================================
// 7. PREVIEW GAMBAR
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

        // Tampilkan preview
        imagePreview.style.display = "block";

    };

    reader.readAsDataURL(file);

    resetButton.disabled = false;

});



// ==========================================
// 8. TRIAGE PROTOTIPE
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
// 9. MENYIMPAN HASIL PEMERIKSAAN
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
// 10. KATEGORI PRIORITAS
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
// 11. MENGAMBIL HASIL TERBARU SETIAP PASIEN
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
// 12. MENAMPILKAN DASHBOARD PRIORITAS
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


    // Menghitung statistik prioritas
    const jumlahTinggi = pasien.filter(item =>
        ["Proliferative DR", "Severe NPDR"].includes(item.hasil)
    ).length;

    const jumlahRutin = pasien.filter(item =>
        item.hasil === "No DR"
    ).length;

    priorityTotalPatients.textContent = pasien.length;

    priorityHighPatients.textContent = jumlahTinggi;

    priorityRoutinePatients.textContent = jumlahRutin;


    // Menghapus daftar sebelumnya
    priorityPatientList.innerHTML = "";


    if (pasien.length === 0) {

        const kosong = document.createElement("p");

        kosong.className = "empty-priority";

        kosong.textContent =
            "Belum ada data pemeriksaan pasien.";

        priorityPatientList.appendChild(kosong);

        return;

    }


    // Membuat kartu setiap pasien
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

        tanggal.textContent =
            formatHariPemeriksaan(item.tanggal);


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
// 13. MENAMPILKAN RIWAYAT PEMERIKSAAN
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
// 14. STATISTIK RIWAYAT
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
// 15. MENGHAPUS SATU RIWAYAT
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
// 16. MENGHAPUS SEMUA RIWAYAT
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
// 17. PENCARIAN PASIEN
// ==========================================

searchPatient.addEventListener("input", function () {

    tampilkanRiwayat();

});


// ==========================================
// 18. ANALISIS GAMBAR DENGAN AI
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
        "Sedang menganalisis gambar...";


    try {

        const formData = new FormData();

        formData.append("file", selectedImage);


        const response = await fetch(API_URL, {

            method: "POST",

            body: formData

        });


        const result = await response.json();


        if (!response.ok) {

            throw new Error(
                result.message || "Analisis gambar gagal."
            );

        }


        if (!result.prediction) {

            throw new Error("Hasil prediksi tidak tersedia.");

        }


        // Menampilkan hasil prediksi
        predictionElement.textContent = result.prediction;


        // Menampilkan triage
        tampilkanTriage(result.prediction);


        // Menampilkan gambar retina asli
        const reader = new FileReader();

        reader.onload = function (event) {

            originalImage.src = event.target.result;

            originalImage.style.display = "block";

        };

        reader.readAsDataURL(selectedImage);


        // Menampilkan Grad-CAM
        if (result.gradcam) {

            gradcamImage.src =
                "data:image/png;base64," + result.gradcam;

            gradcamImage.style.display = "block";

        } else {

            gradcamImage.removeAttribute("src");

            gradcamImage.style.display = "none";

        }


        // Menyimpan riwayat
        const berhasilDisimpan = tambahRiwayat(
            namaPasien,
            result.prediction
        );


        if (berhasilDisimpan) {

            tampilkanRiwayat();

            perbaruiStatistik();

            tampilkanDashboardPrioritas();

            alert(
                "Pemeriksaan berhasil dan riwayat berhasil disimpan."
            );

        } else {

            alert(
                "Analisis berhasil, tetapi riwayat gagal disimpan."
            );

        }


        console.log("Nama pasien:", namaPasien);

        console.log("Hasil screening:", result.prediction);


    } catch (error) {

        console.error("Error:", error);

        predictionElement.textContent = "Analisis gagal.";

        alert(
            "Terjadi kesalahan saat menganalisis gambar: " +
            error.message
        );

    } finally {

        analyzeButton.disabled = false;

        analyzeButton.textContent = "Analisis Gambar";

        resetButton.disabled = !selectedImage;

    }

});


// ==========================================
// 19. RESET / ANALISIS ULANG
// ==========================================


resetButton.addEventListener("click", function () {

    // ==================================
    // 1. RESET IDENTITAS PASIEN
    // ==================================

    patientNameInput.value = "";


    // ==================================
    // 2. RESET FILE GAMBAR
    // ==================================

    selectedImage = null;

    imageInput.value = "";


    // ==================================
    // 3. HAPUS PREVIEW GAMBAR
    // ==================================

    imagePreview.innerHTML = "";
    imagePreview.style.display = "none";


    // ==================================
    // 4. RESET HASIL PREDIKSI
    // ==================================

    predictionElement.textContent =
        "Belum ada gambar yang dianalisis.";


    // ==================================
    // 5. RESET TRIAGE
    // ==================================

    triagePriority.textContent = "Belum tersedia";

    triageDescription.textContent =
        "Prioritas tindak lanjut akan ditampilkan setelah gambar retina selesai dianalisis.";

    triageContent.classList.remove(
        "routine",
        "scheduled",
        "high",
        "urgent"
    );


    // ==================================
    // 6. HAPUS GAMBAR RETINA ASLI
    // ==================================

    originalImage.removeAttribute("src");
    originalImage.style.display = "none";


    // ==================================
    // 7. HAPUS VISUALISASI GRAD-CAM
    // ==================================

    gradcamImage.removeAttribute("src");
    gradcamImage.style.display = "none";


    // ==================================
    // 8. RESET TOMBOL
    // ==================================

    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analisis Gambar";

    resetButton.disabled = true;

});
