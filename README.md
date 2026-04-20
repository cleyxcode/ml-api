# 🌱 Siram Pintar API

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![MySQL](https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white)

**Siram Pintar API** adalah sistem *backend* cerdas berbasis **FastAPI** yang dirancang khusus untuk memanajemen sistem penyiraman tanaman IoT (Internet of Things). Backend ini menggunakan algoritma *Machine Learning* **K-Nearest Neighbors (KNN)** untuk mengklasifikasi kebutuhan air tanaman berdasarkan data sensor (Kelembapan Tanah, Suhu, dan Kelembapan Udara) secara *real-time*.

---

## ✨ Fitur Unggulan

### 🤖 1. Machine Learning (K-Nearest Neighbors)
API dilengkapi skrip `train_model.py` untuk melatih model berdasarkan dataset cuaca/tanah. Model KNN secara cerdas mendeteksi apakah kondisi tanah saat ini tergolong **Kering**, **Ideal**, atau **Basah**, disertai tingkat kepercayaan (*confidence level*).

### 🧠 2. Smart Auto-Mode & Logika Adaptif
Mode otomatis tidak hanya mengandalkan batas tetap, melainkan mesin pengambil keputusan tingkat lanjut:
*   **Weather-Adaptive Thresholds**: Batas penyiraman akan otomatis naik saat cuaca panas (agar tidak dehidrasi) dan turun saat cuaca dingin/lembab.
*   **Adaptive Cooldown**: Waktu jeda (*cooldown*) penyiraman yang dinamis. Jika pompa hanya menyala sebentar, waktu tunggu disingkat menjadi 15 menit (dari 45 menit normal).
*   **Deteksi Hujan Otomatis**: Mendeteksi pola hujan dari kelembapan udara yang tinggi dan lonjakan kelembapan tanah mendadak, lalu membatalkan/menunda penyiraman pompa.

### 🛡️ 3. Proteksi & Keamanan Tingkat Tinggi (Enterprise-Grade)
*   **Safety Lockout (Anti-Banjir)**: Mencegah pompa menyala lebih dari 10 kali sehari akibat error sensor.
*   **Sensor Anomaly Detection**: Mendeteksi nilai yang tidak masuk akal (suhu ekstrem atau lonjakan 30% dalam sedetik) dan mengabaikan data tersebut (filter *noise*).
*   **Anti-Spam & Debounce**: Melindungi database MySQL dari *spam ping* jika ESP32 mengalami *bug looping*.
*   **Night Emergency Micro-Watering**: Mengizinkan penyiraman darurat di luar jam normal (malam hari) jika sangat kritis, namun durasinya dipangkas hanya 60 detik untuk mencegah jamur.

### 🧹 4. Auto-Prune Database
Berjalan secara *background* membuang log data sensor historis yang usianya sudah melewati 14 hari, agar *database* dan *endpoint history* tetap ringan serta tidak memakan biaya server berlebih.

---

## 🚀 Instalasi & Persiapan

1. **Clone Repository**
   ```bash
   git clone https://github.com/cleyxcode/ml-api.git
   cd ml-api
   ```

2. **Buat Virtual Environment (Opsional namun disarankan)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Konfigurasi Environment (`.env`)**
   Buat file bernama `.env` di root direktori dan sesuaikan isi kredensial MySQL Anda:
   ```ini
   DB_HOST=srvXXX.hstgr.io
   DB_PORT=3306
   DB_USER=user_database_anda
   DB_PASS=password_database_anda
   DB_NAME=nama_database_anda
   API_KEY=KODE_RAHASIA_ANDA # Bebas, contoh: A7K2M
   ```

5. **Train Model ML (Wajib di awal)**
   Jika model `knn_model.pkl` belum ada, jalankan proses training satu kali:
   ```bash
   python train_model.py
   ```

---

## 🏃 Menjalankan Server API

Anda bisa menjalankan server secara lokal untuk pengembangan (*development*):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
Atau menggunakan *shell script* yang tersedia:
```bash
bash run.sh
```

Akses dokumentasi interaktif Swagger UI di browser:  
👉 **`http://localhost:8000/docs`**

---

## 📡 Daftar Endpoint Utama

*Semua endpoint kecuali `/` membutuhkan header autorisasi `X-API-Key: <KODE_RAHASIA_ANDA>`*

| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/` | Health check publik (tanpa API Key). |
| `POST` | `/sensor` | Endpoint utama ESP32 untuk mengirim data sensor & menerima aksi pompa. |
| `POST` | `/control` | Endpoint dari Aplikasi/Frontend untuk menyalakan/mematikan pompa manual atau ganti mode. |
| `GET` | `/status` | Mendapatkan status terkini sistem, pompa, mode, dan prediksi KNN terakhir. |
| `GET` | `/history` | Mengambil log riwayat penyiraman dan pembacaan sensor (dilengkapi paginasi/limit). |
| `GET` | `/config` | Menampilkan konfigurasi & *threshold* (batas) yang saat ini berjalan. |
| `POST` | `/reset-rain`| Me-reset paksa status sistem yang sedang mendeteksi/mem-blokir karena "hujan". |

---

## 🛠️ Deployment

Proyek ini telah dikonfigurasi untuk siap rilis ke berbagai *Platform as a Service* (PaaS):
*   **Vercel**: Melalui konfigurasi `vercel.json` (Serverless Function).
*   **Render**: Melalui spesifikasi `render.yaml`.
*   **Docker / VPS**: Melalui `Dockerfile` untuk di-_build_ menjadi *container* mandiri.
