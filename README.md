# Laporan Interaktif Segmentasi Pelanggan Mall (K-Means)

Aplikasi web **Streamlit** yang menyajikan seluruh alur penelitian segmentasi pelanggan mall menggunakan **K-Means Clustering** — mulai dari latar belakang, rumusan masalah, dan tujuan, hingga hasil pembagian klaster dan rekomendasi pemasaran. Aplikasi ini berfungsi sebagai *laporan interaktif*: setiap rumusan masalah dan tujuan di halaman awal terhubung ke halaman/visualisasi jawabannya.

## Struktur Aplikasi (Navigasi Halaman)
1. **Halaman 1 - Latar Belakang, Rumusan Masalah, dan Tujuan** (`app.py`): konteks, RM & tujuan, tabel pemetaan RM → Tujuan → Halaman jawaban.
2. **Halaman 2 - Data dan Kualitas Data**: peran kolom, ukuran data, pratinjau, statistik deskriptif, cek missing/duplikat, peringatan kualitas.
3. **Halaman 3 - Gambaran Pelanggan** (persiapan RM1): histogram, bar chart gender/profesi, heatmap korelasi & alasan clustering multivariat.
4. **Halaman 4 - Menentukan Jumlah Klaster** (RM2/Tujuan 2): pemilihan fitur, StandardScaler, Elbow & Silhouette (k=2–10), pemilihan k final (default 5, `random_state` tetap) disimpan ke `st.session_state`.
5. **Halaman 5 - Hasil Pembagian Klaster** (RM1/Tujuan 1): PCA 2D, scatter Income vs Spending per klaster, jumlah/persentase, tabel berlabel klaster yang dapat difilter.
6. **Halaman 6 - Profil Segmen** (RM3/Tujuan 3): rata-rata atribut vs keseluruhan (heatmap & radar), komposisi gender/profesi, nama deskriptif otomatis segmen.
7. **Halaman 7 - Rekomendasi Pemasaran** (RM4/Tujuan 4): kartu rekomendasi berbasis aturan per segmen (termasuk paket keluarga & profesi dominan).
8. **Halaman 8 - Kesimpulan dan Unduh**: ringkasan temuan otomatis, jawaban singkat RM1–RM4, keterbatasan, unduh CSV hasil, prediksi segmen pelanggan baru.

## Struktur Repo
- `app.py` — Halaman 1 & pengaturan global/sidebar (sumber data, filter usia, state klaster).
- `pages/` — Halaman 2–8 (berawalan angka agar urut).
- `utils.py` — fungsi pemuatan/validasi data, prapemrosesan, kualitas data, klastering (K-Means, scaler, PCA, metrik), penamaan segmen, profil, aturan rekomendasi, prediksi, ringkasan & jawaban otomatis.
- `Customer_mall.csv` — data default.
- `requirements.txt` — dependensi dengan versi dipatok.
- `README.md` — dokumentasi.

## Data
- **File:** `Customer_mall.csv` (2.000 baris, 8 kolom), dimuat dari repo sebagai default. Mendukung upload CSV dengan struktur kolom sama.
- **Kolom wajib:** `CustomerID`, `Gender` (Male/Female), `Age`, `Annual_Income($)`, `Spending Score(1-100)`, `Profession`, `Work_Experience`, `Family_Size`.

### Penanganan & Catatan Kualitas Data
- **Profession** kosong (35 baris): diisi `"Unknown"` (baris tidak dihapus).
- **Age** berkisar 0–99 dengan sebaran sangat lebar dan nilai tidak lazim: tidak dihapus otomatis; ditampilkan peringatan kualitas data dan tersedia filter rentang usia opsional di sidebar.
- **Annual_Income($)** bernilai 0 pada 2 baris: ditampilkan sebagai peringatan.
- Korelasi antar variabel numerik mendekati nol dan **Silhouette Score** cenderung rendah (sekitar 0,15–0,17). Kualitas klaster dilaporkan secara jujur: segmen merupakan **pengelompokan praktis untuk pemasaran**, bukan kelompok alami yang terpisah tegas.
- Semua angka dan kesimpulan pada Halaman 5–8 dihitung dari data (bukan hardcode). Palet warna klaster konsisten di seluruh grafik.

## Prasyarat
- Python 3.10+
- Dependensi: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly` (lihat `requirements.txt`)

## Menjalankan Secara Lokal
1. (Opsional) Buat dan aktifkan virtual environment.
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
4. Buka URL lokal yang ditampilkan (biasanya `http://localhost:8501`) dan gunakan sidebar untuk navigasi halaman.

## Upload ke GitHub
1. Inisialisasi repo dan tambahkan file (jangan pernah meng-commit `.venv` atau `__pycache__`):
   ```bash
   git init
   git add .gitignore app.py utils.py requirements.txt README.md Customer_mall.csv pages .streamlit logo_unpak.png
   git commit -m "Laporan interaktif segmentasi pelanggan K-Means"
   ```
2. Buat repository di GitHub, lalu hubungkan dan push:
   ```bash
   git remote add origin <URL_REPO_GITHUB>
   git branch -M main
   git push -u origin main
   ```

## Deploy di Streamlit Community Cloud (share.streamlit.io)
1. Login ke [share.streamlit.io](https://share.streamlit.io) dengan akun GitHub.
2. Klik **New app** (atau **Create app**).
3. Pilih **Repository**, **Branch** (`master`), dan file utama **`app.py`**.
4. Buka **Advanced settings** → pilih **Python 3.12** (agar `scikit-learn==1.5.2` terpasang dari wheel).
5. Klik **Deploy** dan tunggu proses build hingga selesai.
6. Buka **Settings → Sharing** dan set **"This app is public and searchable"** / **"Anyone with the link can view"** agar pengunjung tidak terjebak redirect/login.

### Troubleshooting: app berputar/redirect ke login
- Pastikan sharing app sudah **public** (langkah 6 di atas).
- Coba buka di jendela **incognito** atau browser lain, dan izinkan cookie untuk `streamlit.app`.
- Cek **Manage app → Logs** jika build gagal; setelah mengubah dependensi, klik **Reboot** / **Clear cache**.
- Pastikan folder `.venv` tidak ada di repo GitHub (build Cloud memakai environment-nya sendiri).
