# Klasifikasi Aksara Jawa dengan ResNet18

Proyek ini adalah aplikasi berbasis Streamlit yang menggunakan model ResNet18 untuk mendeteksi tulisan tangan dan mengklasifikasikan aksara Jawa dalam sebuah gambar. Sistem ini mampu melakukan segmentasi karakter dan memberikan prediksi beserta probabilitasnya.

<a href="https://ibb.co.com/DgWxZmXG"><img src="https://i.ibb.co.com/rf0CVqPb/Screenshot-2025-02-07-144740.png" alt="Screenshot-2025-02-07-144740" border="0"></a>

## 📌 Fitur Utama
- **Unggah Gambar**: Pengguna dapat mengunggah gambar yang mengandung aksara Jawa.
- **Segmentasi Karakter**: Sistem secara otomatis mendeteksi dan mengekstrak aksara individu dari gambar.
- **Klasifikasi dengan ResNet18**: Model deep learning akan memprediksi setiap karakter yang terdeteksi.
- **Visualisasi Hasil**: Hasil klasifikasi ditampilkan dengan bounding box dan probabilitas prediksi.
- **Grafik Prediksi**: Menampilkan 5 kemungkinan terbesar untuk setiap karakter yang terdeteksi.

## 🛠 Teknologi yang Digunakan
- **Python** (Streamlit, OpenCV, PyTorch, PIL, Matplotlib, NumPy, Torchvision)
- **Model Deep Learning**: ResNet18

## 📂 Struktur Direktori
```
├── Datasets/
├── model/
│   └── hancaraka_Model.pth  # Model yang telah dilatih
├── app.py  # Kode utama aplikasi Streamlit
├── README.md  # Dokumentasi proyek
```

## 📜 Dataset
Dataset yang digunakan untuk melatih model dapat diunduh melalui Kaggle:
[https://www.kaggle.com/datasets/vzrenggamani/hanacaraka](https://www.kaggle.com/datasets/vzrenggamani/hanacaraka)

## 🚀 Cara Menjalankan Aplikasi
1. **Clone repositori ini**
   ```sh
   git clone https://github.com/your-repo/aksara-jawa-classifier.git
   cd aksara-jawa-classifier
   ```

2. **Install dependensi**
   ```sh
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi Streamlit**
   ```sh
   streamlit run app.py
   ```

4. **Buka di browser**
   Akses aplikasi di `http://localhost:8501`

## 🔍 Catatan Penggunaan
- Pastikan gambar memiliki latar belakang kontras dengan aksara.
- Setiap aksara sebaiknya tidak saling menempel untuk hasil yang optimal.
- Kualitas gambar yang lebih tinggi akan meningkatkan akurasi prediksi.

## 🤝 Kontribusi
Model saat ini memiliki akurasi sebesar 87%. Jika Anda ingin meningkatkan performa, silakan sesuaikan atau lakukan optimasi lebih lanjut.
Kontribusi sangat diterima! Silakan fork repositori ini dan kirimkan pull request dengan perbaikan atau fitur tambahan.


