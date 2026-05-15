<div align="center">

# Convolutional and Recurrent Neural Networks

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.2-4DABCF?logo=numpy&logoColor=fff)
![SciPy](https://img.shields.io/badge/SciPy-1.17.1-8CA1E5?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557c?logo=python&logoColor=white)

</div>

## 📘 Deskripsi

Di sini, kami membuat package `nn` yang memuat semua keperluan pembangunan suatu _neural network_ secara lengkap. Package ini meliputi layer-layer dense, konvolusi, pooling, rekuren, embedding, dan lainnya. Kedua _forward_ dan _backward pass_ diimplementasikan di package ini juga! Terdapat banyak jenis kustomisasi lainnya seperti fungsi aktivasi, metode inisialisasi bobot, fungsi loss, dan metode optimisasi. Implementasi selengkapnya bisa dilihat di folder [`src/nn/`](./src/nn).

Selanjutnya, kami juga menerapkan package ini dalam menyelesaikan dua permasalahan. Pertama, kami membuat arsitektur CNN secara _custom_ untuk menyelesaikan masalah _image captioning_ dari dataset [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). Kedua, kami mereplikasikan arsitektur RNN berupa Encoder-Decoder "Show and Tell" oleh [Vinyals et al.](https://arxiv.org/abs/1411.4555) untuk menyelesaikan masalah _image captioning_ dari dataset [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k). Implementasi selengkapnya bisa dilihat di folder [`src/1_cnn_image_classification/`](./src/1_cnn_image_classification) dan [`src/2_rnn_image_captioning/`](./src/2_rnn_image_captioning).

---

## ✨ Fitur Utama

- 📐 **Deep Learning Framework**
   - Implementasi framework _neural network_ sepenuhnya berbasis NumPy murni _from scratch_.
   - Engine `Tensor` yang mendukung _forward_ dan _backward pass_ secara bersamaan dengan metode _autodifferentiation_.
   - Mendukung integrasi layer yang beragam (Conv2D, LocallyConnected2D, SimpleRNN, LSTM, Embedding, Pooling).
   - Kustomisasi fungsi aktivasi, metode inisialisasi bobot, fungsi loss, dan metode optimisasi.
   - Mendukung penyimpanan dan pemuatan parameter model, serta penerjemahannya dari `Keras`.
- 🖼️ **Convolutional Neural Network**
   - Pipeline klasifikasi gambar _end-to-end_ untuk [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
   - Visualisasi Grad-CAM untuk menginterpretasikan area fokus model.
- 🗣️ **Recurrent Neural Network (Encoder-Decoder)**
   - Pembangunan arsitektur _Encoder-Decoder_ untuk dataset [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k).
   - Memanfaatkan Pretrained-CNN sebagai _feature extractor_ yang diinjeksi ke dalam RNN/LSTM _decoder_ kustom.
- 🔬 **Comprehensive Experiments & Explorations**
   - Pengujian yang dilakukan secara _rigorous_ untuk mengerti arsitektur yang dibangun.
   - Eksperimen pengaruh _hyperparameter_ dieksplorasi secara khusus.

---

## 📁 Struktur Folder

```
Tubes2_AnakRudyKusdiantara/
├── data/
│   ├── 1_cnn_image_classification/
│   └── 2_rnn_image_captioning/
├── doc/
│   └── Tubes2_AnakRudyKusdiantara.pdf
├── src/
│   ├── 1_cnn_image_classification/
│   │   ├── 00_download_dataset.py
│   │   ├── 01_keras_training.ipynb
│   │   ├── 02_scratch_inference.ipynb
│   │   └── 03_grad_cam.ipynb
│   ├── 2_rnn_image_captioning/
│   │   ├── 00_download_dataset.py
│   │   ├── 01_extract_features.py
│   │   ├── 02_caption_preprocessing.ipynb
│   │   ├── 03_keras_training.ipynb
│   │   ├── 04_scratch_inference.ipynb
│   │   └── 05_bonus_experiments.ipynb
│   ├── nn/
│   │   ├── layers/
│   │   │   ├── conv.py
│   │   │   ├── dense.py
│   │   │   ├── embedding.py
│   │   │   ├── layer.py
│   │   │   ├── normalization.py
│   │   │   ├── pool.py
│   │   │   └── recurrent.py
│   │   ├── activations.py
│   │   ├── base.py
│   │   ├── engine.py
│   │   ├── initializers.py
│   │   ├── losses.py
│   │   ├── model.py
│   │   └── optimizers.py
│   ├── utility/
│   │   ├── image_utils.py
│   │   ├── model_loader.py
│   │   └── text_utils.py
│   └── doc.ipynb
└── README.md
```

---

## ⚙️ Requirement & Instalasi

### Prasyarat Package

- numpy>=2.3.2
- scipy>=1.16.1
- matplotlib>=3.10.6
- pandas>=2.3.2
- tqdm>=4.67.3
- seaborn>=0.13.2
- kaggle>=1.7.4.5
- Pillow>=11.3.0
- keras>=3.11.3
- tensorflow>=2.20.0
- notebook>=7.4.5
- nltk>=3.9.1

### Instalasi Package

1. Clone repository.

   ```bash
   git clone https://github.com/timoruslim/Tubes2_AnakRudyKusdiantara.git
   cd Tubes2_AnakRudyKusdiantara
   ```

2. Pasang dependensi.

   ```bash
   pip install -r requirements.txt
   ```

   Jika file `requirements.txt` belum ada, bisa install manual.

   ```bash
   pip install numpy scipy matplotlib pandas tqdm seaborn kaggle Pillow keras tensorflow notebook nltk
   ```

### Instalasi Dataset

Jika ingin melakukan bereksperimentasi dengan pelatihan atau pengujian kedua model kami, pastikan telah menyiapkan dataset sebagai berikut.

#### Bagian 1: CNN Image Classification

1. **Download Dataset:**
   Unduh dataset [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dengan skrip berikut.
   ```bash
   python src/1_cnn_image_classification/00_download_dataset.py
   ```

#### Bagian 2: RNN Image Captioning

1. **Download Dataset:**
   Unduh dataset [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) dengan skrip berikut.

   ```bash
   python src/2_rnn_image_captioning/00_download_dataset.py
   ```

2. **Feature Extraction:**
   Pre-process gambar menjadi fitur konteks menggunakan model `InceptionV4` dengan skrip berikut.

   ```bash
   python src/2_rnn_image_captioning/01_extract_features.py
   ```

3. **Caption Preprocessing:**
   Pre-process caption menjadi fitur token dengan menjalankan semua sel di [`src/2_rnn_image_captioning/02_caption_preprocessing.ipynb`](./src/2_rnn_image_captioning/02_caption_preprocessing.ipynb).

---

## 🚀 Menggunakan Package

Seluruh dokumentasi teknis, arsitektur kelas, dan contoh penggunaan _package_ `nn` buatan kami telah dirangkum dalam satu tutorial _notebook_ interaktif berikut.

&emsp; 👉 **[Buka Dokumentasi Utama](./src/doc.ipynb)**

Selain itu, untuk melihat penerapannya secara detail pada masalah dunia asli, alur pelatihan dan pengujian spesifik dapat dieksplorasikan pada _notebook_ eksperimen yang telah disusun berikut.

- **Image Classification:** Lihat folder [`src/1_cnn_image_classification/`](./src/1_cnn_image_classification)
- **Image Captioning:** Lihat folder [`src/2_rnn_image_captioning/`](./src/2_rnn_image_captioning)

---

## 👨‍💻 Authors

| Nama                 | NIM      | Pembagian Tugas                                                                                                                                                                            |
| -------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Beni Lesmana         | 10122043 | pelatihan model RNN di Keras, implementasi arsitektur Encoder-Decoder, eksperimen dan evaluasi model Encoder-Decoder, dan laporan                                                          |
| Albi Arrizkya Putra  | 10122062 | pelatihan model CNN di Keras, implementasi arsitektur CNN, eksperimen dan evaluasi model CNN, dan laporan                                                                                  |
| Timothy Niels Ruslim | 10123053 | implementasi utility functions, package utama yang termasuk forward dan backward propagation, layer-layer RNN dan CNN from scratch, feature extraction, preprocessing caption, dan laporan |

Bukti dapat dilihat dari sejarah _commits_.

---

## 🔗 Tautan

- 📂 [Repository GitHub](https://github.com/timoruslim/Tubes2_AnakRudyKusdiantara)

---

> Dibuat sebagai bagian dari Tugas Besar 2 IF3270 Pembelajaran Mesin 2026 – ITB
