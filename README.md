<div align="center">

# Feed-Forward Neural Network

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.2-4DABCF?logo=numpy&logoColor=fff)
![SciPy](https://img.shields.io/badge/SciPy-1.17.1-8CA1E5?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-11557c?logo=python&logoColor=white)

</div>

## 📘 Deskripsi

Di sini, kami membuat package untuk mengimplementasikan **Multi-Layer Perceptron** (MLP) atau **Feed-Forward Neural Network** (FFNN). Package secara efisien dan modular menerapkan _forward pass_, _backpropagation_, dan optimisasi bobot yang menjadi pokok mekanisme MLP. Dengan pacakage ini, pengguna dapat membangun MLP sendiri dengan sangat mudah, seperti `Scikit`, tetapi dengan arsitektur yang sangat _customizable_, seperti `Pytorch` atau `TensorFlow`. Lalu, dilakukan analisis dan pemodelan dari dataset _Global Student Placement \& Salary Dataset_ menggunakan package yang sudah dibuat.

---

## ✨ Fitur Utama

- 📐 **Automatic Differentiation**
   - Kelas `Tensor` yang bisa menyimpan data dan gradien.
   - Pembentukan graf komputasi berisi `Tensor` untuk memudahkan _forward pass_ dan _backpropagation_.
- 🧩 **Customizble Layer**
   - Spesifikasi jumlah neuron, jenis aktivasi, inisialisasi bobot, dan regularisasi.
   - Implementasi RMSNorm.
- 🎯 **Efficient Fitting**
   - Spesifikasi _learning rate_, fungsi loss, dan metode optimisasi.
   - Implementasi Adam Optimizer.
- 🔍 **Model Transparency**
   - Pencatatan rinci nilai dan gradien bobot tiap layer.
   - Memungkinkan visualisasi, penyimpanan, dan pemuatan bobot.

---

## 📁 Struktur Folder

```

Tubes1_Six-Seven/
├── data/
│   └── datasetml_2026.csv
├── doc/
│   └── Tubes1_K1_Six-seven.pdf
├── src/
│   ├── ffnn/
│   │   ├── activation.py
│   │   ├── engine.py
│   │   ├── initialize.py
│   │   ├── loss.py
│   │   ├── nn.py
│   │   └── optimizer.py
│   └── pengujian/
│       ├── pengujian.ipynb
│       └── test.ipynb
└── README.md

```

---

## ⚙️ Requirement & Instalasi

### Prasyarat Package

- python ≥ 3.13
- numpy ≥ 2.2
- scipy ≥ 1.17.1
- matplotlib ≥ 3.10
- tqdm ≥ 4.67.3
- ipywidgets ≥ 8.1.8
- ipykernel (untuk notebook)

### Instalasi Package

1. Clone repository.

   ```bash
   git clone https://github.com/timoruslim/Tubes1_Six-Seven.git
   cd Tubes1_Six-Seven
   ```

2. Pasang dependensi.

   ```bash
   pip install -r requirements.txt
   ```

   Jika file `requirements.txt` belum ada, bisa install manual.

   ```bash
   pip install numpy scipy matplotlib tqdm ipywidgets ipykernel
   ```

### Pengujian

Jika ingin menggunakan file `ipynb` pada folder `\pengujian`, diperlukan juga berikut.

- scikit-learn ≥ 1.8.0
- pandas ≥ 3.0.1

Instalasinya sudah termasuk dalam file `ipynb`.

---

## 🚀 Menggunakan Package

Contoh penggunaan package (yang sederhana) dapat dilihat di `\pengujian\test.ipynb`. Untuk penggunaan langsung pada dataset, bisa lihat `\pengujian\pengujian.ipynb`. Selain itu, berikut "tutorial" atau "dokumentasi" singkat.

### **Consructor**

Untuk membangun suatu MLP, cukup dengan konsruktor `MLP` dengan memasukkan `Layer` yang diinginkan.

```Python
from ffnn import MLP, Layer

model = MLP([
   Layer(16, activation='relu', weight_init='he', l2=0.01),
   Layer(8, activation='relu', weight_init='he', l1=0.005),
   RMSNorm(8),
   Layer(1, activation='sigmoid', weight_init='xavier')
], input_size=2, seed=67)
```

### **Dynamic Layers**

Jika ingin menambah `Layer` baru, bisa dengan metode `add()` dari `MLP`.

```Python
model = MLP(input_size=10, seed=42)

for neurons in [32, 16, 8]:
   model.add(Layer(neurons, activation='swish', weight_init='he'))

model.add(Layer(3, activation='softmax', weight_init='xavier'))
```

### **Compilation**

Dengan metode `compile()`, suatu `MLP` dilengkapi dengan fungsi _loss_ dan metode optimisasi yang diinginkan.

- String:

   ```Python
   model.compile(optimizer='sgd', loss='mse')
   ```

- Hyperparameter:

   ```Python
   model.compile(
      optimizer={'method': 'adam', 'learning_rate': 0.05},
      loss='cce'
   )
   ```

### **Training and Prediction**

Untuk melakukan _training_, bisa dengan metode `fit()` dari `MLP`, yang melakukan validasi juga secara bersamaan. Untuk prediksi, bisa memanggil `MLP()` langsung.

```Python
history = model.fit(
   X_train, y_train,
   batch_size=32,
   epochs=100,
   learning_rate=0.01,
   validation_data=(X_val, y_val),
   verbose=1
)

plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')

model(X_pred)
```

### **Save and Load**

Simpan semua bobot suatu MLP ke file `.pkl` dengan `save()`. Lalu, jika arsitektur model sesuai, bobot dari suatu `.pkl` bisa dimuat dengan `load()`.

```Python
model.save('../saved_model.pkl')

blank_model = MLP([
   Layer(16, activation='relu'),
   Layer(8, activation='relu'),
   RMSNorm(8),
   Layer(1, activation='sigmoid')
], input_size=2)

blank_model.load('../saved_model.pkl')
```

---

## 👨‍💻 Author

| Nama                 | NIM      | Pembagian Tugas                          |
| -------------------- | -------- | ---------------------------------------- |
| Albi Arrizkya Putra  | 10122062 | analisis dan pemodelan dataset, laporan  |
| Timothy Niels Ruslim | 10123053 | implementasi FFNN dari _scrach_, laporan |

Bukti dapat dilihat dari sejarah _commits_.

---

## 🔗 Tautan

- 📂 [Repository GitHub](https://github.com/timoruslim/Tubes1_Six-seven)

---

> Dibuat sebagai bagian dari Tugas Besar 1 IF3270 Pembelajaran Mesin 2026 – ITB
