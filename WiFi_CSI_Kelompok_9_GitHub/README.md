# WiFi CSI Indoor Localization — Kelompok 9

**Mata kuliah:** Pembelajaran Mesin Lanjut (UTS)  
**Track:** B — Indoor Localization & Positioning  
**Baseline:** Multi-Layer Perceptron (MLP)  
**Pembanding:** CNN-LSTM (PyTorch)  
**Output:** regresi posisi dua dimensi `(x, y)` dalam satuan **grid**, bukan meter.

> Proyek ini merupakan pemisahan kode modular dari `notebooks/Deep_Learning_Kelompok_9.ipynb` milik Kelompok 9. Notebook tersebut merupakan rekaman eksperimen yang dipresentasikan; skrip modular di sini memungkinkan alur penelitian dijalankan ulang. Angka hasil run baru dapat berbeda dan **tidak boleh diklaim sebagai hasil final sebelumnya** tanpa verifikasi.

## Dataset dan rujukan

Dataset **subset demonstrasi** diunduh dari repositori pihak ketiga:

- [WIFI-SCI-Indoor-Positioning](https://github.com/zhangleino1/WIFI-SCI-Indoor-Positioning) — folder `dataset` (data CSV CSI berlabel koordinat).
- [Awesome-WiFi-CSI-Sensing](https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing) — kumpulan referensi bidang WiFi CSI sensing; **bukan dataset utama** pada eksperimen ini.

CSV sumber sudah menyediakan `amplitude_0..29`, `phase_0..29` dan `timestamp`. Kode ini **tidak mengekstrak CSI mentah kompleks dari perangkat WiFi**. Nama file, misalnya `antenna_2_3_5.csv`, memberikan antena=2, X=3, Y=5. Dataset contoh notebook terdiri dari **270 CSV, 90 koordinat, 3 antena**. Data mentah tidak disertakan dalam repositori untuk menghormati hak distribusi sumber.

## Struktur proyek

```text
WiFi_CSI_Kelompok_9_GitHub/
├── README.md
├── requirements.txt
├── .gitignore
├── download_data.py           # Mengunduh data pihak ketiga
├── prepare_data.py            # Build tensor, split zona per lokasi, normalisasi
├── train.py                   # Training MLP + CNN-LSTM, simpan .pt
├── evaluate.py                # Mean, median, P90, Confusion Matrix, grafik
├── ablation.py                # MLP dengan vs tanpa Butterworth
├── demo.py                    # Demo koordinat asli vs prediksi
├── src/
│   ├── config.py             # Parameter eksperimen
│   ├── preprocessing.py      # Noise handling, filter, phase sanitization
│   ├── dataset.py            # Split per lokasi, tanpa kebocoran data
│   ├── models.py             # Definisi arsitektur PyTorch
│   └── engine.py             # Training, early stopping, inference
├── tests/test_project.py      # Pemeriksaan kode otomatis
├── notebooks/Deep_Learning_Kelompok_9.ipynb
├── data/raw/                 # CSV akan berada di sini (tidak masuk Git)
└── artifacts/
    ├── prepared.npz           # Data testing/split yang diproses
    ├── models/                # Checkpoint .pt (tidak masuk Git)
    └── results/               # Hasil CSV/PNG (tidak masuk Git)
```

## Cara menjalankan: Windows (untuk pemula)

**Prasyarat:** Python 3.10 atau 3.11 dan Git for Windows telah terinstal. Pastikan terhubung ke internet untuk mengunduh dataset dan library. GPU **tidak wajib**; training CPU bisa lebih lama.

1. Unduh ZIP proyek, klik kanan > **Extract All**, kemudian buka folder hasil ekstrak.
2. Klik bilah alamat folder di File Explorer, ketik `powershell`, tekan Enter.
3. Jalankan satu per satu:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python download_data.py
python prepare_data.py
python train.py
python evaluate.py
python ablation.py
python demo.py --index 0
```

Jika PowerShell menolak aktivasi `Activate.ps1`, gunakan `cmd` dengan `.venv\Scripts\activate.bat`, atau langsung panggil `.venv\Scripts\python.exe` sebagai pengganti `python` pada perintah setelahnya. Di Windows, apabila `python` tidak ditemukan, coba perintah `py -3.11` untuk membuat environment.

**Catatan waktu:** `train.py` dan `ablation.py` melatih model. Tidak perlu dijalankan ulang jika kamu hanya ingin menampilkan model yang sudah dilatih sebelumnya.

## Cara menjalankan: Google Colab (cocok jika notebook sebelumnya di Colab)

Di sel Colab, unggah ZIP proyek dari komputer, lalu jalankan:

```python
from google.colab import files
uploaded = files.upload()  # pilih WiFi_CSI_Kelompok_9_GitHub.zip
!unzip -oq WiFi_CSI_Kelompok_9_GitHub.zip -d /content
%cd /content/WiFi_CSI_Kelompok_9_GitHub
!pip install -q -r requirements.txt
!python download_data.py
!python prepare_data.py
!python -m unittest discover -s tests -v
```

**Menggunakan checkpoint `.pt` lama dari Google Drive tanpa training ulang:**

```python
from google.colab import drive
drive.mount('/content/drive')
!python evaluate.py --model-dir "/content/drive/MyDrive/Project_UTS_CSI_Localization/models"
!python demo.py --index 0 --model-dir "/content/drive/MyDrive/Project_UTS_CSI_Localization/models"
```

Perintah di atas memuat `MLP.pt` dan `CNN_LSTM.pt` lama, lalu mengevaluasi ulang dengan preprocessing dan split yang ditentukan di proyek ini. **Periksa bahwa metriknya sama dengan laporan; jika berbeda, jangan menimpa angka laporan tanpa investigasi** (dataset, checkpoint, seed, split, versi library, atau urutan sampel bisa berbeda). Model disimpan sebagai `state_dict`, sehingga nama arsitekturnya harus sama.

Untuk melatih **dari awal** di Colab:

```python
!python train.py --epochs 20
!python evaluate.py
!python ablation.py --epochs 20
!python demo.py --index 0
```

Pada run baru, nilai statistik tidak dijamin identik dengan angka sebelumnya meskipun split, model dan parameter utamanya mengacu notebook; hasil training dipengaruhi inisialisasi, pengacakan dan jenis perangkat.

## Metodologi

1. **Baca data**: 3 antena tiap koordinat, amplitude + phase pada 30 subcarrier.
2. **Noise handling**: nilai `NaN`, `inf`, `-inf`, dan string tidak valid diganti dengan `NaN`; interpolasi temporal dilakukan **hanya pada file antena yang sama**. Kolom yang seluruhnya invalid menghasilkan error eksplisit.
3. **Filtering**: Butterworth low-pass orde 3, cutoff ternormalisasi 0.25, `sosfiltfilt`, **hanya untuk amplitude**.
4. **Phase sanitization**: `np.unwrap` melintasi sumbu subcarrier dan `scipy.signal.detrend` linier pada sumbu yang sama. Ini **sanitasi sederhana**, bukan kalibrasi fase CSI perangkat secara penuh.
5. **Tensor**: tiga antena × (amplitude + phase) → bentuk `(N,6,15,30)`, window 15, stride 5.
6. **Split**: stratifikasi empat zona pada tingkat **lokasi**, bukan window; train sekitar 70%, validation 15%, test 15%. Lokasi yang sama tidak muncul pada lebih dari satu split.
7. **Normalisasi**: mean/std tiap channel dihitung dari training saja.
8. **Training**: MLP dan CNN-LSTM, Adam LR=0.001, SmoothL1, batch size 32, maksimum 20 epoch, early stopping patience 6.
9. **Regresi**: metrik utama jarak Euclidean rata-rata, median, persentil 90 dalam **grid**.
10. **Evaluasi zona**: klasifikasi empat zona **diturunkan dari koordinat regresi** dengan batas X=4.5, Y=4.5; gambar Confusion Matrix adalah evaluasi tambahan, bukan model klasifikasi tersendiri.
11. **Ablation**: MLP dengan vs tanpa Butterworth, split sama, epoch sama, seed training ablation 2026, normalisasi dihitung masing-masing dari data training skenario.

## Hasil pada notebook yang dilaporkan 

| Model | Mean error (grid) | Median (grid) | P90 (grid) | Akurasi zona |
|---|---:|---:|---:|---:|
| MLP | 3.542555 | 3.712326 | 5.568629 | 33.57% |
| CNN-LSTM | **3.369374** | **3.611019** | **5.108917** | 30.00% |

Penurunan mean error model CNN-LSTM terhadap baseline MLP sekitar **4.89%**. Evaluasi zona menunjukkan MLP sedikit lebih baik; keduanya masih rendah dan dekat baseline kelas mayoritas (28.57%). **Tidak ada klaim akurasi dalam meter maupun kesiapan penerapan lapangan.**

**Ablation MLP:** Dengan Butterworth mean error 3.563384 grid, tanpa Butterworth 3.345981 grid. Pada percobaan ini tanpa filter lebih rendah, sehingga filter tidak selalu meningkatkan performa.

Angka di atas merupakan **hasil notebook yang diberikan**, bukan metrik dari skrip yang baru saya jalankan pada data asli. Perintah `evaluate.py` akan menulis ulang hasil sesuai checkpoint yang digunakan ke folder `artifacts/results/`.

## Berkas hasil otomatis

- `artifacts/models/MLP.pt` dan `CNN_LSTM.pt`: checkpoint model.
- `artifacts/results/model_comparison.csv`: perbandingan error posisi.
- `artifacts/results/zone_classification_comparison.csv`: accuracy, precision/recall/F1 makro.
- `artifacts/results/confusion_matrix_{mlp,cnn_lstm}.{csv,png}`: matriks kebingungan.
- `artifacts/results/localization_scatter.png`: visualisasi posisi.
- `artifacts/results/loss_comparison.png`: loss training dan validasi.
- `artifacts/results/ablation_mlp_filter.csv` dan grafik ablation.

Grafik yang sudah ada di notebook Colab dapat dilampirkan melalui hasil ZIP dari Colab. Folder `artifacts` dan `data/raw` sengaja tidak ikut `git add` supaya repositori tetap ringan dan tidak menduplikasi dataset pihak ketiga.





