# Sistem Deteksi dan Perhitungan Real-time Benur Udang Windu

Repositori ini memuat *source code* murni dari sistem deteksi dan perhitungan benur udang secara _real-time_ menggunakan YOLOv8 dan antarmuka pengguna grafis (GUI) berbasis PyQt5.

**Catatan**: Repositori ini hanya berisi kode sumber. Dataset (`data/`), model terlatih (`models/`), luaran sistem (`outputs/`), catatan log (`logs/`), dan file dokumentasi hasil (`docs/`) sengaja diabaikan (`.gitignore`) untuk menghindari pembengkakan ukuran repositori GitHub.

## Struktur Repositori

```text
pengujian/
├── logs/                  # Folder untuk menyimpan log terminal/sistem
├── src/
│   ├── gui/                 # Source code untuk antarmuka pengguna (PyQt5)
│   │   ├── main.py          # Entry point aplikasi GUI
│   │   ├── UI.py            # Kode desain UI
│   │   └── UI.ui            # File desain Qt Designer
│   └── scripts/             # Kumpulan script eksekusi sistem yolov8
│       ├── 01_unzip_dataset.py
│       ├── 02_check_dataset.py
│       ├── 03_train_all_models.py
│       ├── 04_eval_and_compare.py
│       ├── 05_test_video.py
│       ├── 06_rotate_video.py
│       └── custom_tracker.yaml
├── README.md                # Dokumentasi utama proyek
├── requirements.txt         # Daftar dependensi library Python
└── .gitignore               # Konfigurasi pengabaian file Git
```

## Prasyarat dan Instalasi

Karena *Virtual Environment* lokal disarankan untuk manajemen proyek Python, Anda perlu membuatnya dari awal. Ikuti langkah-langkah berikut:

1. Buka terminal/Command Prompt di dalam folder utama repositori.
2. Buat *Virtual Environment* baru:
   ```bash
   python -m venv .venv
   ```
3. Aktifkan *Virtual Environment*:
   - Pada **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - Pada **Windows (CMD)**:
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
4. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

## Menjalankan Aplikasi

Pastikan *Virtual Environment* sudah aktif sebelum menjalankan aplikasi.

### 1. Menjalankan GUI Real-time Kamera
Masuk ke direktori GUI lalu eksekusi `main.py`:
```bash
python src/gui/main.py
```
*(Aplikasi akan otomatis mencari file model terlatih di direktori `models/trained/best.pt`. Pastikan Anda telah menempatkan bobot hasil pelatihan di sana sebelum menjalankan program).*

### 2. Menjalankan Skrip Pendukung
Skrip untuk mempersiapkan dataset, melakukan _training_, dan pengujian komparatif dapat ditemukan di dalam direktori `src/scripts/`. Semua skrip ini telah dikonfigurasi untuk menyesuaikan dengan _path_ repositori yang baru.
```bash
python src/scripts/05_test_video.py
```
