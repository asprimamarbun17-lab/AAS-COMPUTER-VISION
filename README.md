ASPRIMA SABAR YUSUF MARBUN
4222301062
ROBOTIKA C PAGI 
SEMESTER 6 
UAS Computer Vision (RE604), Genap 2025/2026

# OCR Plat Nomor Kendaraan Indonesia menggunakan Visual Language Model (VLM)

Proyek Ujian Akhir Semester (UAS) mata kuliah **Computer Vision (RE604)** — Program Studi Teknik Robotika, Politeknik Negeri Batam.

Program ini melakukan *Optical Character Recognition* (OCR) pada plat nomor kendaraan Indonesia menggunakan **Visual Language Model (VLM)** yang dijalankan secara lokal melalui **LM Studio**, dan diintegrasikan dengan Python menggunakan library `openai` (memanfaatkan kompatibilitas API LM Studio dengan OpenAI API).

---

## Daftar Isi

- [Ringkasan Metode](#ringkasan-metode)
- [Struktur Dataset](#struktur-dataset)
- [Struktur Repository](#struktur-repository)
- [Kebutuhan Sistem](#kebutuhan-sistem)
- [Instruksi Eksekusi](#instruksi-eksekusi)
- [Output Program](#output-program)
- [Metrik Evaluasi (CER)](#metrik-evaluasi-cer)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Catatan Teknis](#catatan-teknis)

---

## Ringkasan Metode

1. Program terhubung ke **LM Studio** melalui local server (`http://127.0.0.1:1234/v1`) menggunakan library `openai`.
2. Model VLM yang sedang di-load di LM Studio (`qwen2-vl-2b-instruct`) diambil secara otomatis melalui endpoint `models.list()`.
3. Setiap gambar plat nomor pada folder `images/test` dikirim ke model dalam bentuk *data URI Base64*, disertai prompt:
   > "Read the vehicle license plate. Respond ONLY with the license plate number."
4. Ground truth (teks plat sebenarnya) **direkonstruksi dari label deteksi karakter berformat YOLO** yang tersedia pada dataset (lihat [Catatan Teknis](#catatan-teknis)), karena nama file gambar tidak merepresentasikan teks plat.
5. Hasil prediksi dibandingkan dengan ground truth menggunakan metrik **Character Error Rate (CER)**, lalu seluruh hasil disimpan ke `hasil_evaluasi_ocr.csv`.

---

## Struktur Dataset

Dataset yang digunakan: [Indonesian License Plate Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesianlicense-plate-dataset) (Kaggle), khusus folder **`test`**.

```
Indonesian License Plate Recognition Dataset/
├── images/
│   ├── test/        <- Digunakan pada proyek ini
│   ├── train/
│   └── val/
├── labels/
│   ├── test/        <- Label karakter format YOLO, digunakan sebagai ground truth
│   ├── train/
│   └── val/
├── classes.names     <- Daftar 34 karakter (0-9, A-X), urutan baris = class_id
├── train.txt
└── val.txt
```

Setiap file pada `labels/test/*.txt` berisi satu baris per karakter plat, dengan format:

```
class_id x_center y_center width height
```

Program ini merekonstruksi teks plat nomor dengan mengurutkan seluruh karakter pada satu file label berdasarkan koordinat `x_center` (kiri ke kanan), lalu memetakan tiap `class_id` ke karakter sesuai `classes.names`.

---

## Struktur Repository

```
.
├── main.py                    # Source code utama
├── hasil_evaluasi_ocr.csv     # Output evaluasi (dihasilkan setelah dijalankan)
└── README.md
```

> **Penting:** `main.py` harus diletakkan di dalam folder `images/test/` pada dataset yang sudah diunduh, karena program membaca gambar dari folder tempat ia berada, dan mencari `classes.names` serta `labels/test/` dua tingkat di atasnya (`../../classes.names` dan `../../labels/test`).

---

## Kebutuhan Sistem

- Python 3.9 atau lebih baru
- [LM Studio](https://lmstudio.ai/) (sudah terinstal dan berjalan)
- Model VLM multimodal yang sudah di-download dan di-load di LM Studio (proyek ini menggunakan **Qwen2-VL-2B-Instruct**)
- Library Python:
  ```bash
  pip install openai
  ```
  (`os`, `csv`, dan `base64` adalah modul bawaan Python, tidak perlu instalasi tambahan)

---

## Instruksi Eksekusi

1. **Unduh dataset** dari Kaggle: [Indonesian License Plate Dataset](https://www.kaggle.com/datasets/juanthomaswijaya/indonesianlicense-plate-dataset), lalu ekstrak.

2. **Jalankan LM Studio**, muat model VLM multimodal (misalnya Qwen2-VL-2B-Instruct), lalu aktifkan **Local Server** dari tab Developer (pastikan berjalan di port `1234`, sesuai `base_url` pada `main.py`).

3. **Clone repository ini**:
   ```bash
   git clone <link-repository-anda>
   ```

4. **Salin `main.py`** ke dalam folder:
   ```
   <lokasi-dataset>/Indonesian License Plate Recognition Dataset/images/test/
   ```

5. **Install dependency**:
   ```bash
   pip install openai
   ```

6. **Jalankan program** dari dalam folder `images/test/`:
   ```bash
   python main.py
   ```

7. Program akan:
   - Menampilkan nama model yang terdeteksi di LM Studio.
   - Merekonstruksi ground truth dari label YOLO.
   - Memproses seluruh gambar pada folder `test`, menampilkan progres di terminal.
   - Menyimpan hasil akhir ke `hasil_evaluasi_ocr.csv` pada folder yang sama.

---

## Output Program

File `hasil_evaluasi_ocr.csv` berisi kolom:

| Kolom | Keterangan |
|---|---|
| `image` | Nama file gambar |
| `ground_truth` | Teks plat nomor asli (hasil rekonstruksi dari label YOLO) |
| `prediction` | Teks plat nomor hasil prediksi model VLM |
| `CER_score` | Nilai Character Error Rate untuk gambar tersebut |

---

## Metrik Evaluasi (CER)

**Character Error Rate (CER)** mengukur seberapa banyak karakter pada prediksi berbeda dari ground truth:

```
CER = (S + D + I) / N
```

- **S** — jumlah karakter salah (substitusi)
- **D** — jumlah karakter yang hilang (delesi)
- **I** — jumlah karakter tambahan (insersi)
- **N** — jumlah karakter pada ground truth

Semakin kecil nilai CER, semakin akurat prediksi model. Nilai 0 berarti prediksi identik dengan ground truth. Perhitungan diimplementasikan menggunakan algoritma **Levenshtein Distance** berbasis *dynamic programming*.

---

## Hasil Evaluasi

Dari 197 gambar pada folder `test`:

| Metrik | Nilai |
|---|---|
| Rata-rata CER | 0,0577 (~5,8%) |
| Prediksi sempurna (CER = 0) | 157 dari 197 gambar (~79,7%) |

Contoh prediksi berhasil sempurna:

| Gambar | Ground Truth | Prediksi | CER |
|---|---|---|---|
| `test001_1.jpg` | `B9140BCD` | `B 9140 BCD` | 0,0 |

Contoh kasus gagal:

| Gambar | Ground Truth | Prediksi | CER | Penyebab |
|---|---|---|---|---|
| `test013_2.jpg` | `B1128WOS` | `The license plate number is WB 1,128 W0S.` | 3,375 | Model tidak patuh instruksi "respond only with the plate number" |
| `test065_1.jpg` | `H1706SW` | `H 1706 SW 11-26` | 0,7143 | Model ikut membaca teks masa berlaku plat |

---

## Catatan Teknis

- Ground truth **tidak** diambil dari nama file gambar (nama file bersifat generik, misalnya `test001_1.jpg`, dan tidak merepresentasikan teks plat). Ground truth direkonstruksi dari label deteksi karakter berformat YOLO pada folder `labels/test/`.
- Sebelum dibandingkan, teks ground truth dan prediksi dinormalisasi (spasi dihapus, huruf diseragamkan menjadi kapital) agar variasi format penulisan spasi pada plat tidak dihitung sebagai kesalahan.
- Parameter `temperature=0` digunakan pada pemanggilan model agar output bersifat deterministik (greedy decoding), sesuai kebutuhan tugas OCR yang menuntut satu jawaban paling mungkin.
- Program secara otomatis menyesuaikan tipe MIME gambar (`image/jpeg` atau `image/png`) berdasarkan ekstensi file sebelum dikirim ke LM Studio.

---

## Penulis# AAS-COMPUTER-VISION
