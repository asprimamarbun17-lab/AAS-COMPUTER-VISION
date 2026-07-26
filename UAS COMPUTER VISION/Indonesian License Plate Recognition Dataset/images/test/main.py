import os
import csv
import base64
from openai import OpenAI

# =====================================================
# Koneksi ke LM Studio
# =====================================================
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

# =====================================================
# Ambil model yang sedang di-load di LM Studio
# =====================================================
try:
    models = client.models.list()
    MODEL_NAME = models.data[0].id
    print(f"Model digunakan : {MODEL_NAME}")
except Exception as e:
    print("Tidak dapat mengambil model dari LM Studio.")
    print(e)
    exit()

# =====================================================
# Hitung CER
# =====================================================
def calculate_cer(reference, hypothesis):

    reference = reference.replace(" ", "").upper()
    hypothesis = hypothesis.replace(" ", "").upper()

    m = len(reference)
    n = len(hypothesis)

    d = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        d[i][0] = i

    for j in range(n+1):
        d[0][j] = j

    for i in range(1,m+1):
        for j in range(1,n+1):

            cost = 0 if reference[i-1] == hypothesis[j-1] else 1

            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )

    if m == 0:
        return 0

    return d[m][n] / m


# =====================================================
# [PERBAIKAN] Muat Ground Truth dari label YOLO per-karakter
# -----------------------------------------------------
# Struktur dataset asli (dikonfirmasi dari isi folder):
#
#   Indonesian License Plate Recognition Dataset/
#     images/test/<nama>.jpg   <- lokasi main.py & gambar
#     labels/test/<nama>.txt   <- 1 baris per karakter plat:
#                                  class_id x_center y_center w h
#     classes.names             <- daftar karakter, 1 per baris,
#                                  urutan baris = class_id (0-indexed)
#
# Karena tiap baris di file .txt merepresentasikan SATU
# karakter (bukan satu plat utuh), teks plat direkonstruksi
# dengan mengurutkan karakter berdasarkan koordinat
# x_center dari kiri ke kanan, lalu memetakan tiap class_id
# ke karakter sesuai classes.names.
# =====================================================
def load_classes(dataset_folder):

    classes_path = os.path.normpath(
        os.path.join(dataset_folder, "..", "..", "classes.names")
    )

    if not os.path.isfile(classes_path):
        return None

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [baris.strip() for baris in f if baris.strip() != ""]

    return classes


def load_ground_truth_labels(dataset_folder):

    classes = load_classes(dataset_folder)

    if not classes:
        print("[PERINGATAN] classes.names tidak ditemukan di root dataset.")
        return {}

    labels_test_folder = os.path.normpath(
        os.path.join(dataset_folder, "..", "..", "labels", "test")
    )

    if not os.path.isdir(labels_test_folder):
        print(f"[PERINGATAN] Folder label tidak ditemukan: {labels_test_folder}")
        return {}

    label_map = {}

    for file in os.listdir(labels_test_folder):

        if not file.lower().endswith(".txt"):
            continue

        image_stem = os.path.splitext(file)[0]

        path_txt = os.path.join(labels_test_folder, file)

        karakter_list = []

        with open(path_txt, "r", encoding="utf-8") as f:

            for baris in f:

                baris = baris.strip()

                if not baris:
                    continue

                bagian = baris.split()

                if len(bagian) < 5:
                    continue

                class_id = int(bagian[0])
                x_center = float(bagian[1])

                if class_id < 0 or class_id >= len(classes):
                    continue

                karakter_list.append((x_center, classes[class_id]))

        # Urutkan karakter dari kiri ke kanan berdasarkan posisi x_center
        karakter_list.sort(key=lambda pasangan: pasangan[0])

        teks_plat = "".join(karakter for _, karakter in karakter_list)

        for ekstensi in (".jpg", ".jpeg", ".png"):
            label_map[image_stem + ekstensi] = teks_plat

    if label_map:
        print(f"Ground truth direkonstruksi dari label YOLO: {labels_test_folder} ({len(label_map)} entri)")

    return label_map


# =====================================================
# [PERBAIKAN] Deteksi MIME type gambar sesuai ekstensi
# -----------------------------------------------------
# Sebelumnya MIME type selalu di-hardcode "image/jpeg"
# walau file aslinya .png, yang secara teknis kurang tepat.
# =====================================================
def get_mime_type(image_path):

    ekstensi = os.path.splitext(image_path)[1].lower()

    if ekstensi == ".png":
        return "image/png"

    if ekstensi in (".jpg", ".jpeg"):
        return "image/jpeg"

    return "image/jpeg"


# =====================================================
# OCR
# =====================================================
def predict_license_plate(image_path):

    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()

    mime_type = get_mime_type(image_path)

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            messages=[
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"text",
                            "text":"Read the vehicle license plate. Respond ONLY with the license plate number."
                        },
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":f"data:{mime_type};base64,{img}"
                            }
                        }
                    ]
                }
            ],

            temperature=0

        )

        return response.choices[0].message.content.strip()

    except Exception as e:

        return "ERROR : " + str(e)


# =====================================================
# Folder Dataset
# =====================================================

dataset_folder = os.path.dirname(os.path.abspath(__file__))

csv_filename = os.path.join(dataset_folder, "hasil_evaluasi_ocr.csv")

print("="*60)
print("OCR Plat Nomor Menggunakan LM Studio")
print("="*60)

# [PERBAIKAN] Muat ground truth asli sebelum mulai proses.
ground_truth_map = load_ground_truth_labels(dataset_folder)

if not ground_truth_map:
    print("\n[PERINGATAN] Ground truth tidak berhasil direkonstruksi.")
    print("Program akan tetap berjalan, tapi ground_truth akan dikosongkan")
    print("dan CER TIDAK akan dihitung, agar hasil evaluasi tidak menyesatkan.")
    print("Pastikan struktur folder berikut ada relatif terhadap folder gambar ini:")
    print("  ../../classes.names")
    print("  ../../labels/test/<nama_gambar>.txt\n")

images = []

for file in os.listdir(dataset_folder):

    if file.lower().endswith((".jpg",".jpeg",".png")):
        images.append(file)

images.sort()

print(f"Jumlah gambar ditemukan : {len(images)}")

if len(images) == 0:
    print("Tidak ada file gambar pada folder:")
    print(dataset_folder)
    exit()

with open(csv_filename,"w",newline="",encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    # [PERBAIKAN] Nama kolom disesuaikan dengan spesifikasi soal UAS
    writer.writerow([
        "image",
        "ground_truth",
        "prediction",
        "CER_score"
    ])

    total = len(images)

    for i, filename in enumerate(images, start=1):

        image_path = os.path.join(dataset_folder, filename)

        # [PERBAIKAN] Ambil ground truth dari file label, bukan nama file.
        ground_truth = ground_truth_map.get(filename, "")

        print(f"\n[{i}/{total}] {filename}")

        prediction = predict_license_plate(image_path)

        if prediction.startswith("ERROR") or ground_truth == "":

            cer = ""

        else:

            cer = round(
                calculate_cer(
                    ground_truth,
                    prediction
                ),
                4
            )

        writer.writerow([
            filename,
            ground_truth,
            prediction,
            cer
        ])

        print("GT   :", ground_truth if ground_truth else "(tidak ada label)")
        print("Pred :", prediction)
        print("CER  :", cer)

print("\n=========================================")
print("SELESAI")
print("CSV disimpan pada:")
print(csv_filename)
print("=========================================")