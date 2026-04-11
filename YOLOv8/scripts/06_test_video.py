from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

# ===============================
# ======= KONFIGURASI ===========
# ===============================

# GANTI sesuai path model kamu (Otomatis diseragamkan dengan konfigurasi best.pt terbaru Anda)
WEIGHTS_PATH = r"YOLOv8\runs\detect\outputs\runs\yolov8s\train_20260220-004520\weights\best.pt"

# GANTI sesuai lokasi file video Anda
VIDEO_PATH = r"YOLOv8\data\k.mp4"

# Lokasi Hasil Render Video (Pastikan path folder benar)
OUTPUT_DIR = r"YOLOv8\outputs\reports_video"

CONFIDENCE = 0.55
IMG_SIZE = 640

# --- EVALUASI SKRIPSI ---
# Masukkan jumlah hitungan BENAR (Ground-Truth) menurut mata kepala / perhitungan manual Anda sendiri pada video ini.
# Tujuannya untuk mencari rumus "Tingkat Akurasi Penghitungan Sistem (%)".
MANUAL_COUNT = 100 

# GANTI INI UNTUK MENYESUAIKAN TINGGI GARIS VIRTUAL (Format rasio persentase ketinggian bingkai).
# 0.6 artinya garis horisontal terletak sedikit mendistribusi agak ke bawah (60% dari Puncak layar ke Dasar layar).
LINE_POSITION_RATIO = 0.6 

DEVICE = 0 # Angka 0 bermakna 'GPU CUDA ke-1'. Ini MEMAKSA YOLO harus pakai GPU tanpa toleransi.

# ===============================

def main():
    weights = Path(WEIGHTS_PATH)
    video_path = Path(VIDEO_PATH)
    output_dir = Path(OUTPUT_DIR).absolute()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        print(f"Peringatan: Model {weights} tidak ditemukan. Harap pastikan WEIGHTS_PATH sudah benar.")
        return

    if not video_path.exists():
        print(f"Peringatan: Video {video_path} tidak ditemukan. Harap pastikan lokasi VIDEO_PATH sudah benar.")
        return

    print(f"Loading model dari: {weights}")
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Gagal membuka video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Koordinat sumbu Y mutlak untuk garis Virtual (Horisontal)
    LINE_Y = int(height * LINE_POSITION_RATIO) 

    output_video_path = output_dir / f"{video_path.stem}_counted.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    # Variabel Penghitung Memori Tracker
    total_benur = 0
    track_history = {} # Penyimpan jejak riwayat Y koordinat si benur

    print("\n" + "="*50)
    print("MEMULAI INFERENSI TRACKING BENUR (TOP -> BOTTOM)")
    print(f"Resolusi Layar: {width} x {height}")
    print(f"Garis penghitung berada pada Kordinat Y: {LINE_Y}")
    print("="*50)

    frame_count = 0
    total_pre_time = 0.0
    total_inf_time = 0.0
    total_post_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # PERUBAHAN SKRIPSI: Memakai track bukan sekedar murni predict
        # Agar identitas/KTP masing-masing benur dipertahankan antara frame-ke-frame.
        results = model.track(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            persist=True,  
            tracker="botsort.yaml", # Algoritma pelacak resmi Ultralytics
            verbose=False
        )

        r = results[0]
        
        # Merekam latensi kecepatan (Ms) tiap iterasi frame
        total_pre_time += r.speed['preprocess']
        total_inf_time += r.speed['inference']
        total_post_time += r.speed['postprocess']

        # Menggambar bounding box bawaan YOLO + Nomer ID pada layar video 
        annotated_frame = r.plot()

        # Menggambar Garis Melintang OpenCV berwarna Merah (BGR = 0, 0, 255)
        cv2.line(annotated_frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 3)

        # Logika Persilangan Layam (*Line Crossing*)
        if r.boxes is not None and r.boxes.id is not None:
            # Mengekstrak list [ID benur] beserta kordinat box-nya di detik ini
            track_ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().tolist()
            
            for track_id, box in zip(track_ids, boxes):
                x1, y1, x2, y2 = box
                # Menghitung titik pusat badan benur (Centroid)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Menandai lambung benur dengan titik bulat biru muda untuk visual pelacakan
                cv2.circle(annotated_frame, (cx, cy), 6, (255, 255, 0), -1)

                if track_id in track_history:
                    prev_cy = track_history[track_id]
                    
                    # SYARAT HITUNG: 
                    # Jika bingkai kemarin benur berada DI ATAS garis (< LINE_Y)
                    # Namun bingkai sekarang perut dia melewati batas BAWAH garis (>= LINE_Y)
                    if prev_cy < LINE_Y and cy >= LINE_Y:
                        total_benur += 1
                        print(f"[Frame {frame_count}] +1 Benur Melintas! (ID Pelacakan: {track_id} | Total Sekarang: {total_benur})")
                        
                        # Animasi Kilatan Flash Garis: Mengubah garis jadi warna hijau sesaat ada benur yg menyeberang
                        cv2.line(annotated_frame, (0, LINE_Y), (width, LINE_Y), (0, 255, 0), 8)

                # Simpan kembali posisi y yang terbaru bagi id pelacakan ini ke kantong memori untuk dibandingkan sesaat lagi di frame berikutnya.
                track_history[track_id] = cy

        # Papan Informasi Perhitungan 
        teks_hitung = f"TOTAL BENUR: {total_benur}"
        (tw, th), _ = cv2.getTextSize(teks_hitung, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)
        cv2.rectangle(annotated_frame, (20, 15), (30 + tw + 10, 50 + 20), (0, 0, 0), -1)
        cv2.putText(annotated_frame, teks_hitung, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 4, cv2.LINE_AA)

        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Sedang merender frame ke-{frame_count} ...")

    cap.release()
    out.release()

    print("\n" + "="*50)
    print("SELESAI. RENDERING FILE VIDEO BERHASIL.")
    print(f"Total bingkai video (frames) : {frame_count}")
    print(f"Total Akhir Benur Masuk      : {total_benur} objek")
    
    # --- HITUNGAN EVALUASI KECEPATAN (FPS / MS) BAB 4 SKRIPSI ---
    avg_inf = total_inf_time / frame_count if frame_count > 0 else 0
    avg_pre = total_pre_time / frame_count if frame_count > 0 else 0
    avg_post = total_post_time / frame_count if frame_count > 0 else 0
    
    avg_total_latency_ms = avg_pre + avg_inf + avg_post
    avg_fps = 1000 / avg_total_latency_ms if avg_total_latency_ms > 0 else 0

    print("-" * 50)
    print("====== HASIL EVALUASI KECEPATAN (GPU) ======")
    print(f"Rata-rata Preprocess   : {avg_pre:.2f} ms")
    print(f"Rata-rata Inference    : {avg_inf:.2f} ms")
    print(f"Rata-rata Postprocess  : {avg_post:.2f} ms")
    print(f"Total Latensi / Frame  : {avg_total_latency_ms:.2f} ms")
    print(f"=====> RATA-RATA F.P.S : {avg_fps:.2f} Frame/Detik <=====")
    
    # --- HITUNGAN EVALUASI AKURASI BAB 4 SKRIPSI ---
    if MANUAL_COUNT > 0:
        selisih = abs(total_benur - MANUAL_COUNT)
        error_rate = (selisih / MANUAL_COUNT) * 100
        akurasi = 100 - error_rate
        print("-" * 50)
        print("====== HASIL EVALUASI AKURASI SKRIPSI ======")
        print(f"Hitungan Mata Manusia (Ground-Truth) : {MANUAL_COUNT} objek")
        print(f"Hitungan Tracking AI (System Count)  : {total_benur} objek")
        print(f"Selisih Margin Error                 : {selisih} objek ({error_rate:.2f}%)")
        print(f"=====> TINGKAT AKURASI SISTEM A.I    : {akurasi:.2f}% <=====")
        print("============================================")

    print(f"\nHasil output video disimpan di:\n-> {output_video_path}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
