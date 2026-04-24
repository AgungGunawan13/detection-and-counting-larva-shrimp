from pathlib import Path
import cv2
from ultralytics import YOLO
from collections import defaultdict

# ===============================
# ======= KONFIGURASI ===========
# ===============================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

WEIGHTS_PATH = ROOT_DIR / "YOLOv8" / "outputs" / "runs" / "yolov8s" / "train_20260418-000303" / "weights" / "best.pt"
VIDEO_PATH   = ROOT_DIR / "YOLOv8" / "data" / "k.mp4"
OUTPUT_DIR   = ROOT_DIR / "YOLOv8" / "outputs" / "reports_video"

CONFIDENCE = 0.15
IMG_SIZE   = 1280
MANUAL_COUNT = 50

LINE_POSITION_RATIO = 0.6
OFFSET = 20
ARAH   = "TOP_BOTTOM"
DEVICE = 0
ID_EXPIRY_FRAMES = 90
BRIGHTNESS_ALPHA = 0.8 # Mengurangi brightness 20% (1.0 = normal)

# ===============================

def main():
    print("\n--- PENGATURAN KECEPATAN VIDEO HASIL ---")
    print("1: Normal (Default)")
    print("2: Slow Motion (Diperlambat menjadi 30% / 0.3x)")
    pilihan_speed = input("Pilih kecepatan video (1/2): ").strip()

    weights    = Path(WEIGHTS_PATH)
    video_path = Path(VIDEO_PATH)
    output_dir = Path(OUTPUT_DIR).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        print(f"[ERROR] Model tidak ditemukan: {weights}")
        return
    if not video_path.exists():
        print(f"[ERROR] Video tidak ditemukan: {video_path}")
        return

    print(f"\nLoading model dari: {weights}")
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[ERROR] Gagal membuka video.")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    LINE_POS    = int(height * LINE_POSITION_RATIO)
    garis_atas  = LINE_POS - OFFSET
    garis_bawah = LINE_POS + OFFSET

    fps_out = fps * 0.3 if pilihan_speed == "2" else fps
    output_video_path = output_dir / f"{video_path.stem}_counted.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(output_video_path), fourcc, fps_out, (width, height))

    total_benur     = 0
    track_history   = {}   # {id: {'zone': str|None, 'last_seen': int}}
    track_paths     = defaultdict(list)
    counted_ids     = set()
    flash_remaining = 0

    frame_count     = 0
    total_pre_time  = 0.0
    total_inf_time  = 0.0
    total_post_time = 0.0

    print("\n" + "="*50)
    print(f"ARAH          : {ARAH}")
    print(f"Resolusi      : {width} x {height}")
    print(f"Garis utama   : y={LINE_POS}  (zona: {garis_atas} ~ {garis_bawah})")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- AUGMENTASI: Mengubah Brightness ---
        if BRIGHTNESS_ALPHA != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=BRIGHTNESS_ALPHA, beta=0)

        results = model.track(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            persist=True,
            tracker=str(ROOT_DIR / "YOLOv8" / "scripts" / "custom_tracker.yaml"),
            verbose=False
        )

        r = results[0]
        total_pre_time  += r.speed['preprocess']
        total_inf_time  += r.speed['inference']
        total_post_time += r.speed['postprocess']

        annotated_frame = r.plot()

        # --- Gambar garis zona ---
        cv2.line(annotated_frame, (0, garis_atas),  (width, garis_atas),  (255, 200, 0), 1)
        cv2.line(annotated_frame, (0, garis_bawah), (width, garis_bawah), (255, 200, 0), 1)
        line_color = (0, 255, 0) if flash_remaining > 0 else (0, 0, 255)
        line_thick = 8           if flash_remaining > 0 else 3
        cv2.line(annotated_frame, (0, LINE_POS), (width, LINE_POS), line_color, line_thick)
        if flash_remaining > 0:
            flash_remaining -= 1

        # --- Logika crossing ---
        if r.boxes is not None and r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
            boxes     = r.boxes.xyxy.cpu().tolist()

            for track_id, box in zip(track_ids, boxes):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                cv2.circle(annotated_frame, (cx, cy), 6, (255, 255, 0), -1)

                track_paths[track_id].append((cx, cy))
                if len(track_paths[track_id]) > 40:
                    track_paths[track_id].pop(0)
                for i in range(1, len(track_paths[track_id])):
                    cv2.line(annotated_frame,
                             track_paths[track_id][i-1],
                             track_paths[track_id][i], (0, 165, 255), 2)

                if track_id not in counted_ids:
                    info = track_history.get(track_id, {'zone': None, 'last_seen': frame_count})
                    info['last_seen'] = frame_count

                    if ARAH == "TOP_BOTTOM":
                        # Benur bergerak dari atas ke bawah
                        # Harus terlihat di ATAS garis dulu, baru dihitung saat di BAWAH
                        if cy < garis_atas:
                            info['zone'] = 'ATAS'
                        elif cy > garis_bawah and info['zone'] == 'ATAS':
                            total_benur += 1
                            counted_ids.add(track_id)
                            info['zone'] = 'BAWAH'
                            flash_remaining = 8
                            print(f"[Frame {frame_count}] +1 Benur! ID={track_id} | Total={total_benur}")

                    elif ARAH == "BOTTOM_TOP":
                        # Benur bergerak dari bawah ke atas
                        # Harus terlihat di BAWAH garis dulu, baru dihitung saat di ATAS
                        if cy > garis_bawah:
                            info['zone'] = 'BAWAH'
                        elif cy < garis_atas and info['zone'] == 'BAWAH':
                            total_benur += 1
                            counted_ids.add(track_id)
                            info['zone'] = 'ATAS'
                            flash_remaining = 8
                            print(f"[Frame {frame_count}] +1 Benur! ID={track_id} | Total={total_benur}")

                    track_history[track_id] = info

        # Bersihkan ID kedaluwarsa
        expired = [
            tid for tid, info in track_history.items()
            if frame_count - info.get('last_seen', 0) > ID_EXPIRY_FRAMES
            and tid not in counted_ids
        ]
        for tid in expired:
            track_history.pop(tid, None)
            track_paths.pop(tid, None)

        # --- HUD ---
        teks = f"TOTAL BENUR: {total_benur}"
        (tw, _), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4)
        cv2.rectangle(annotated_frame, (20, 15), (30 + tw + 10, 70), (0, 0, 0), -1)
        cv2.putText(annotated_frame, teks, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 4, cv2.LINE_AA)

        out.write(annotated_frame)
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Frame ke-{frame_count} ...")

    cap.release()
    out.release()

    # ==============================
    # LAPORAN AKHIR (cetak 1x saja)
    # ==============================
    avg_pre   = total_pre_time  / frame_count if frame_count > 0 else 0
    avg_inf   = total_inf_time  / frame_count if frame_count > 0 else 0
    avg_post  = total_post_time / frame_count if frame_count > 0 else 0
    avg_total = avg_pre + avg_inf + avg_post
    avg_fps   = 1000 / avg_total if avg_total > 0 else 0

    print("\n" + "="*50)
    print("SELESAI.")
    print(f"  Total frame diproses   : {frame_count}")
    print(f"  Total benur terhitung  : {total_benur}")
    print("-"*50)
    print("  EVALUASI KECEPATAN")
    print(f"  Preprocess   : {avg_pre:.2f} ms")
    print(f"  Inference    : {avg_inf:.2f} ms")
    print(f"  Postprocess  : {avg_post:.2f} ms")
    print(f"  Total latency: {avg_total:.2f} ms")
    print(f"  FPS rata-rata: {avg_fps:.2f}")

    if MANUAL_COUNT > 0:
        selisih    = abs(total_benur - MANUAL_COUNT)
        error_rate = (selisih / MANUAL_COUNT) * 100
        akurasi    = 100 - error_rate
        print("-"*50)
        print("  EVALUASI AKURASI")
        print(f"  Ground-truth (manual): {MANUAL_COUNT}")
        print(f"  Sistem AI            : {total_benur}")
        print(f"  Selisih / Error      : {selisih} ({error_rate:.2f}%)")
        print(f"  Akurasi              : {akurasi:.2f}%")

    print("="*50)
    print(f"\nOutput: {output_video_path}\n")


if __name__ == "__main__":
    main()