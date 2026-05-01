from pathlib import Path
import cv2
from ultralytics import YOLO

# ===============================
# ======= KONFIGURASI ===========
# ===============================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Menggunakan model yolov8s yang sudah ditraining
WEIGHTS_PATH = ROOT_DIR / "models" / "trained" / "best.pt"

CONFIDENCE = 0.15
IMG_SIZE   = 640
DEVICE     = 0 # Gunakan 0 untuk GPU, atau 'cpu' jika ingin pakai CPU
# ===============================

def main():
    weights = Path(WEIGHTS_PATH)
    if not weights.exists():
        print(f"[ERROR] Model tidak ditemukan: {weights}")
        return

    print(f"\n[INFO] Loading model dari: {weights}")
    model = YOLO(str(weights))

    # Buka kamera laptop (index 0 biasanya adalah kamera bawaan)
    print("\n[INFO] Mencoba membuka kamera laptop...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Gagal membuka kamera. Pastikan kamera tidak sedang dipakai aplikasi lain (seperti Zoom/Meet).")
        return

    print("\n" + "="*50)
    print("KAMERA AKTIF: Tekan tombol 'q' pada keyboard untuk keluar.")
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal membaca frame dari kamera.")
            break

        # Lakukan deteksi YOLOv8
        # Catatan: Kita gunakan predict karena ini sekadar tes deteksi sederhana, belum tracking/counting
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        # Ambil hasil frame yang sudah digambar kotak (bounding box)
        annotated_frame = results[0].plot()

        # Tampilkan ke layar
        cv2.imshow("YOLOv8 - Realtime Webcam Test (Tekan Q untuk keluar)", annotated_frame)

        # Keluar jika menekan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[INFO] Mematikan kamera...")
            break

    # Bersihkan memori dan tutup jendela
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Selesai.")

if __name__ == "__main__":
    main()
