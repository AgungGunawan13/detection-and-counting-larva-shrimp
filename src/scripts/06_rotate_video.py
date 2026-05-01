import cv2
from pathlib import Path

# ===============================
# ======= KONFIGURASI ===========
# ===============================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
VIDEO_PATH = ROOT_DIR / "data" / "VID_20260421_145813.mp4"
OUTPUT_DIR = ROOT_DIR / "data"
# ===============================

def main():
    print("\n--- PROGRAM ROTASI VIDEO TERPISAH ---")
    print("0: Tidak Dirotasi (Batal)")
    print("1: Putar 90 derajat searah jarum jam (+90)")
    print("2: Putar 180 derajat")
    print("3: Putar 90 derajat berlawanan jarum jam (-90)")
    
    pilihan_rotasi = input("Pilih opsi rotasi (0/1/2/3): ").strip()

    if pilihan_rotasi == "1":
        ROTATE_OPTION = cv2.ROTATE_90_CLOCKWISE
        suffix = "_rot90cw"
    elif pilihan_rotasi == "2":
        ROTATE_OPTION = cv2.ROTATE_180
        suffix = "_rot180"
    elif pilihan_rotasi == "3":
        ROTATE_OPTION = cv2.ROTATE_90_COUNTERCLOCKWISE
        suffix = "_rot90ccw"
    else:
        print("Membatalkan proses rotasi.")
        return

    video_path = Path(VIDEO_PATH)
    output_dir = Path(OUTPUT_DIR).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Peringatan: Video {video_path} tidak ditemukan.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Gagal membuka video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if ROTATE_OPTION in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        width, height = height, width

    output_video_path = output_dir / f"{video_path.stem}{suffix}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    print(f"\nMemulai proses rotasi... ({total_frames} frame)")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rotated_frame = cv2.rotate(frame, ROTATE_OPTION)
        out.write(rotated_frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Memproses frame ke-{frame_count} / {total_frames}...")

    cap.release()
    out.release()
    print(f"\nSelesai! Video hasil rotasi disimpan di:\n{output_video_path}")

if __name__ == "__main__":
    main()
