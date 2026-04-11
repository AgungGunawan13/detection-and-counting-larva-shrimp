from pathlib import Path
from ultralytics import YOLO
from utils import ensure_dir, find_yaml, print_header, timestamp

EXTRACTED_DIR = Path("data/extracted")
RUNS_DIR = Path("outputs/runs")

DEVICE = 0
EPOCHS = 50

# Untuk RTX 3050 4GB:
# - kalau OOM pada L/X, turunkan IMG_SIZE jadi 512
IMG_SIZE = 640

# (weights, batch)
MODELS = [
    ("yolov8n.pt", 4),
    ("yolov8s.pt", 4),
    ("yolov8m.pt", 2),
    ("yolov8l.pt", 1),
    ("yolov8x.pt", 1),
]

def find_data_yaml() -> Path:
    for d in EXTRACTED_DIR.iterdir():
        if d.is_dir():
            y = find_yaml(d)
            if y:
                return y
    raise FileNotFoundError("Tidak menemukan data.yaml di data/extracted/*")

def main():
    print_header("STEP 12 - TRAIN ALL MODELS (n/s/m/l/x)")
    ensure_dir(RUNS_DIR)

    data_yaml = find_data_yaml()
    print(f"[DATA] {data_yaml}")

    tag = timestamp()

    for weights, batch in MODELS:
        model_name = Path(weights).stem  # yolov8n, yolov8s, ...
        out_dir = RUNS_DIR / model_name
        ensure_dir(out_dir)

        print_header(f"TRAIN {model_name} | batch={batch} imgsz={IMG_SIZE} epochs={EPOCHS}")

        model = YOLO(weights)
        model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=batch,
            device=DEVICE,
            project=str(out_dir),
            name=f"train_{tag}",
            exist_ok=True,
            plots=True,
            patience=20,
        )

        print(f"[DONE] {model_name}")

    print("\nSelesai.")
    print("Lanjut evaluasi semua model).")

if __name__ == "__main__":
    main()

