from pathlib import Path
from ultralytics import YOLO
from utils import ensure_dir, find_yaml, print_header

EXTRACTED_DIR = Path("data/extracted")
EVAL_DIR = Path("outputs/eval")
DEVICE = 0

def find_data_yaml() -> Path:
    for d in EXTRACTED_DIR.iterdir():
        if d.is_dir():
            y = find_yaml(d)
            if y:
                return y
    raise FileNotFoundError("Tidak menemukan data.yaml di data/extracted/*")

def infer_model_name(weight_path: Path) -> str:
    # coba tebak nama model dari path: yolov8n/yolov8s/...
    for part in weight_path.parts:
        if part.startswith("yolov8"):
            return part
    # fallback: pakai parent folder terdekat
    return weight_path.parent.parent.name if weight_path.parent.parent else "model"

def main():
    print_header("EVAL ALL MODELS (GLOBAL SEARCH best.pt)")
    ensure_dir(EVAL_DIR)

    data_yaml = find_data_yaml()

    # cari best.pt di seluruh project (tapi skip folder .venv agar cepat)
    roots = [Path(".")]
    best_pts = []
    for r in roots:
        for p in r.rglob("best.pt"):
            # skip .venv
            if ".venv" in p.parts:
                continue
            best_pts.append(p)

    if not best_pts:
        raise FileNotFoundError("Tidak menemukan best.pt di project ini. Pastikan training sudah selesai.")

    # urutkan berdasarkan waktu terbaru
    best_pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # ambil yang unik per model (yolov8n/yolov8s/...)
    picked = {}
    for p in best_pts:
        m = infer_model_name(p)
        if m not in picked:
            picked[m] = p

    print("Model ditemukan:")
    for m, p in picked.items():
        print(f" - {m}: {p}")

    for model_name, best in picked.items():
        out = EVAL_DIR / model_name
        ensure_dir(out)

        print_header(f"EVAL {model_name}")
        print(f"[WEIGHTS] {best}")

        model = YOLO(str(best))

        for split in ["val", "test"]:
            print(f"\n[RUN] split={split}")
            model.val(
                data=str(data_yaml),
                split=split,
                device=DEVICE,
                project=str(out),
                name=split,
                exist_ok=True,
                plots=True,  # confusion matrix + kurva
            )

        print(f"\n[DONE] {model_name}")

    print("\nSelesai Step 13. Confusion matrix ada di outputs/eval/<model>/val/ dan /test/")

if __name__ == "__main__":
    main()
