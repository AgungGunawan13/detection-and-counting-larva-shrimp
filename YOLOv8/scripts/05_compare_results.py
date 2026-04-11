from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from utils import ensure_dir, find_yaml, print_header, chdir_project_root, PROJECT_ROOT

DEVICE = 0

def find_data_yaml() -> Path:
    extracted = PROJECT_ROOT / "data" / "extracted"
    for d in extracted.iterdir():
        if d.is_dir():
            y = find_yaml(d)
            if y:
                return y
    raise FileNotFoundError("Tidak menemukan data.yaml di data/extracted/*")

def infer_model_name(weight_path: Path) -> str:
    # ambil "yolov8n/yolov8s/..." dari path kalau ada
    for part in weight_path.parts:
        if part.startswith("yolov8"):
            return part
    # fallback
    return "model"

def main():
    chdir_project_root()
    print_header("COMPARE RESULTS (global best.pt search)")

    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    ensure_dir(reports_dir)

    data_yaml = find_data_yaml()

    # Cari best.pt di seluruh project (termasuk runs/detect/outputs/...)
    best_pts = []
    for p in Path(".").rglob("best.pt"):
        if ".venv" in p.parts:
            continue
        best_pts.append(p)

    if not best_pts:
        raise FileNotFoundError("Tidak menemukan best.pt di project ini.")

    # Urutkan terbaru dulu
    best_pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Ambil yang terbaru per model (yolov8n/s/m/l/x)
    picked = {}
    for p in best_pts:
        m = infer_model_name(p)
        if m not in picked:
            picked[m] = p

    print("Weights yang akan dipakai:")
    for m, p in picked.items():
        print(f" - {m}: {p}")

    rows = []
    for model_name, weight in picked.items():
        model = YOLO(str(weight))
        for split in ["val", "test"]:
            print(f"[EVAL] {model_name} split={split}")
            metrics = model.val(
                data=str(data_yaml),
                split=split,
                device=DEVICE,
                verbose=False,
                plots=False, 
            )

            row = {
                "model": model_name,
                "split": split,
                "weights": str(weight),
                "mAP50": getattr(metrics.box, "map50", None) if hasattr(metrics, "box") else None,
                "mAP50_95": getattr(metrics.box, "map", None) if hasattr(metrics, "box") else None,
                "precision": getattr(metrics.box, "mp", None) if hasattr(metrics, "box") else None,
                "recall": getattr(metrics.box, "mr", None) if hasattr(metrics, "box") else None,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    out_csv = reports_dir / "comparison.csv"
    out_md = reports_dir / "comparison.md"
    df.to_csv(out_csv, index=False)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Perbandingan Hasil YOLOv8 (n/s/m/l/x)\n\n")
        f.write("Metrik diambil dari evaluasi ulang `model.val()` berdasarkan `best.pt` terbaru yang ditemukan.\n\n")

        for split in ["val", "test"]:
            f.write(f"## Split: {split}\n\n")
            sdf = df[df["split"] == split].copy()
            # urutkan pakai mAP50_95 kalau ada
            if sdf["mAP50_95"].notna().any():
                sdf = sdf.sort_values("mAP50_95", ascending=False)
            elif sdf["mAP50"].notna().any():
                sdf = sdf.sort_values("mAP50", ascending=False)

            f.write(sdf[["model", "mAP50", "mAP50_95", "precision", "recall", "weights"]].to_markdown(index=False))
            f.write("\n\n")

        f.write("## Cara baca\n")
        f.write("- **mAP50**: lebih longgar (biasanya lebih tinggi)\n")
        f.write("- **mAP50_95**: lebih ketat (lebih penting untuk kualitas)\n")
        f.write("- **precision**: makin tinggi = FP makin sedikit\n")
        f.write("- **recall**: makin tinggi = miss makin sedikit\n")

    print(f"\n[SAVED] {out_csv}")
    print(f"[SAVED] {out_md}")
    print("Silakan buka: outputs/reports/comparison.md")

if __name__ == "__main__":
    main()
