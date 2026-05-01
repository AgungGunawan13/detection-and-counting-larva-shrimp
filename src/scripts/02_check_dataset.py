from pathlib import Path
import yaml
from utils import find_yaml, print_header

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXTRACTED_DIR = BASE_DIR / "data/extracted"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def count_images(images_dir: Path):
    return len([p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

def count_labels(labels_dir: Path):
    return len(list(labels_dir.rglob("*.txt")))

def main():
    print_header("CHECK DATASET STRUCTURE")

    yaml_path = None
    for d in EXTRACTED_DIR.iterdir():
        if d.is_dir():
            y = find_yaml(d)
            if y:
                yaml_path = y
                break

    if not yaml_path:
        raise FileNotFoundError("Tidak menemukan data.yaml di dalam data/extracted/*")

    print(f"[FOUND] YAML: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    print("\n[INFO dari YAML]")
    for k in ["path", "train", "val", "test", "nc", "names"]:
        if k in data:
            print(f" - {k}: {data[k]}")

    base = yaml_path.parent
    if "path" in data:
        base = Path(data["path"])
        if not base.is_absolute():
            base = (yaml_path.parent / base).resolve()

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (base / p).resolve()

    for split in ["train", "val", "test"]:
        if split not in data:
            print(f"\n[WARN] '{split}' tidak ada di YAML.")
            continue

        images_dir = resolve(data[split])

        labels_dir = Path(str(images_dir).replace("images", "labels"))
        if not labels_dir.exists():
            candidates = list(images_dir.parent.parent.rglob("labels"))
            if candidates:
                labels_dir = candidates[0]

        print(f"\n[{split.upper()}]")
        print(f" images_dir: {images_dir}")
        print(f" labels_dir: {labels_dir}")
        print(f" images: {count_images(images_dir)}")
        print(f" labels: {count_labels(labels_dir) if labels_dir.exists() else 0} (labels_dir exists: {labels_dir.exists()})")

    print("\nSelesai")

if __name__ == "__main__":
    main()
