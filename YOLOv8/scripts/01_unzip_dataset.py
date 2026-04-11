from pathlib import Path
import zipfile
from utils import ensure_dir, print_header

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/extracted")

def unzip_one(zip_path: Path, dest_dir: Path):
    ensure_dir(dest_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

def main():
    print_header("UNZIP DATASET")
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"Tidak ada file .zip di {RAW_DIR.resolve()}")

    for z in zips:
        target = OUT_DIR / z.stem
        print(f"[OK] Unzip: {z.name} -> {target}")
        unzip_one(z, target)

    print("\nSelesai unzip.")

if __name__ == "__main__":
    main()
