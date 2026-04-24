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
    for part in weight_path.parts:
        if part.startswith("yolov8"):
            return part
    return "model"

def main():
    chdir_project_root()
    print_header("EVALUASI DAN KOMPARASI MODEL (Gabungan Skrip 04 & 05)")
    print("Mencari bobot terbaik (best.pt) untuk dijalankan dalam tahap evaluasi...")

    # Folder tujuan (satu untuk gambar, satu untuk tabel)
    eval_dir = PROJECT_ROOT / "outputs" / "eval"
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    ensure_dir(eval_dir)
    ensure_dir(reports_dir)

    data_yaml = find_data_yaml()

    # Cari best.pt di seluruh project (kecuali .venv)
    best_pts = []
    for p in Path(".").rglob("best.pt"):
        if ".venv" in p.parts:
            continue
        best_pts.append(p)

    if not best_pts:
        raise FileNotFoundError("Tidak menemukan file best.pt di project ini. Pastikan training sudah selesai.")

    # Urutkan berdasarkan yang terupdate
    best_pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Ambil 1 yang terbaik/terbaru per varian model
    picked = {}
    for p in best_pts:
        m = infer_model_name(p)
        if m not in picked:
            picked[m] = p

    print("\nBobot AI yang siap diuji secara mandiri:")
    for m, p in picked.items():
        print(f" ► {m}: {p}")
    print("\n" + "="*50)

    rows = []
    # Jalankan evaluasi untuk masing-masing model
    for model_name, weight in picked.items():
        model = YOLO(str(weight))
        
        # Uji pada 2 pembagian data terpisah: "val" dan "test"
        for split in ["val", "test"]:
            print(f"\n[PROSES] Menjalankan ujian pada {model_name} (Data: {split}) ...")
            
            # FITUR GABUNGAN: 
            # plots=True (Fungsi Skrip 04: Mencetak Gambar Confusion Matrix)
            # object returns (Fungsi Skrip 05: Mengambil angka komparasi)
            metrics = model.val(
                data=str(data_yaml),
                split=split,
                device=DEVICE,
                verbose=False,
                plots=True, 
                project=str(eval_dir),
                name=f"{model_name}_{split}",
                exist_ok=True,
            )

            # Tarik keluar angka mAP dan lain-lain ke memori
            row = {
                "Model": model_name,
                "Data_Split": split,
                "mAP_50": getattr(metrics.box, "map50", None) if hasattr(metrics, "box") else None,
                "mAP_50_95": getattr(metrics.box, "map", None) if hasattr(metrics, "box") else None,
                "Precision": getattr(metrics.box, "mp", None) if hasattr(metrics, "box") else None,
                "Recall": getattr(metrics.box, "mr", None) if hasattr(metrics, "box") else None,
            }
            # Kita bulatkan angkanya ke 5 di belakang koma (jika ada nilainya)
            row = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}
            rows.append(row)

    # ==========================================
    # PEMBUATAN TABEL LAPORAN (FUNGSI KOMPARASI)
    # ==========================================
    if rows:
        df = pd.DataFrame(rows)
        out_csv = reports_dir / "comparison_gabungan.csv"
        out_excel = reports_dir / "comparison_gabungan.xlsx"
        out_md = reports_dir / "comparison_gabungan.md"
        
        # Simpan ke CSV
        df.to_csv(out_csv, index=False)
        
        # Simpan ke Excel (.xlsx) biar gampang dicopy ke doc skripsi
        try:
            df.to_excel(out_excel, index=False, engine='openpyxl')
        except Exception:
            pass # fallback aman jika openpyxl bermasalah
            
        print(f"\n\n[SELESAI] ")

    else:
        print("[-] Gagal mengekstrak komparasi.")

if __name__ == "__main__":
    main()
