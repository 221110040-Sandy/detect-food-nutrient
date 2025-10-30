#!/usr/bin/env python3
import tarfile
import urllib.request
import shutil
from pathlib import Path

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"
TAR_PATH = RAW_DIR / "food-101.tar.gz"
EXTRACT_DIR = RAW_DIR / "food-101"
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

OUT_DIR = DATA_ROOT / "food_images"
TRAIN_OUT = OUT_DIR / "train"
VAL_OUT = OUT_DIR / "val"

def download_food101():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TAR_PATH.exists():
        print("[i] food-101.tar.gz already exists, skip download")
        return
    print(f"[+] Downloading Food-101 from {FOOD101_URL} ...")
    urllib.request.urlretrieve(FOOD101_URL, TAR_PATH)
    print("[+] Download complete:", TAR_PATH)

def extract_food101() -> Path:
    if (EXTRACT_DIR / "meta").exists() and (EXTRACT_DIR / "images").exists():
        print(f"[i] Found extracted dataset at {EXTRACT_DIR}")
        return EXTRACT_DIR
    print("[+] Extracting tar.gz ...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(path=RAW_DIR)
    print("[+] Extract done")
    for cand in [EXTRACT_DIR, EXTRACT_DIR / "food-101"]:
        if (cand / "meta").exists() and (cand / "images").exists():
            print(f"[i] Dataset ready at: {cand}")
            return cand
    raise FileNotFoundError("Extraction finished but meta/ or images/ not found.")

def load_split_list(p: Path):
    d = {}
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, img_stub = line.split("/", 1)
            d.setdefault(cls, []).append(img_stub + ".jpg")
    return d

def mirror_split(root: Path, split_map: dict, dest_base: Path):
    img_dir = root / "images"
    for cls, files in split_map.items():
        (dest_base / cls).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = img_dir / cls / name
            dst = dest_base / cls / name
            if not dst.exists():
                if not src.exists():
                    continue
                shutil.copy2(src, dst)

def main():
    download_food101()
    root = extract_food101()
    meta = root / "meta"
    train_map = load_split_list(meta / "train.txt")
    test_map  = load_split_list(meta / "test.txt")
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    VAL_OUT.mkdir(parents=True, exist_ok=True)
    print("[+] Mirroring train split...")
    mirror_split(root, train_map, TRAIN_OUT)
    print("[+] Mirroring val split...")
    mirror_split(root, test_map, VAL_OUT)
    print("[✓] Done. Train/val at:", OUT_DIR)

if __name__ == "__main__":
    main()
