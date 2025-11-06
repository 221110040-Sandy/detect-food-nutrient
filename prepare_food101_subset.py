#!/usr/bin/env python3
import os
import tarfile
import urllib.request
import shutil
from pathlib import Path

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"
TAR_PATH = RAW_DIR / "food-101.tar.gz"

EXTRACT_CANDIDATE = RAW_DIR / "food-101"

SUBSET_CLASSES = [
    "apple_pie",
    "baby_back_ribs",
    "bibimbap",
    "caesar_salad",
    "carrot_cake",
    "cheesecake",
    "chicken_curry",
    "chicken_wings",
    "chocolate_cake",
    "clam_chowder",
    "club_sandwich",
    "donuts",
    "dumplings",
    "eggs_benedict",
    "french_fries",
    "fried_rice",
    "frozen_yogurt",
    "grilled_cheese_sandwich",
    "grilled_salmon",
    "hamburger",
    "hot_dog",
    "ice_cream",
    "lasagna",
    "macaroni_and_cheese",
    "nachos",
    "omelette",
    "onion_rings",
    "pad_thai",
    "paella",
    "pancakes",
    "panna_cotta",
    "pho",
    "pizza",
    "pork_chop",
    "ramen",
    "ravioli",
    "red_velvet_cake",
    "risotto",
    "samosa",
    "sashimi",
    "spaghetti_bolognese",
    "spaghetti_carbonara",
    "spring_rolls",
    "steak",
    "strawberry_shortcake",
    "sushi",
    "tacos",
    "takoyaki",
    "tiramisu",
    "waffles",
]

TARGET_DIR = DATA_ROOT / "food_images"
TRAIN_OUT = TARGET_DIR / "train"
VAL_OUT   = TARGET_DIR / "val"
TEST_OUT  = TARGET_DIR / "test"

MAX_TRAIN_PER_CLASS = 500
MAX_VAL_PER_CLASS   = 100
MAX_TEST_PER_CLASS  = 150 

AUTO_CLEAN = False
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"


def debug_list_dir(p: Path, depth=1):
    if not p.exists():
        print(f"[debug] {p} (MISSING)")
        return
    print(f"[debug] listing {p}:")
    for item in p.iterdir():
        kind = "DIR " if item.is_dir() else "FILE"
        print(" " * depth + f"- {kind} {item.name}")


def download_food101():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TAR_PATH.exists():
        print("[i] food-101.tar.gz already exists, skip download")
        return
    print(f"[+] Downloading Food-101 from {FOOD101_URL} ...")
    urllib.request.urlretrieve(FOOD101_URL, TAR_PATH)
    print("[+] Download complete")


def extract_food101() -> Path:
    if EXTRACT_CANDIDATE.exists():
        for cand in [EXTRACT_CANDIDATE, EXTRACT_CANDIDATE / "food-101"]:
            if (cand / "meta").exists() and (cand / "images").exists():
                print(f"[i] Found extracted dataset at {cand}")
                return cand

    print("[+] Extracting tar.gz ...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        tar.extractall(path=RAW_DIR)
    print("[+] Extract done")

    for cand in [EXTRACT_CANDIDATE, EXTRACT_CANDIDATE / "food-101"]:
        if (cand / "meta").exists() and (cand / "images").exists():
            print(f"[i] Found extracted dataset at {cand}")
            return cand

    print("[!] Could not find meta/ and images/ after extraction.")
    debug_list_dir(RAW_DIR)
    debug_list_dir(EXTRACT_CANDIDATE)
    debug_list_dir(EXTRACT_CANDIDATE / "food-101")
    raise FileNotFoundError("Dataset extracted but structure not recognized.")


def load_split_list(split_txt_path: Path) -> dict[str, list[str]]:
    split_map: dict[str, list[str]] = {}
    with open(split_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cls, img_stub = line.split("/", 1)
            img_name = img_stub + ".jpg"
            split_map.setdefault(cls, []).append(img_name)
    return split_map


def copy_subset_images(dataset_root: Path,
                       train_map: dict[str, list[str]],
                       test_map: dict[str, list[str]]):
    img_dir = dataset_root / "images"

  
    for base in [TRAIN_OUT, VAL_OUT, TEST_OUT]:
        base.mkdir(parents=True, exist_ok=True)

    for cls in SUBSET_CLASSES:
        if cls not in train_map or cls not in test_map:
            print(f"[!] WARNING: class '{cls}' not found in Food-101 splits. Skipping.")
            continue

        class_train_dir = TRAIN_OUT / cls
        class_val_dir   = VAL_OUT / cls
        class_test_dir  = TEST_OUT / cls
        class_train_dir.mkdir(parents=True, exist_ok=True)
        class_val_dir.mkdir(parents=True, exist_ok=True)
        class_test_dir.mkdir(parents=True, exist_ok=True)

        train_list = train_map[cls]
        if MAX_TRAIN_PER_CLASS is not None:
            train_list = train_list[:MAX_TRAIN_PER_CLASS]

        copied_train = 0
        for img_name in train_list:
            src = img_dir / cls / img_name
            dst = class_train_dir / img_name
            if not dst.exists() and src.exists():
                shutil.copy2(src, dst)
                copied_train += 1


        remaining_train = train_map[cls][MAX_TRAIN_PER_CLASS:]
        val_list = remaining_train[:MAX_VAL_PER_CLASS] if remaining_train else []
        
        copied_val = 0
        for img_name in val_list:
            src = img_dir / cls / img_name
            dst = class_val_dir / img_name
            if not dst.exists() and src.exists():
                shutil.copy2(src, dst)
                copied_val += 1

        test_list = test_map[cls]
        if MAX_TEST_PER_CLASS is not None:
            test_list = test_list[:MAX_TEST_PER_CLASS]

        copied_test = 0
        for img_name in test_list:
            src = img_dir / cls / img_name
            dst = class_test_dir / img_name
            if not dst.exists() and src.exists():
                shutil.copy2(src, dst)
                copied_test += 1

        print(f"[+] {cls}: {copied_train} train, {copied_val} val, {copied_test} test")


    print("Sekarang bisa training dengan DATA_DIR = 'data/food_images'")


def clean_big_files(dataset_root: Path):
    if AUTO_CLEAN:
        print("[i] AUTO_CLEAN enabled. Removing original Food-101 data...")
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
        if TAR_PATH.exists():
            TAR_PATH.unlink()
        print("[✓] Cleanup done. You only keep subset data.")


def main():
    download_food101()
    dataset_root = extract_food101()

    train_txt_path = dataset_root / "meta" / "train.txt"
    test_txt_path  = dataset_root / "meta" / "test.txt"

    if not train_txt_path.exists() or not test_txt_path.exists():
        print("[!] train.txt/test.txt tidak ketemu di:", dataset_root / "meta")
        debug_list_dir(dataset_root / "meta")
        raise FileNotFoundError("train.txt/test.txt missing.")

    train_map = load_split_list(train_txt_path)
    test_map  = load_split_list(test_txt_path)

    copy_subset_images(dataset_root, train_map, test_map)
    clean_big_files(dataset_root)


if __name__ == "__main__":
    main()
