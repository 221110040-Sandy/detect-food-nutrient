#!/usr/bin/env python3
"""
Final Test Evaluation Script (Notebook/Colab-safe)
Evaluasi model pada test set yang tidak pernah dilihat saat training.

Fitur:
- Kebal argumen asing dari Jupyter/Colab (parse_known_args)
- Menangani mismatch urutan kelas (ImageFolder vs checkpoint)
- Menghitung Top-1, Top-5, Test Loss (CE)
- Classification report (macro/weighted F1)
- Simpan hasil ke JSON & TXT, serta plot confusion matrix
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns  # kalau mau tanpa seaborn, bisa diganti matplotlib-only

from src.model import build_model, load_checkpoint


# ----------------------------
# Utils & Config
# ----------------------------

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_transforms(eval_size: int = 260, resize_scale: float = 1.15):
    """Transform eval tanpa augmentasi (samakan dengan setup validasi)."""
    return transforms.Compose([
        transforms.Resize(int(round(eval_size * resize_scale))),
        transforms.CenterCrop(eval_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_loader(test_dir: str,
                    batch_size: int = 32,
                    num_workers: int = 2,
                    eval_size: int = 260,
                    resize_scale: float = 1.15):
    """Buat DataLoader untuk test set (tanpa augmentasi)."""
    transform = build_transforms(eval_size, resize_scale)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, test_dataset.classes, test_dataset.class_to_idx


def make_index_mapping(test_class_to_idx, saved_class_names):
    """
    Buat mapping index: (idx dari ImageFolder/test) -> (idx urutan kelas checkpoint).
    Akan raise error jika ada kelas yang tidak ditemukan pada checkpoint.
    """
    idx_to_class = {v: k for k, v in test_class_to_idx.items()}
    mapping = {}
    for cur_idx, cls_name in idx_to_class.items():
        try:
            target_idx = saved_class_names.index(cls_name)
        except ValueError:
            raise RuntimeError(
                f"Class '{cls_name}' dari test set tidak ada di checkpoint!"
            )
        mapping[cur_idx] = target_idx
    return mapping


# ----------------------------
# Evaluation
# ----------------------------

@torch.no_grad()
def evaluate_model(model, test_loader, device, idx_map=None):
    """
    Evaluasi komprehensif.
    idx_map: dict[int_current -> int_saved] untuk menyamakan index label dengan urutan kelas di model.
    """
    model.eval()
    ce = nn.CrossEntropyLoss(reduction='sum')

    all_preds, all_labels_mapped, all_probs = [], [], []
    total_loss, n = 0.0, 0

    for images, labels in test_loader:
        # Map label (ImageFolder) -> urutan kelas di checkpoint
        if idx_map is not None:
            labels_m = torch.as_tensor([idx_map[int(l)] for l in labels.tolist()])
        else:
            labels_m = labels

        images = images.to(device)
        labels_m = labels_m.to(device)

        logits = model(images)
        total_loss += ce(logits, labels_m).item()
        n += labels_m.size(0)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels_mapped.append(labels_m.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels_mapped, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    # Top-1
    top1_acc = accuracy_score(all_labels, all_preds)

    # Top-5 (descending argsort biar eksplisit)
    top5_idx = np.argsort(-all_probs, axis=1)[:, :5]
    top5_acc = (top5_idx == all_labels.reshape(-1, 1)).any(axis=1).mean()

    test_loss = float(total_loss / n)

    return {
        "loss": test_loss,
        "accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "predictions": all_preds,      # mapped index (sesuai urutan kelas checkpoint)
        "labels": all_labels,          # mapped index (sesuai urutan kelas checkpoint)
        "probabilities": all_probs,    # probabilitas sesuai urutan kelas checkpoint
    }


def plot_confusion_matrix(labels, predictions, class_names, save_path="test_confusion_matrix.png"):
    """Plot confusion matrix (count)."""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to", save_path)
    plt.close()


def save_json_results(results, class_names, save_path="test_results.json"):
    """Simpan hasil evaluasi ke JSON (serializable)."""
    safe = {
        "loss": float(results["loss"]),
        "accuracy": float(results["accuracy"]),
        "top5_accuracy": float(results["top5_accuracy"]),
        "class_names": list(class_names),
        "predictions": list(map(int, results["predictions"])),
        "labels": list(map(int, results["labels"])),
        "probabilities": np.asarray(results["probabilities"]).tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(safe, f, indent=2)
    print("Results saved to", save_path)


def save_txt_report(labels_mapped, preds_mapped, class_names, acc, top5, loss,
                    txt_path="test_classification_report.txt"):
    """Simpan classification report + ringkasan metrik ke TXT."""
    report_str = classification_report(
        labels_mapped,
        preds_mapped,
        target_names=class_names,
        digits=4
    )
    report_dict = classification_report(
        labels_mapped,
        preds_mapped,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    macro_f1 = report_dict["macro avg"]["f1-score"]
    weighted_f1 = report_dict["weighted avg"]["f1-score"]

    with open(txt_path, "w") as f:
        f.write("Final Test Set Classification Report\n")
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Top-1 Accuracy: {acc*100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5*100:.2f}%\n")
        f.write(f"Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}\n")
        f.write("=" * 80 + "\n")
        f.write(report_str)

    print("Classification report saved to", txt_path)
    print(f"Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")


# ----------------------------
# Main
# ----------------------------

def parse_args_colab_safe():
    """Parse args tapi abaikan argumen asing dari Jupyter/Colab."""
    parser = argparse.ArgumentParser(description="Final Test Evaluation")
    parser.add_argument("--model", type=str, default="food_classifier_dataminds.pt",
                        help="Path ke checkpoint .pt")
    parser.add_argument("--test_dir", type=str, default="data/food_images/test",
                        help="Folder test (ImageFolder layout)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_size", type=int, default=260,
                        help="Ukuran CenterCrop untuk evaluasi")
    parser.add_argument("--resize_scale", type=float, default=1.15,
                        help="Skala Resize sebelum CenterCrop (eval)")

    # >>> Ini kunci agar kebal -f ...json dari Jupyter/Colab
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args_colab_safe()

    MODEL_PATH = args.model
    TEST_DATA_DIR = args.test_dir
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    EVAL_SIZE = args.eval_size
    RESIZE_SCALE = args.resize_scale

    print("DataMinds Food Classifier - Final Test Evaluation")
    print("=" * 70)

    # Cek path
    if not Path(TEST_DATA_DIR).exists():
        print(f"Test directory not found: {TEST_DATA_DIR}")
        print("Pastikan kamu sudah membuat split test yang benar.")
        return
    if not Path(MODEL_PATH).exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Latih model dulu dan simpan best checkpoint.")
        return

    device = get_device()
    print(f"Test data directory: {TEST_DATA_DIR}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Device: {device}")

    # Data
    test_loader, test_class_names, test_class_to_idx = get_test_loader(
        TEST_DATA_DIR, BATCH_SIZE, NUM_WORKERS, EVAL_SIZE, RESIZE_SCALE
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(test_class_names)}")

    # Model
    print("\nLoading trained model...")
    try:
        state_dict, saved_class_names = load_checkpoint(MODEL_PATH, device)
        model = build_model(num_classes=len(saved_class_names)).to(device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Cek & mapping kelas
    if saved_class_names != test_class_names:
        print("Warning: Class names don't match (test vs checkpoint). Remapping labels...")
        idx_map = make_index_mapping(test_class_to_idx, saved_class_names)
        class_names_eval = saved_class_names  # evaluasi pakai urutan checkpoint
    else:
        print("Class names match.")
        idx_map = None
        class_names_eval = saved_class_names

    # Evaluasi
    print("\nStarting final evaluation...")
    results = evaluate_model(model, test_loader, device, idx_map=idx_map)

    # Ringkasan metrik
    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Top-1 Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']*100:.2f}%")
    print(f"Test Samples: {len(results['labels'])}")

    if results["accuracy"] >= 0.90:
        print("\nEXCELLENT! 90%+ accuracy achieved!")
    elif results["accuracy"] >= 0.85:
        print("\nGREAT! 85%+ accuracy achieved!")
    elif results["accuracy"] >= 0.80:
        print("\nGOOD! 80%+ accuracy achieved!")
    else:
        print("\nPertimbangkan tuning tambahan (augmentasi, LR, epochs, unfreeze deeper layers).")

    # Classification report + F1
    print("\nGenerating detailed analysis...")
    save_txt_report(
        labels_mapped=results["labels"],
        preds_mapped=results["predictions"],
        class_names=class_names_eval,
        acc=results["accuracy"],
        top5=results["top5_accuracy"],
        loss=results["loss"],
        txt_path="test_classification_report.txt"
    )

    # JSON
    save_json_results(results, class_names_eval, save_path="test_results.json")

    # Confusion Matrix
    plot_confusion_matrix(
        labels=results["labels"],
        predictions=results["predictions"],
        class_names=class_names_eval,
        save_path="test_confusion_matrix.png"
    )

    print("\n" + "=" * 70)
    print("Final evaluation completed!")
    print("Generated files:")
    print("  test_confusion_matrix.png")
    print("  test_classification_report.txt")
    print("  test_results.json")


if __name__ == "__main__":
    main()
