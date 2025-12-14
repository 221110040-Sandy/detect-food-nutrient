#!/usr/bin/env python3
"""
FINAL TEST EVALUATION (DUAL MODEL)
- Evaluasi EfficientNet & ResNet50 pada test set
- Top-1, Top-5, Loss
- Macro & Weighted F1
- Confusion Matrix
- Jumlah parameter (analisis arsitektur)
- Output bukti per model (JSON, TXT, PNG)
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from src.model import build_model, load_checkpoint


# ======================================================
# CONFIG
# ======================================================
TEST_DIR = "data/food_images/test"
CHECKPOINTS = {
    "efficientnet": "food_classifier_dataminds.pt",
    "resnet50": "food_classifier_resnet.pt",
}

BATCH_SIZE = 32
NUM_WORKERS = 2
EVAL_SIZE = 260
RESIZE_SCALE = 1.15


# ======================================================
# UTILS
# ======================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_transforms():
    return transforms.Compose([
        transforms.Resize(int(round(EVAL_SIZE * RESIZE_SCALE))),
        transforms.CenterCrop(EVAL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_loader():
    dataset = datasets.ImageFolder(TEST_DIR, transform=build_transforms())
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    return loader, dataset.classes, dataset.class_to_idx


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_index_mapping(test_class_to_idx, saved_class_names):
    idx_to_class = {v: k for k, v in test_class_to_idx.items()}
    mapping = {}
    for cur_idx, cls_name in idx_to_class.items():
        mapping[cur_idx] = saved_class_names.index(cls_name)
    return mapping


# ======================================================
# EVALUATION
# ======================================================
@torch.no_grad()
def evaluate(model, loader, device, idx_map=None):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")

    all_preds, all_labels, all_probs = [], [], []
    total_loss, n = 0.0, 0
    infer_times = []

    for imgs, labels in loader:
        if idx_map:
            labels = torch.tensor([idx_map[int(l)] for l in labels])

        imgs, labels = imgs.to(device), labels.to(device)

        start = time.time()
        logits = model(imgs)
        if device == "cuda":
            torch.cuda.synchronize()
        infer_times.append(time.time() - start)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += ce(logits, labels).item()
        n += labels.size(0)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    top1 = accuracy_score(all_labels, all_preds)
    top5 = (np.argsort(-all_probs, axis=1)[:, :5] ==
            all_labels.reshape(-1, 1)).any(axis=1).mean()

    return {
        "loss": total_loss / n,
        "top1": top1,
        "top5": top5,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
        "avg_infer_time": float(np.mean(infer_times))
    }


def save_confusion_matrix(labels, preds, class_names, name):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix – {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = f"confusion_matrix_{name}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


def save_reports(results, class_names, name):
    report_txt = classification_report(
        results["labels"],
        results["preds"],
        target_names=class_names,
        digits=4
    )

    report_dict = classification_report(
        results["labels"],
        results["preds"],
        target_names=class_names,
        output_dict=True
    )

    with open(f"report_{name}.txt", "w") as f:
        f.write(f"MODEL: {name}\n")
        f.write(f"Loss: {results['loss']:.4f}\n")
        f.write(f"Top-1: {results['top1']*100:.2f}%\n")
        f.write(f"Top-5: {results['top5']*100:.2f}%\n")
        f.write(f"Macro-F1: {report_dict['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted-F1: {report_dict['weighted avg']['f1-score']:.4f}\n")
        f.write("=" * 80 + "\n")
        f.write(report_txt)

    with open(f"results_{name}.json", "w") as f:
        json.dump({
            "model": name,
            "loss": results["loss"],
            "top1": results["top1"],
            "top5": results["top5"],
            "avg_infer_time": results["avg_infer_time"],
        }, f, indent=2)

    print("Saved report & JSON for", name)


# ======================================================
# MAIN
# ======================================================
def main():
    device = get_device()
    print("Device:", device)

    loader, test_classes, test_class_to_idx = get_test_loader()

    for name, ckpt_path in CHECKPOINTS.items():
        print("\n" + "=" * 70)
        print("Evaluating:", name)

        state_dict, saved_classes, arch = load_checkpoint(ckpt_path, device)
        model = build_model(num_classes=len(saved_classes), backbone=arch).to(device)
        model.load_state_dict(state_dict)

        idx_map = None
        if saved_classes != test_classes:
            idx_map = make_index_mapping(test_class_to_idx, saved_classes)

        params = count_parameters(model)
        print(f"Trainable parameters: {params:,}")

        results = evaluate(model, loader, device, idx_map)

        print(f"Loss: {results['loss']:.4f}")
        print(f"Top-1: {results['top1']*100:.2f}%")
        print(f"Top-5: {results['top5']*100:.2f}%")
        print(f"Avg inference time/batch: {results['avg_infer_time']:.4f}s")

        save_confusion_matrix(results["labels"], results["preds"], saved_classes, name)
        save_reports(results, saved_classes, name)

    print("\n✅ FINAL EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
