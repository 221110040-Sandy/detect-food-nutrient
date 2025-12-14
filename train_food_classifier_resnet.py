#!/usr/bin/env python3
import math, copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.cuda import amp
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.dataset import get_loaders
from src.model import build_model, save_checkpoint

DATA_DIR = "data/food_images"
SAVE_PATH = "food_classifier_resnet.pt"
BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
WARMUP_EPOCHS = 3

USE_MIXUP = True
MIXUP_ALPHA = 0.2
FREEZE_MIXUP_EPOCH = 15

ACC_STEPS = 1

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, 1.0, y
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    b = x.size(0)
    idx = torch.randperm(b, device=x.device)
    return lam * x + (1 - lam) * x[idx], y, lam, y[idx]

@torch.no_grad()
def validate(model, val_loader, device, label_smooth=0.0):
    model.eval()
    preds_all, labels_all = [], []
    loss_total = 0.0
    top5_correct = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        loss_total += loss.item() * labels.size(0)

        pred = out.argmax(dim=1)
        preds_all.extend(pred.cpu().tolist())
        labels_all.extend(labels.cpu().tolist())

        top5 = out.topk(5, dim=1).indices
        top5_correct += top5.eq(labels.unsqueeze(1)).any(dim=1).float().sum().item()

    n = max(len(labels_all), 1)
    return (loss_total / n), accuracy_score(labels_all, preds_all), (top5_correct / n)

def train():
    train_loader, val_loader, class_names = get_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    model = build_model(num_classes=len(class_names), arch="resnet50").to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR * 0.01)

    best_acc = 0.0
    best_state = None
    patience = 7
    bad_epochs = 0
    min_delta = 1e-4

    scaler = amp.GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        if epoch < WARMUP_EPOCHS:
            for g in optimizer.param_groups:
                g["lr"] = LR * float(epoch + 1) / float(WARMUP_EPOCHS)

        optimizer.zero_grad(set_to_none=True)
        use_mixup_now = USE_MIXUP and (epoch < FREEZE_MIXUP_EPOCH)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [ResNet50]")
        for step, (imgs, labels) in enumerate(pbar, start=1):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            if use_mixup_now:
                imgs, y_a, lam, y_b = mixup_data(imgs, labels, MIXUP_ALPHA)

            with amp.autocast(enabled=(DEVICE == "cuda")):
                out = model(imgs)
                if use_mixup_now:
                    loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
                else:
                    loss = criterion(out, labels)
                loss = loss / ACC_STEPS

            scaler.scale(loss).backward()

            if step % ACC_STEPS == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += (loss.item() * ACC_STEPS) * imgs.size(0)
            pbar.set_postfix({"loss": f"{loss.item() * ACC_STEPS:.4f}"})

        if epoch >= WARMUP_EPOCHS:
            scheduler.step()

        train_loss = running_loss / max(len(train_loader.dataset), 1)

        val_ls = LABEL_SMOOTH if epoch < FREEZE_MIXUP_EPOCH else max(0.0, LABEL_SMOOTH * 0.5)
        val_loss, val_top1, val_top5 = validate(model, val_loader, DEVICE, label_smooth=val_ls)

        print(f"[Epoch {epoch+1:02d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_top1={val_top1:.4f} | val_top5={val_top5:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | mixup={'on' if use_mixup_now else 'off'}")

        if (val_top1 - best_acc) > min_delta:
            best_acc = val_top1
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            print(f"  ✓ New best! Saved checkpoint (val_top1={best_acc:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[EarlyStop] No improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_checkpoint(SAVE_PATH, model, class_names, arch="resnet50")
    print(f"\n[✓] Training complete! Best val_top1={best_acc:.4f}")
    print(f"Model saved to {SAVE_PATH}")
    print("Classes:", class_names)

if __name__ == "__main__":
    train()
