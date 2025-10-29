import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.dataset import get_loaders
from src.model import build_model, save_checkpoint

DATA_DIR = "data/food_images"
SAVE_PATH = "food_classifier_dataminds.pt"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def validate(model, val_loader, device):
    model.eval()
    preds_all, labels_all = [], []
    loss_total = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            loss_total += loss.item() * labels.size(0)
            pred = out.argmax(dim=1)
            preds_all.extend(pred.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    avg_loss = loss_total / max(len(labels_all), 1)
    acc = accuracy_score(labels_all, preds_all)
    return avg_loss, acc

def train():
    train_loader, val_loader, class_names = get_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    model = build_model(num_classes=len(class_names)).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_loss, val_acc = validate(model, val_loader, DEVICE)
        print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

    save_checkpoint(SAVE_PATH, model, class_names)
    print(f"Model saved to {SAVE_PATH}")
    print("Classes:", class_names)

if __name__ == "__main__":
    train()
