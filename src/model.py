import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def save_checkpoint(path: str, model: nn.Module, class_names):
    ckpt = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
    }
    torch.save(ckpt, path)

def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt["state_dict"], ckpt["class_names"]
