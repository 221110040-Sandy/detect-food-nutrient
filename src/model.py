import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int):
    weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
    model = models.efficientnet_b2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model

def save_checkpoint(path: str, model: nn.Module, class_names):
    ckpt = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "arch": "efficientnet_b2",
    }
    torch.save(ckpt, path)

def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    return ckpt["state_dict"], ckpt["class_names"]
