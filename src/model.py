import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int, arch: str = "efficientnet_b2"):
    """Build model based on architecture type"""
    if arch == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = models.efficientnet_b2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes),
        )
        return model
    else:
        raise ValueError(f"Unknown architecture: {arch}")

def save_checkpoint(path: str, model: nn.Module, class_names, arch: str = "efficientnet_b2"):
    ckpt = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "arch": arch,
    }
    torch.save(ckpt, path)

def load_checkpoint(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    arch = ckpt.get("arch", "efficientnet_b2")  # backward compatibility
    return ckpt["state_dict"], ckpt["class_names"], arch
