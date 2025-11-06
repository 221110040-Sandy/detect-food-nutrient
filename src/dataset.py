import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode as I

def get_loaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 2,
    img_size: int = 260,
    jitter: float = 0.4,
):
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=I.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=0.05),
        transforms.RandomRotation(15, interpolation=I.BICUBIC),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])

    val_resize = int(round(img_size * 1.15))
    val_tfm = transforms.Compose([
        transforms.Resize(val_resize, interpolation=I.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_tfm)
    val_ds   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=val_tfm)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader, train_ds.classes
