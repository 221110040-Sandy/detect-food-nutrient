import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_loaders(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 2,
):
    # expected structure:
    # data/food_images/train/<class>/*.jpg
    # data/food_images/val/<class>/*.jpg

    train_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_tfm)
    val_ds   = datasets.ImageFolder(root=f"{data_dir}/val",   transform=val_tfm)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_ds.classes
