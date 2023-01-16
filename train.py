from tqdm import tqdm


def train_step(model, dataloader, criterion, optimizer, metric, device="cpu"):
    model.train()
    for image, label, _ in tqdm(dataloader, total=len(dataloader)):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        pred = model(image)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        metric(pred, label)

    acc = metric.compute()

    print(f"train_loss: {loss}, train_acc: {acc}")
    metric.reset()


def val_step(model, dataloader, criterion, metric, device="cpu"):
    model.eval()
    for image, label, _ in tqdm(dataloader, total=len(dataloader)):
        image = image.to(device)
        label = label.to(device)

        pred = model(image)

        loss = criterion(pred, label)

        metric(pred, label)

    acc = metric.compute()
    print(f"val_loss: {loss}, val_acc: {acc}")
    metric.reset()


if __name__ == "__main__":
    import argparse
    from torch.optim import AdamW
    import torchmetrics

    from model import CatsVsDogsMobileNet
    from dataset import CatVsDogsDataset

    from random import shuffle
    from glob import glob

    import torch.nn as nn
    from torch.utils.data import    DataLoader

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    classes = ["Cat", "Dog"]

    parser = argparse.ArgumentParser(
        prog="Dogs vs Cats Trainer", description="Dogs vs Cats Trainer"
    )
    parser.add_argument("images_glob", type=str)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model = CatsVsDogsMobileNet().to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(args.device)

    images = glob(args.images_glob)
    shuffle(images)

    train_images = images[:20000]
    val_images = images[20000:]

    train_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2(),
        ]
    )

    train_dataset = CatVsDogsDataset(
        train_images, classes=classes, transform=train_transform
    )
    val_dataset = CatVsDogsDataset(val_images, classes=classes, transform=val_transform)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        train_step(model, train_dataloader, criterion, optimizer, metric, device=args.device)
        val_step(model, val_dataloader, criterion, metric, device=args.device)