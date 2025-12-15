import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision
import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm


# -----------------------------
# CONFIG
# -----------------------------
ROOT_PATH = Path("rsna-pneumonia-detection-challenge")
DATA_PATH = Path("data")
TRAIN_SIZE = 24000
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 35
LR = 1e-4
MEAN = 0.49
STD = 0.248


# -----------------------------
# DATA PREPROCESSING
# -----------------------------
def preprocess_data():
    labels = pd.read_csv(ROOT_PATH / "stage_2_train_labels.csv")
    labels = labels.drop_duplicates("patientId")

    for split in ["train", "val"]:
        for cls in ["0", "1"]:
            (DATA_PATH / split / cls).mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(labels.iterrows(), total=len(labels)):
        patient_id = row.patientId
        label = row.Target

        dcm_path = ROOT_PATH / "stage_2_train_images" / f"{patient_id}.dcm"
        image = pydicom.dcmread(dcm_path).pixel_array.astype(np.float32)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE)) / 255.0

        split = "train" if idx < TRAIN_SIZE else "val"
        save_path = DATA_PATH / split / str(label) / f"{patient_id}.npy"
        np.save(save_path, image)


# -----------------------------
# DATASET LOADER
# -----------------------------
def load_npy(path):
    return np.load(path).astype(np.float32)


train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomAffine(degrees=5, translate=(0, 0.05), scale=(0.9, 1.1)),
    transforms.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.35, 1.0))
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# -----------------------------
# MODEL
# -----------------------------
class PneumoniaModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = torchvision.models.resnet18(weights=None)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.model.fc = torch.nn.Linear(512, 1)

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        y = y.float()

        logits = self(x).squeeze(1)
        loss = self.loss_fn(logits, y)

        self.train_acc(logits, y.int())
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss

    def validation_step(self, batch, _):
        x, y = batch
        y = y.float()

        logits = self(x).squeeze(1)
        loss = self.loss_fn(logits, y)

        self.val_acc(logits, y.int())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def on_validation_epoch_end(self):
        self.val_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


# -----------------------------
# TRAINING
# -----------------------------
def train():
    train_dataset = torchvision.datasets.DatasetFolder(
        DATA_PATH / "train",
        loader=load_npy,
        extensions="npy",
        transform=train_transforms,
    )

    val_dataset = torchvision.datasets.DatasetFolder(
        DATA_PATH / "val",
        loader=load_npy,
        extensions="npy",
        transform=val_transforms,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = PneumoniaModel()

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=3
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=EPOCHS,
        callbacks=[checkpoint_cb],
        logger=TensorBoardLogger("logs")
    )

    trainer.fit(model, train_loader, val_loader)
    return model, val_loader


# -----------------------------
# EVALUATION
# -----------------------------
def evaluate(model, val_loader):
    preds, targets = [], []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(val_loader):
            logits = model(x).squeeze(1)
            pred = (torch.sigmoid(logits) > 0.5).int()
            preds.extend(pred.cpu())
            targets.extend(y.cpu())

    preds = torch.tensor(preds)
    targets = torch.tensor(targets)

    metrics = {
        "Accuracy": torchmetrics.Accuracy(task="binary"),
        "Precision": torchmetrics.Precision(task="binary"),
        "Recall": torchmetrics.Recall(task="binary"),
        "F1": torchmetrics.F1Score(task="binary"),
        "Confusion Matrix": torchmetrics.classification.BinaryConfusionMatrix()
    }

    for name, metric in metrics.items():
        print(f"{name}:")
        print(metric(preds, targets))
        print()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    if not DATA_PATH.exists():
        preprocess_data()

    model, val_loader = train()
    evaluate(model, val_loader)

