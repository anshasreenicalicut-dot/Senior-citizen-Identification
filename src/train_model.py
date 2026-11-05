
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.model import AgeGenderCNN


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "age_gender_model.pth"
HISTORY_PATH = MODEL_DIR / "training_history.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

SENIOR_AGE = 60
EARLY_STOPPING_PATIENCE = 5

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class UTKFaceDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        required = {"image_path", "age", "gender"}

        if not required.issubset(self.data.columns):
            raise ValueError(
                f"{csv_path} must contain columns: {required}"
            )

        self.data["age"] = pd.to_numeric(
            self.data["age"], errors="coerce"
        )
        self.data["gender"] = pd.to_numeric(
            self.data["gender"], errors="coerce"
        )

        self.data = self.data.dropna(
            subset=["image_path", "age", "gender"]
        ).reset_index(drop=True)

        if not self.data["gender"].isin([0, 1]).all():
            raise ValueError("Gender labels must be 0 or 1.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        age = torch.tensor(
            float(row["age"]),
            dtype=torch.float32
        )

        gender = torch.tensor(
            int(row["gender"]),
            dtype=torch.long
        )

        return image, age, gender


# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def calculate_metrics(model, loader, age_loss_fn, gender_loss_fn):

    model.eval()

    total_loss = 0.0

    actual_ages = []
    predicted_ages = []

    actual_genders = []
    predicted_genders = []

    with torch.no_grad():

        for images, ages, genders in loader:

            images = images.to(DEVICE)
            ages = ages.to(DEVICE)
            genders = genders.to(DEVICE)

            predicted_age, gender_logits = model(images)

            loss = (
                age_loss_fn(predicted_age, ages)
                + gender_loss_fn(gender_logits, genders)
            )

            total_loss += loss.item() * len(images)

            actual_ages.extend(ages.cpu().numpy().tolist())
            predicted_ages.extend(
                predicted_age.cpu().numpy().tolist()
            )

            actual_genders.extend(
                genders.cpu().numpy().tolist()
            )

            predicted_genders.extend(
                gender_logits.argmax(dim=1).cpu().numpy().tolist()
            )

    actual_ages = np.array(actual_ages)
    predicted_ages = np.array(predicted_ages)

    actual_senior = (actual_ages > SENIOR_AGE).astype(int)
    predicted_senior = (
        np.rint(predicted_ages) > SENIOR_AGE
    ).astype(int)

    age_errors = np.abs(predicted_ages - actual_ages)

    senior_mask = actual_senior == 1
    non_senior_mask = actual_senior == 0

    senior_mae = (
        float(age_errors[senior_mask].mean())
        if senior_mask.any() else None
    )

    non_senior_mae = (
        float(age_errors[non_senior_mask].mean())
        if non_senior_mask.any() else None
    )

    return {
        "loss": float(total_loss / max(len(actual_ages), 1)),

        "age_mae": float(age_errors.mean()),

        "senior_age_mae": senior_mae,

        "non_senior_age_mae": non_senior_mae,

        "gender_accuracy": float(
            accuracy_score(actual_genders, predicted_genders)
        ),

        "senior_precision": float(
            precision_score(
                actual_senior,
                predicted_senior,
                zero_division=0
            )
        ),

        "senior_recall": float(
            recall_score(
                actual_senior,
                predicted_senior,
                zero_division=0
            )
        ),

        "senior_f1": float(
            f1_score(
                actual_senior,
                predicted_senior,
                zero_division=0
            )
        ),

        "senior_confusion_matrix": confusion_matrix(
            actual_senior,
            predicted_senior,
            labels=[0, 1]
        ).tolist(),

        "gender_confusion_matrix": confusion_matrix(
            actual_genders,
            predicted_genders,
            labels=[0, 1]
        ).tolist(),
    }


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def main():

    train_csv = DATA_DIR / "train.csv"
    val_csv = DATA_DIR / "val.csv"
    test_csv = DATA_DIR / "test.csv"

    for path in [train_csv, val_csv, test_csv]:

        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. "
                "Run: python -m src.prepare_dataset"
            )

    train_ds = UTKFaceDataset(train_csv, transform)
    val_ds = UTKFaceDataset(val_csv, transform)
    test_ds = UTKFaceDataset(test_csv, transform)

    # --------------------------------------------------
    # WEIGHTED SAMPLING
    # --------------------------------------------------

    train_ages = train_ds.data["age"].to_numpy()

    senior_labels = (train_ages > SENIOR_AGE).astype(int)

    class_counts = np.bincount(senior_labels, minlength=2)

    if np.any(class_counts == 0):
        raise ValueError(
            "Both senior and non-senior samples are required."
        )

    # Square-root weighting increases senior representation
    # without forcing a 50/50 training distribution.
    class_weights = 1.0 / np.sqrt(class_counts)

    sample_weights = class_weights[senior_labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double
        ),
        num_samples=len(train_ds),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = AgeGenderCNN().to(DEVICE)

    age_loss_fn = nn.SmoothL1Loss()
    gender_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0

    history = []

    print("\n--------------------------------")
    print("SENIOR CITIZEN IDENTIFICATION")
    print("--------------------------------")

    print(f"Device: {DEVICE}")
    print(f"Training images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Test images: {len(test_ds)}")

    print(f"Senior training images: {class_counts[1]}")
    print(f"Non-senior training images: {class_counts[0]}")

    # --------------------------------------------------
    # EPOCH LOOP
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for images, ages, genders in progress:

            images = images.to(DEVICE)
            ages = ages.to(DEVICE)
            genders = genders.to(DEVICE)

            optimizer.zero_grad()

            predicted_age, gender_logits = model(images)

            loss_age = age_loss_fn(predicted_age, ages)

            loss_gender = gender_loss_fn(
                gender_logits,
                genders
            )

            loss = loss_age + loss_gender

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_train_loss += loss.item() * len(images)

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = total_train_loss / len(train_ds)

        val_metrics = calculate_metrics(
            model,
            val_loader,
            age_loss_fn,
            gender_loss_fn
        )

        scheduler.step(val_metrics["loss"])

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            }
        }

        history.append(epoch_metrics)

        print("\nEpoch results:")
        print(json.dumps(epoch_metrics, indent=2))

        # --------------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------------

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "normalization_mean": [0.5, 0.5, 0.5],
                    "normalization_std": [0.5, 0.5, 0.5],
                    "gender_classes": ["Male", "Female"],
                    "senior_age_threshold": SENIOR_AGE,
                },
                MODEL_PATH
            )

            print(f"Best model saved: {MODEL_PATH}")

        else:

            patience_counter += 1

            print(
                f"No validation improvement: "
                f"{patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print("\nEarly stopping activated.")
            break

    # --------------------------------------------------
    # TEST BEST CHECKPOINT
    # --------------------------------------------------

    if not MODEL_PATH.exists():
        raise RuntimeError("Training did not produce a model checkpoint.")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = calculate_metrics(
        model,
        test_loader,
        age_loss_fn,
        gender_loss_fn
    )

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:

        json.dump(
            {
                "epochs": history,
                "test_metrics": test_metrics,
            },
            file,
            indent=2
        )

    # --------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------

    print("\n--------------------------------")
    print("FINAL TEST RESULTS")
    print("--------------------------------")

    print(f"Age MAE: {test_metrics['age_mae']:.2f} years")

    print(
        "Senior age MAE: "
        f"{test_metrics['senior_age_mae']:.2f} years"
        if test_metrics["senior_age_mae"] is not None
        else "Senior age MAE: N/A"
    )

    print(
        "Non-senior age MAE: "
        f"{test_metrics['non_senior_age_mae']:.2f} years"
        if test_metrics["non_senior_age_mae"] is not None
        else "Non-senior age MAE: N/A"
    )

    print(
        f"Gender accuracy: "
        f"{test_metrics['gender_accuracy'] * 100:.2f}%"
    )

    print(
        f"Senior precision: "
        f"{test_metrics['senior_precision'] * 100:.2f}%"
    )

    print(
        f"Senior recall: "
        f"{test_metrics['senior_recall'] * 100:.2f}%"
    )

    print(
        f"Senior F1-score: "
        f"{test_metrics['senior_f1'] * 100:.2f}%"
    )

    print("\nSenior confusion matrix:")
    print(test_metrics["senior_confusion_matrix"])

    print("\nGender confusion matrix:")
    print(test_metrics["gender_confusion_matrix"])

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Training history saved: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
