from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.model import AgeGenderCNN


# ==================================================
# 1. PATHS AND CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "age_gender_model.pth"

YUNET_MODEL_PATH = (
    ROOT / "models" / "face_detection_yunet_2023mar.onnx"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==================================================
# 2. YUNET FACE DETECTOR
# ==================================================

class FaceLocator:
    """Detect multiple faces using OpenCV YuNet."""

    def __init__(self):

        if not YUNET_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {YUNET_MODEL_PATH}"
            )

        if not hasattr(cv2, "FaceDetectorYN_create"):
            raise RuntimeError(
                "YuNet is unavailable. Install "
                "opencv-python==4.10.0.84"
            )

        self.detector = cv2.FaceDetectorYN_create(
            str(YUNET_MODEL_PATH),
            "",
            (320, 320),
            0.85,
            0.3,
            5000,
        )

        self.confidence_threshold = 0.85

    def detect(self, frame):

        if frame is None or frame.size == 0:
            return []

        height, width = frame.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(frame)

        if faces is None:
            return []

        results = []

        for face in faces:

            x, y, w, h = face[:4]
            confidence = float(face[14])

            if confidence < self.confidence_threshold:
                continue

            x, y, w, h = map(
                int, (x, y, w, h)
            )

            if w < 40 or h < 40:
                continue

            ratio = w / float(h)

            if ratio < 0.65 or ratio > 1.35:
                continue

            x = max(0, x)
            y = max(0, y)

            w = min(w, width - x)
            h = min(h, height - y)

            if w > 0 and h > 0:
                results.append((x, y, w, h))

        return results


# ==================================================
# 3. AGE AND GENDER CNN DETECTOR
# ==================================================

class FaceAgeGenderDetector:
    """Predict age and gender using the trained CNN."""

    def __init__(self, model_path=MODEL_PATH):

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {model_path}"
            )

        # Load saved training checkpoint.
        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
        )

        self.image_size = int(
            checkpoint["image_size"]
        )

        self.senior_age_threshold = int(
            checkpoint.get("senior_age_threshold", 60)
        )

        self.gender_classes = checkpoint.get(
            "gender_classes",
            ["Male", "Female"],
        )

        # Initialize the original CNN architecture.
        self.model = AgeGenderCNN().to(DEVICE)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # Same preprocessing used during training.
        self.transform = transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=checkpoint["normalization_mean"],
                std=checkpoint["normalization_std"],
            ),
        ])

    # --------------------------------------------------
    # AGE AND GENDER PREDICTION
    # --------------------------------------------------

    def predict(self, face_bgr):

        if face_bgr is None or face_bgr.size == 0:
            return None

        # Convert OpenCV BGR to RGB.
        face_rgb = cv2.cvtColor(
            face_bgr,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(face_rgb)

        # Apply training preprocessing.
        image_tensor = self.transform(
            image
        ).unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE)

        # CNN prediction.
        with torch.no_grad():

            predicted_age, gender_logits = self.model(
                image_tensor
            )

            raw_age = float(
                predicted_age.item()
            )

            gender_probabilities = torch.softmax(
                gender_logits,
                dim=1,
            )

            gender_index = int(
                gender_probabilities.argmax(
                    dim=1
                ).item()
            )

            gender_confidence = float(
                gender_probabilities[
                    0, gender_index
                ].item()
            )

        # Convert predictions into application values.
        age = max(
            0,
            min(100, round(raw_age)),
        )

        gender = self.gender_classes[gender_index]

        is_senior = (
            age > self.senior_age_threshold
        )

        status = (
            "Senior Citizen"
            if is_senior
            else "Non-Senior"
        )

        # Return every key expected by app.py
        return {
            "age": age,
            "gender": gender,
            "senior": is_senior,
            "is_senior": is_senior,
            "category": status,
            "status": status,
            "gender_confidence": round(
                gender_confidence,
                4,
            ),
        }