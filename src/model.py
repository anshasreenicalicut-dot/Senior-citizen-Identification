
import torch
from torch import nn


class AgeGenderCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
        )

        self.age_head = nn.Linear(128, 1)
        self.gender_head = nn.Linear(128, 2)

    def forward(self, x):
        features = self.features(x)
        shared = self.shared(features)

        # Age is predicted in years.
        age = self.age_head(shared).squeeze(1)

        # Gender class 0 = Male, 1 = Female.
        gender_logits = self.gender_head(shared)

        return age, gender_logits
