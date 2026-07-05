"""
cnn_model.py

Residual 1D CNN for Automatic Modulation Classification.
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(

                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm1d(out_channels)

            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)

        out = self.bn2(out)

        out += identity

        out = self.relu(out)

        return out
class CNNModel(nn.Module):

    def __init__(self, num_classes=9):

        super().__init__()

        self.stem = nn.Sequential(

            nn.Conv1d(
                2,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1
            )

        )

        self.layer1 = ResidualBlock(
            64,
            64
        )

        self.layer2 = ResidualBlock(
            64,
            128,
            stride=2
        )

        self.layer3 = ResidualBlock(
            128,
            256,
            stride=2
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.5),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )

        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x
    
def count_parameters(model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

if __name__ == "__main__":

    model = CNNModel()

    print(model)

    print()

    print(
        "Trainable Parameters :",
        count_parameters(model)
    )

    dummy = torch.randn(
        8,
        2,
        128
    )

    output = model(dummy)

    print()

    print("Output Shape :", output.shape)


