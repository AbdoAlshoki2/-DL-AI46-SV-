import torch
import torch.nn as nn


class GestureNet(nn.Module):
    """
    Feed-forward neural network for hand gesture classification.

    Architecture
    ------------
    Input  (63)  ->  Block 1 (256)  ->  Block 2 (128)  ->  Block 3 (64)  ->  Output (num_classes)

    Each hidden block: Linear -> BatchNorm -> ReLU -> Dropout
    """

    def __init__(self, input_size: int = 63, num_classes: int = 10):
        super().__init__()

        self.network = nn.Sequential(
 
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),


            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),


            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),


            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
