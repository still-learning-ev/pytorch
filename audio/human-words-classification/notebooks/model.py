import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Prooposed DCNN from paper Classification of Apple Disease Based on Non-Linear Deep Features
class DCNN(nn.Module):
    def __init__(self, dropout ):
        super().__init__()
        self.convs = nn.Sequential(
            self._block(1, 128),
            self._block(128, 64),
            self._block(64, 64),
            self._block(64, 32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2)
        )
        
        self.dense = nn.Sequential(
            nn.Flatten(),
            
            nn.Dropout(p=dropout),
            nn.Linear(in_features=1024, out_features=256),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256, out_features=3)
        )
    
    def forward(self, x):
        x = self.convs(x)
        return self.dense(x)
        
    def _block(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU()
        )
model = DCNN(dropout=0.25)