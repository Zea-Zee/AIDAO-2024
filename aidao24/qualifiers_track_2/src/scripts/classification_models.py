import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class MLP(nn.Module):
    def __init__(self, input_size, hidden_units=[64, 32]):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_units[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_units[-1], 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
