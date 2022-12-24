""" From the official Pytorch tutorial:
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html """

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    def reward(self, outcome:int,y:torch.Tensor):
            """ outcome should be a 1 or -1 """
            loss_fn = torch.nn.BCEWithLogitsLoss()
            reward = outcome
            



model = NeuralNetwork().to(device) # initialize the model
print(model)