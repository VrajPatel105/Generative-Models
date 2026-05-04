import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Batch_size = 128
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNSIT(
    root="./data",
    train = True,
    download = True,
    transform = transform
)

train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)

# coding the vae class now

latent_dim = 64

class VAE(nn.Module):

    def __init__(self):
        super.__init__()

    # Encoder
    

    # Decoder 


    # Forward Pass
    def forward(self, x):
        pass