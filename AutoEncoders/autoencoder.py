import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# configuring the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# data -> train and val
batch_size = 128

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# class that defines the autoencoder

class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=256):
        super().__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 784),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    
# defining the model and it's loss

model = AutoEncoder().to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10

# training loop

model.train()

for epoch in range(epochs):
    total_loss = 0
    for x, _ in train_loader:
        x = x.view(-1, 784).to(device)
    
        optimizer.zero_grad()

        x_reconstructed = model(x)

        loss = criterion(x_reconstructed, x)

        loss.backward()

        optimizer.step()

        total_loss = total_loss + loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"epoch [{epoch+1}] | average loss [{avg_loss:.6f}]")

model.eval()

with torch.no_grad():
    # Get a batch of data
    x, _ = next(iter(train_loader))
    x = x.view(-1, 784).to(device)

    # Perform inference
    x_reconstructed = model(x)

    # Prepare for plotting
    n = 10
    plt.figure(figsize=(12, 4))

    for i in range(n):
        # Original images
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i].cpu().view(28, 28), cmap='gray')
        plt.axis("off")

        # Reconstructed images
        plt.subplot(2, n, i + 1 + n)
        # Using .detach() ensures no gradients are tracked for plotting
        plt.imshow(x_reconstructed[i].cpu().detach().view(28, 28), cmap='gray')
        plt.axis("off")

    # Save the figure to a file
    plt.tight_layout()
    plt.savefig("reconstruction_results.png", dpi=300)
    
    # Display the plot
    plt.show()
    
# plotting out a random point 
model.eval()

with torch.no_grad():
    # 1. Sample 5 random points from the latent space
    # NOTE: These values don't follow a N(0, 1) distribution, 
    # so the results will likely be visual noise.
    z = torch.randn(5, 32).to(device) 
    
    # 2. Decode the points
    generated_images = model.decoder(z)
    
    # 3. Reshape and plot
    generated_images = generated_images.view(-1, 1, 28, 28).cpu()
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Random Sample {i+1}")
    plt.tight_layout()
    plt.show()