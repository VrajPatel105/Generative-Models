import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

Batch_size = 128
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(
    root="./data",
    train = True,
    download = True,
    transform = transform
)

train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)

# coding the vae class now

class VAE(nn.Module):

    def __init__(self, input_dim = 784, hidden_dim = 256, latent_dim = 64):
        super().__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder 
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z 

    def decode(self, z):
        h = self.relu(self.fc2(z))
        output = self.sigmoid(self.fc3(h))
        return output

    # Forward Pass
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
    

# coding the loss 

def vae_loss(reconstructed_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kl_loss) / x.size(0)


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    
    for x, _ in (train_loader):
        x = x.view(-1, 784).to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss = vae_loss(x_recon, x , mu, logvar)

        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch [{epoch+1}/{epochs}], Loss : {avg_loss:.4f}")


# plotting 

model.eval()

with torch.no_grad():
    x, _ = next(iter(train_loader))
    x = x.view(-1, 784).to(device)

    x_recon, _, _ = model(x)

    # Reshape back to image format: (batch, 1, 28, 28)
    x = x.view(-1, 1, 28, 28).cpu()
    x_recon = x_recon.view(-1, 1, 28, 28).cpu()

    # Clamp values to valid image range
    x = x.clamp(0, 1)
    x_recon = x_recon.clamp(0, 1)

    # Take first 8 images for display
    x = x[:8]
    x_recon = x_recon[:8]

    # Create grids
    original_grid = make_grid(x, nrow=8, padding=2)
    recon_grid = make_grid(x_recon, nrow=8, padding=2)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))

    axes[0].imshow(original_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    axes[0].set_title("Original Images")
    axes[0].axis("off")

    axes[1].imshow(recon_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    axes[1].set_title("Reconstructed Images")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


model.eval()

with torch.no_grad():
    # 1. Sample 5 random vectors from a standard normal distribution
    # latent_dim=64 must match your VAE class definition
    z = torch.randn(5, 64).to(device)
    
    # 2. Decode the random latent vectors into images
    generated_images = model.decode(z)
    
    # 3. Reshape (batch, 1, 28, 28) and plot
    generated_images = generated_images.view(-1, 1, 28, 28).cpu()
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        axes[i].imshow(generated_images[i].squeeze(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Sample {i+1}")
    plt.tight_layout()
    plt.show()