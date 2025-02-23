import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import numpy as np
import os

# Define a simple U-Net model for AI keying
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load a dummy dataset (you can replace this with real training data)
def load_data():
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    dataset = datasets.FakeData(transform=transform)  # Dummy dataset for testing
    return torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Train the AI model
def train_model():
    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()

    data_loader = load_data()

    print("🔄 Training AI Model for Keying...")
    for epoch in range(5):  # Reduce epochs for a quick test
        for img, _ in data_loader:
            output = model(img)
            loss = loss_fn(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), "ai_keyer_model.pth")
    print("✅ AI Model Saved: ai_keyer_model.pth")

# Run the training function
if __name__ == "__main__":
    train_model()
