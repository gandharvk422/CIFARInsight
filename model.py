import torch
import torch.nn as nn
import torch.optim as optim
import tarfile
import urllib.request
import os
from torchvision import datasets, transforms

# Define the URL and download path
url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
data_dir = "data"

# Download the dataset if not already downloaded
if not os.path.exists(data_dir):
    print("Downloading CIFAR-10 dataset...")
    urllib.request.urlretrieve(url, "cifar10.tgz")
    with tarfile.open("cifar10.tgz", "r:gz") as tar:
        tar.extractall()

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load the training dataset
train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

class Cifar10Model(nn.Module):
    def __init__(self):
        super(Cifar10Model, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 classes for CIFAR-10
        )

    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Create and train the model
model = Cifar10Model()
train_model(model, train_loader, num_epochs=10)

# Save the model
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)

# Call save function
save_model(model)