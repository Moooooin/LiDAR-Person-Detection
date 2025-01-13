# LiDAR CNN Detection
# Analysing data from a Blickfeld Cube 1 LiDAR sensor with a simple convolutional neural network
# **HERE**: Create a simple CNN and train it for EPOCHS loops through the training and testing set

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from pathlib import Path

# Hyperparameters
LR = 0.001
BATCHES = 32
DEVICE = "cpu"
EPOCHS = 400
MODEL_SAVE_DIR = "models"

# Initialize dataset class
class NumpyBinaryClassificationDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data1_dir = os.path.join(data_dir, 'data1')
        self.data0_dir = os.path.join(data_dir, 'data0')

        # List .npy files in both directories
        self.data1_files = [(np.load(f"{self.data1_dir}/{f}"), 1) for f in os.listdir(self.data1_dir) if f.endswith('.npy')]
        self.data0_files = [(np.load(f"{self.data0_dir}/{f}"), 0) for f in os.listdir(self.data0_dir) if f.endswith('.npy')]

        self.files = []
        self.files.extend([(torch.from_numpy(data).unsqueeze(dim=0), label) for data, label in self.data1_files])
        self.files.extend([(torch.from_numpy(data).unsqueeze(dim=0), label) for data, label in self.data0_files])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]


# Load the full dataset
dataset = NumpyBinaryClassificationDataset(data_dir="data/")

# Calculate train/test split (80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Use random_split for train/test split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders (python iterables over our datasets with batches)
train_loader = DataLoader(train_dataset, batch_size=BATCHES, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCHES, shuffle=False)

# Initialize model (CNN - Convolutional Neural Network)
model = nn.Sequential(
    nn.Conv2d(1, 16, 3),
    nn.ReLU(),
    nn.MaxPool2d(4),
    nn.Flatten(),
    nn.Linear(9856, 32),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(32, 1)
)

# Standard non-linear ANN for comparison if you'd like to try
# model = nn.Sequential(
    # nn.Flatten(),
    # nn.Linear(10860, 32),
    # nn.ReLU(),
    # nn.Linear(32, 32),
    # nn.ReLU(),
    # nn.Linear(32, 1)
# )

# Move model to device (GPU or CPU)
model = model.to(DEVICE)

# Define optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# Training loop
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    # Iterate over batches of our training set
    for inputs, labels in train_loader:
        # Device agnostic
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Reshape labels to (batch_size, 1) to match model output shape
        labels = labels.view(-1, 1).float()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Ensure that the output is a scalar for each sample [batch_size, 1]
        outputs = outputs.view(-1, 1) 

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Gradient descent
        optimizer.step()

        running_loss += loss.item()

        # Make predictions (0 or 1) by rounding the output (since it's a probability after applying sigmoid)
        predicted = torch.round(torch.sigmoid(outputs))

        # Calculate accuracy
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_preds / total_preds
    return epoch_loss, epoch_acc

# Evaluation loop
def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(dim=1)

            # Forward pass
            outputs = model(inputs)
        
            # Calculate the loss
            loss = criterion(outputs, labels.float())
            running_loss += loss.item()

            # Compute accuracy
            predicted = (outputs.squeeze() > 0.5).float()
            correct_preds += (predicted == labels.squeeze()).sum().item()
            total_preds += labels.size(0)

    epoch_loss = running_loss / len(test_loader)
    epoch_accuracy = correct_preds / total_preds

    return epoch_loss, epoch_accuracy


# Execute training and evaluation loop
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"Epoch [{epoch+1}/{EPOCHS}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
          f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Save model
model_save_num = len(list(Path(MODEL_SAVE_DIR).glob('*.pt')))
torch.save(model, f"{MODEL_SAVE_DIR}/model{model_save_num}.pt")
