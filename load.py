# LiDAR CNN Detection
# Analysing data from a Blickfeld Cube 1 LiDAR sensor with a simple convolutional neural network
# **HERE**: Load a model from PATH, make predictions of VALIDATION_SAMPLES in VALIDATION_DIR, and plot them

import torch
import numpy as np
import matplotlib.pyplot as plt

PATH = "models/model1.pt" # Path to model to use
VALIDATION_DIR = "data/data2" # Path to data to make predictions
VALIDATION_SAMPLES = 20 # Number of samples to analyse
PLOTS = (4, 5) # Product must be equal to VALIDATION_SAMPLES

# Load model
model = torch.load(PATH, weights_only=False)
model.eval()

# Make predictions
preds = []
labels = []
with torch.inference_mode():
    for i in range(VALIDATION_SAMPLES):
        cloud = np.load(f"{VALIDATION_DIR}/{i}.npy")
        cloud = torch.from_numpy(cloud).unsqueeze(dim=0)

        # Save predictions and labels (19% threshhold)
        preds.append(torch.sigmoid(model(cloud.unsqueeze(dim=0))))
        labels.append(1 if float(preds[i].squeeze(dim=1)) > 0.19 else 0) # Note: Change the classification threshhold as it fits per model (e.g.: testmodel0.pt: 0.03, model1.pt: 0.19)

# Process inputs to be plotted like images
array = []
for i in range(VALIDATION_SAMPLES):
    temp_array = np.load(f"{VALIDATION_DIR}/{i}.npy")
    temp_array *= 50
    temp_array = temp_array.astype(np.uint8)
    array.append(temp_array)
    
# Plot images of inputs with prediction probabilities (1 = person, 0 = no person)
fig = plt.figure(figsize=(10, 7))

for i in range(1, VALIDATION_SAMPLES):
    plt.subplot(PLOTS[0], PLOTS[1], i)
    plt.title(f"{labels[i-1]}") # Show predicted label (1 = person, 0 = no person)
    # plt.title(f"{labels[i-1]}  |  {float(preds[i-1].squeeze(dim=1)):.4f}") # Show label and prediction probability
    plt.suptitle(VALIDATION_DIR[5:])
    plt.imshow(array[i-1], interpolation="none", cmap="gray", vmin=0, vmax=255)
    plt.axis(False)

plt.show()
