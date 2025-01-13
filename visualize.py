# LiDAR CNN Detection
# Analysing data from a Blickfeld Cube 1 LiDAR sensor with a simple convolutional neural network
# **HERE**: Visualize the LiDAR point clouds stored in DIR as Numpy arrays using matplotlib

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DIR = "data/data1" # Data directory to visualize from
SAMPLES = 400 # Max number of samples from DIR to be plotted
PLOTS = (20, 20) # Product must be equal to SAMPLES

array = []
path = Path(DIR)
count = min(len(list(path.glob('*.npy'))), SAMPLES)

# Save all numpy arrays from dir and convert them to image-like structure
for i in range(count):
    temp_array = np.load(f"{DIR}/{i}.npy")
    temp_array *= 50
    temp_array = temp_array.astype(np.uint8)
    array.append(temp_array)
    
# Plot the data
fig = plt.figure(figsize=(10, 7))

for i in range(1, count):
    plt.subplot(PLOTS[0], PLOTS[1], i)
    plt.imshow(array[i], interpolation="none", cmap="gray", vmin=0, vmax=255)
    plt.axis(False)

plt.show()
