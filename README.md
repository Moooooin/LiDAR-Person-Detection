# BLickfeld Cube 1 LiDAR binary classification for detecting humans in a set doorframe

Created 10.01.2025 Brunswick, Germany

### Feel free to create your own data and train your own models.

**! Make sure you have all the required libraries installed:**

- ``PyTorch`` version 2.2.2 or higher
- ``Matplotlib`` version 3.8.2 or higher
- ``Numpy`` version 1.26.2 or higher
- ``Pathlib`` (In Python 3.3 or lower, otherwise no need to install)
- ``Blickfeld_scanner`` (only required for live detection or creating your own data)

### How to use:
Please take a look at the files themselves and the comments in the code for further clarification.
- ``load.py`` Loads a trained model and visualizes predictions of it.
- ``train.py`` Creates and trains a binary classification model on specified data before saving it.
- ``visualize.py`` Visualizes numpy arrays of scans made with the LiDAR which are located a specified directory.
- ``get_lidar_data.py`` Saves scans of a connected Blickfeld Cube 1 LiDAR to a specified directory as a numpy array.
- ``live_detection.py`` Prints out real time prediction of a specified model analysing the data of a connected Blickfeld Cube 1 LiDAR (1: person, 0: no person).

<br>

Special thanks to the German Aerospace Center (DLR) for providing access to this LiDAR sensor and enabling this.