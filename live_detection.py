# LiDAR CNN Detection
# Analysing data from a Blickfeld Cube 1 LiDAR sensor with a simple convolutional neural network
# **HERE**: Load trained binary-classification model from LOAD_PATH and make live prediction from recieved LiDAR data every second

import argparse
import blickfeld_scanner
import numpy as np
import time
import torch

RUN_TIME = 60 # Seconds
LOAD_PATH = "models/model1.pt"# Path to model to use

def lidar_person_live_detection(args):
    # Connect to the device
    scanner = blickfeld_scanner.scanner(args.host)
    print(scanner)
    
    for _ in range(RUN_TIME):
        # Create a point cloud stream object and get a frame
        stream = scanner.get_point_cloud_stream()
        frame = stream.recv_frame()

        scanlines = 60  # Number of scanlines (rows) for Blickfeld Cube 1 LiDAR sensor
        points = 181   # Number of points per scanline (columns)

        # Initialize an array to store the ranges
        range_array = np.zeros((scanlines, points), dtype=np.float32)

        # Scanlines are recieved twice per scan (from left to right and right to left) -> Saving every other scanline to get actual point cloud
        for i, scanline in enumerate(frame.scanlines):
            if i % 2 != 0:
                continue
            for j, point in enumerate(frame.scanlines[i].points):
                for r_ind in range(len(point.returns)):
                    # Save the range/distance of the last return per point (the furthest the light travelled)
                    ret = point.returns[r_ind]
                    range_array[int(scanline.id/2), j] = max(ret.range, range_array[int(scanline.id/2), j])

        # Load model
        model = torch.load(LOAD_PATH, weights_only=False)
        model.eval()

        # Make predictions
        with torch.inference_mode():
            cloud = torch.from_numpy(range_array).unsqueeze(dim=0)

            # Save predictions and labels (20% threshhold)
            preds = (torch.sigmoid(model(cloud.unsqueeze(dim=0))))
            label = (1 if float(preds.squeeze(dim=1)) > 0.2 else 0) # Note: Change the classification threshhold as it fits best per model (e.g.: testmodel0.pt: 0.03, model1.pt: 0.2)
            print(label)
        
        # Wait 1 sec before next scan
        time.sleep(1)

    # Stop pointcloud stream
    del stream

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", help="hostname or IP of device") # Host name or IP address of the device
    args = parser.parse_args()

    lidar_person_live_detection(args)
