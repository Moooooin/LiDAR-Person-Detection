# LiDAR CNN Detection
# Analysing data from a Blickfeld Cube 1 LiDAR sensor with a simple convolutional neural network
# **HERE**: Get SAMPLE frames point clouds from LiDAR sensor and save it as numpy array in SAVE_PATH every SLEEP_TIME seconds

import argparse
import blickfeld_scanner
import numpy as np
import time

SAVE_PATH = "data/data2" # Where to save the data
SLEEP_TIME = 0.5 # Time between each scan
SAMPLES = 10 # Number of scans/samples to generate
START_IDX = 20 # Start name of the first .npy file (will increase by SAMPLES till the program ends)

def get_lidar_data(args):
    # Connect to the device
    scanner = blickfeld_scanner.scanner(args.host)
    print(scanner)

    for zahl in range(START_IDX, (START_IDX + SAMPLES)):
         # Create a point cloud stream object and get a frame
        stream = scanner.get_point_cloud_stream()
        frame = stream.recv_frame()

        scanlines = 60  # Number of scanlines (rows) for Blickfeld Cube 1 LiDAR sensor
        points = 181   # Number of points per scanline (columns)

        # Initialize array to store the ranges
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

        # Save the current frame and wait SLEEP_TIME seconds before next iteration
        print(f"Saving frame {zahl}.")
        np.save(f"{SAVE_PATH}/{zahl}", range_array)
        time.sleep(SLEEP_TIME)

    # Stop pointcloud stream
    del stream

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", help="hostname or IP of device") # Host name or IP address of the device
    args = parser.parse_args()

    get_lidar_data(args) 
