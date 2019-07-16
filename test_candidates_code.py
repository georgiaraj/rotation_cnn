#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, cv2, time
import numpy as np
import argparse, h5py, math
import scipy.stats as stats
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

from getrotation import get_rotation

def angle_difference(x, y):
    return 180 - abs(abs(x - y) - 180)

def get_arguments():
    parser = argparse.ArgumentParser(description='Takes a pair of images and tests against a trained rotation CNN, returning the angle between them.')
    parser.add_argument('--test_data_file', type=str, required=True, help='Testing dataset HDF5 file (generated via create_data_set.py).')
    parser.add_argument('--results_file', type=str, required=True, help='Results file to save comparison true and predicted labels.')
    parser.add_argument('--image_format', type=str, choices={'rgb','bw'}, default='rgb', help='Whether to test model on RGB or grayscale images.')
    return parser.parse_args()

### Main function ###
if __name__ == "__main__":
    # Parse console and print messages
    args = get_arguments()

    test_data_f = h5py.File(args.test_data_file,'r')
    results_file = args.results_file

    if args.image_format == 'rgb':
        test_images = test_data_f['images_rgb']
        len_idx = 3
    else:
        test_images = test_data_f['images']
        len_idx = 2
    test_labels = test_data_f['rotations']

    angles = np.empty(shape=(test_images.shape[len_idx],1))
    for idx in range(test_images.shape[len_idx]):
        if args.image_format == 'rgb':
            im1 = test_images[:100,:,:,idx]
            im2 = test_images[100:,:,:,idx]
        else:
            im1 = test_images[:100,:,idx]
            im2 = test_images[100:,:,idx]
        angles[idx] = get_rotation(im1,im2)

    angles[angles < 0] = 360+angles[angles<0]

    print(angles.shape)
    print(test_labels.shape)

    # Calculate metrics for labels against returnes angles
    pcc = stats.pearsonr(test_labels, angles)
    rmse = math.sqrt(mse(test_labels, angles))
    ad = np.mean(angle_difference(test_labels, angles))

    print('Test set PCC:', pcc[0],'RMSE:', rmse, 'AD:', ad)

    errors = np.array(test_labels-angles)
    errors = np.sign(errors)*angle_difference(test_labels, angles)

    # Plot histogram of errors for different angles
    hist = np.histogram(errors, 36)
    n, bins, patches = plt.hist(errors, bins=36, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlim([-180,180])
    plt.xlabel('Angle Error')
    plt.xticks(np.arange(-180,180,30), fontsize=6)
    plt.ylabel('Frequency of Errors')
    plt.title('Error Distribution')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    plt.savefig(results_file.replace('.csv','.png'))

    # Output resulting pair to results file
    f = open(results_file,'w')

    for idx,angle in enumerate(angles):
        f.write(str(test_labels[idx][0])+' '+str(angle[0])+'\n')

    f.close()
