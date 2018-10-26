#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import argparse, h5py, math
import scipy.stats as stats
from sklearn.metrics import mean_squared_error as mse

import rot_cnn

def get_arguments():
    parser = argparse.ArgumentParser(description='Takes a pair of images and tests against a trained rotation CNN, returning the angle between them.')
    parser.add_argument('--test_data_file', type=str, required=True, help='Testing dataset HDF5 file (generated via create_data_set.py).')
    parser.add_argument('--model_base', type=str, required=True, help='Base for pre-trained model file.')
    parser.add_argument('--results_file', type=str, required=True, help='Results file to save comparison true and predicted labels.')
    return parser.parse_args()

def test_image_pairs(image_pairs, output_model_base):

    cnn = rot_cnn.cnn('rot_cnn', model_file_base)

    if not cnn.is_trained():
        print('Error! model not trained.')

    cnn.set_session()
    cnn.load_model()

    # TODO Check size of image pairs matches that for trained CNN

    return cnn.test(image_pairs)

### Main function ###
if __name__ == "__main__":
    # Parse console and print messages
    args = get_arguments()

    model_file_base = args.model_base
    test_data_f = h5py.File(args.test_data_file,'r')
    results_file = args.results_file

    test_images = test_data_f['image_data']
    test_labels = test_data_f['rotations']

    angles = test_image_pairs(test_images, model_file_base)

    # Calculate PCC for labels against returnes angles

    pcc = stats.pearsonr(test_labels, angles)
    rmse = math.sqrt(mse(test_labels, angles))

    print('Test set PCC:', pcc,'RMSE:', rmse)

    # Output resulting pairs to results file
    f = open(results_file,'w')

    for idx,angle in enumerate(angles):
        f.write(str(test_labels[idx])+' '+str(angle)+'\n')

    f.close()
