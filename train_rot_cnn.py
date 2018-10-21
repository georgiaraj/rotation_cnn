#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import argparse, h5py
import rot_cnn

def get_arguments():
    parser = argparse.ArgumentParser(description='Trains a rotation CNN on the given training data HDF5 file.')
    parser.add_argument('--train_data_file', type=str, required=True, help='Training dataset HDF5 file (generated via create_data_set.py).')
    parser.add_argument('--output_model_base', type=str, required=True, help='Base for output file for model')

    return parser.parse_args()


### Main function ###
if __name__ == "__main__":
    # Parse console and print messages
    args = get_arguments()

    model_file_base = args.output_model_base
    train_data_f = h5py.File(args.train_data_file,'r')

    training_images = train_data_f['image_data']
    training_labels = train_data_f['rotations']

    image_size = training_images.shape[1]

    cnn = rot_cnn.cnn('rot_cnn', model_file_base, image_size)

    cnn.train(training_images, training_labels)
