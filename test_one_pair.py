#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os
import numpy as np
import argparse, cv2

import test_data_set

def get_arguments():
    parser = argparse.ArgumentParser(description='Takes a pair of images and tests against a trained rotation CNN, returning the angle between them.')
    parser.add_argument('--first_image', type=str, required=True, help='First image to test.')
    parser.add_argument('--second_image', type=str, required=True, help='Second image to test.')
    parser.add_argument('--data_im_size', type=int, default=100, help='Size images are rescaled to for test.')
    parser.add_argument('--output_model_base', type=str, required=True, help='Base for output file for model')

    return parser.parse_args()

### Main function ###
if __name__ == "__main__":
    # Parse console and print messages
    args = get_arguments()
    first_image = args.first_image
    second_image = args.first_image
    data_im_size = args.data_im_size
    output_model_base = args.output_model_base

    im = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE)
    rescale_ratio = data_im_size/im.shape[0]
    im = cv2.resize(im,(0,0),fx=rescale_ratio,fy=rescale_ratio)

    im2 = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE)
    im = im.append(cv2.resize(im2,(0,0),fx=rescale_ratio,fy=rescale_ratio))

    angle = test_data_set.test_image_pairs(im, output_model_base)[0]

    print('Angle between images:',angle)
