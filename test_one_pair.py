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
    parser.add_argument('--model_base', type=str, required=True, help='Base for output file for model')
    parser.add_argument('--output_test_image', action='store_true', help='Output the combined image to test.jpg')

    return parser.parse_args()

### Main function ###
if __name__ == "__main__":
    # Parse console and print messages
    args = get_arguments()
    first_image = args.first_image
    second_image = args.second_image
    data_im_size = args.data_im_size
    model_base = args.model_base
    output_test_image = args.output_test_image

    final_im = np.zeros([1,2*data_im_size, data_im_size])

    im = cv2.imread(first_image, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(second_image, cv2.IMREAD_GRAYSCALE)

    # Check input images are both square and same size
    if (im.shape != im2.shape):
        print('Error! Images must be of same size')
        sys.exit(0)

    if (im.shape[0] != im.shape[1]) or (im2.shape[0] != im2.shape[1]):
        print('Error! Images must both be square')
        sys.exit(0)

    rescale_ratio = data_im_size/im.shape[0]

    final_im[0,:data_im_size,:] = cv2.resize(im,(0,0),fx=rescale_ratio,fy=rescale_ratio)
    final_im[0,data_im_size:,:] = cv2.resize(im2,(0,0),fx=rescale_ratio,fy=rescale_ratio)

    if output_test_image:
        cv2.imwrite('test.jpg',final_im[0], [cv2.IMWRITE_JPEG_QUALITY, 100])

    angle = test_data_set.test_image_pairs(final_im, model_base)[0]

    print('Angle between images:',angle[0])
