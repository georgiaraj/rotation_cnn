#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import sys, os
import imutils, cv2
import numpy as np
import glob, h5py, argparse, random

def get_arguments():
    parser = argparse.ArgumentParser(description='Generates rotations of images found in image_folder and saves pairs of images (original and rotated) with the rotation angle to HDF5 file.')
    parser.add_argument('--image_folder', type=str, required=True, help='Image folder for generating dataset.')
    parser.add_argument('--output_file', type=str, required=True, help='Output file for dataset')
    parser.add_argument('--data_im_size', type=int, default=200, help='Square size images will be resized to.')
    parser.add_argument('--num_rots', type=int, default=5, help='Number of rotations to add for each image.')
    parser.add_argument('--image_format', type=str, default='jpg', help='Format of images to use.')
    parser.add_argument('--save_images', action='store_true', help='If set to false don\'t save images in viewable form in HDF5 file.')


    return parser.parse_args()

def add_image_rotations(image):

    print('Add',num_rots,'pairs of images for',image,'to dataset.')

    image_ind = dset_image_data.shape[0]

    # Increase the dataset size according to the number of patches returned
    if save_images:
        dset_images.resize(dset_images.shape[2]+num_rots, axis=2)
    dset_image_data.resize(dset_image_data.shape[0]+num_rots, axis=0)
    dset_labels.resize(dset_labels.shape[0]+num_rots, axis=0)

    # Create a square image from the original image
    square_im = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
    im_size = min(square_im.shape)
    rescale_ratio = data_im_size/im_size
    square_im = cv2.resize(square_im[:im_size,:im_size],(0,0),fx=rescale_ratio,fy=rescale_ratio)

    for i in range(num_rots):

        # Generate random angle of rotation
        angle = random.randint(0,360)

        # Create rotated image
        rotated = imutils.rotate(square_im, angle)

        print('Image pair',image_ind,': image',image,'rotated by',str(angle))

        # Add both images and angle label to dataset
        if save_images:
            dset_images[:data_im_size,:,image_ind] = square_im
            dset_images[data_im_size:,:,image_ind] = rotated
        dset_image_data[image_ind,:data_im_size,:] = square_im
        dset_image_data[image_ind,data_im_size:,:] = rotated
        dset_labels[image_ind] = angle

        image_ind+=1


### Main function ###
if __name__ == "__main__":
    # Parse console and print messages
    args = get_arguments()

    image_folder = args.image_folder
    image_format = args.image_format
    data_im_size = args.data_im_size
    hdf5_file = args.output_file
    num_rots = args.num_rots
    save_images = args.save_images

    print('Image folder:',image_folder)
    print('Output file:',hdf5_file)

    if save_images:
        print('Saving viewable images alongside image data')

    # Read in appropriate images from folder for extracting patches
    image_files = glob.glob(image_folder+'/*.'+image_format)

    if len(image_files)==0:
        print('Error: no ',image_format,'image files found.')
        sys.exit(0)

    image_files.sort()
    print(len(image_files),'images to add with',num_rots,'each.')

    # Create HDF5 file and set up the data sets
    f = h5py.File(hdf5_file,'w')

    dset_image_data = f.create_dataset("image_data",(0,data_im_size*2,data_im_size),maxshape=(None,data_im_size*2,data_im_size))
    dset_labels = f.create_dataset("rotations",(0,1),maxshape=(None,1))

    if save_images:
        dset_images = f.create_dataset("images",(data_im_size*2,data_im_size,0),maxshape=(data_im_size*2,data_im_size,None))
        # Set the image attributes
        dset_images.attrs['CLASS'] = 'IMAGE'
        dset_images.attrs['IMAGE_VERSION'] = '1.2'
        dset_images.attrs['IMAGE_SUBCLASS'] =  'IMAGE_INDEXED'
        dset_images.attrs['IMAGE_MINMAXRANGE'] = np.array([0,255], dtype=np.uint8)

    # Add image files to the dataset
    for idx, im in enumerate(image_files):
        add_image_rotations(im)
