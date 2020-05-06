#!/usr/bin/env python3


import nibabel as nib
import tensorflow as tf
import numpy as np
import os
import glob
import math
import sys


def get_images(imgdir):
    # get image paths
    imgpaths = []
    imgpaths = sorted(glob.glob(imgdir + '/*nii.gz'))
    numimgs = len(imgpaths)
    
    # get image size
    img = nib.load(imgpaths[0])
    imgdata = img.get_fdata()
    [x, y, z] = imgdata[:, :, :, 0].shape
    
    # initialize image arrays
    imgs = np.zeros((x, y, z, 1, numimgs))
    
    # read in images
    for i in range(numimgs):
        img = nib.load(imgpaths[i])
        data = img.get_fdata()
        imgs[:, :, :, :, i] = data
    
    # return image array
    return imgs, imgpaths


# Create some wrappers for simplicity
@tf.function
def conv3d(x, W, b, strides=2):
    # conv2d wrapper, with bias and leaky_relu activation
    x = tf.nn.conv3d(x, W, strides=[1, strides, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x)

@tf.function
def conv3dt(x, W, b, output_shape, strides=2):
    # conv3d_transpose wrapper
    x = tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, strides, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.leaky_relu(x)

@tf.function
def cnn(x, weights, bias, batch_size, num_nodes):
    # add 3 convolutional layers (dimensionality reduction)
    conv1 = conv3d(x, weights['wc1'], bias['bc1'])
    conv2 = conv3d(conv1, weights['wc2'], bias['bc2'])
    conv3 = conv3d(conv2, weights['wc3'], bias['bc3'])
    
    # add 3 deconvolutional layers (dimensionality increase)
    deconv1 = conv3dt(conv3, weights['wc4'], bias['bc4'], (batch_size, 51, 74, 79, num_nodes)) # hard coded size
    deconv2 = conv3dt(deconv1, weights['wc5'], bias['bc5'], (batch_size, 103, 149, 159, num_nodes)) # hard coded size
    deconv3 = conv3dt(deconv2, weights['wc6'], bias['bc6'], (batch_size, 208, 300, 320, 1)) # hard coded size
    
    return deconv3


def main(args):
    # NOTE: input directories must have corresponding t1 and t2s for training ONLY
    #       t1s and t2s must also be aligned
    if len(args) < 3:
        print("Usage:\t./t1_to_t2_train.py <t1dir> <outdir> <wc1.npy> <wc2.npy> <wc3.npy> <wc4.npy> <wc5.npy> <wc6.npy> <bc1.npy> <bc2.npy> <bc3.npy> <bc4.npy> <bc5.npy> <bc6.npy>")
        exit()
    
    # get input arguments
    t1dir = args[0]
    outdir = args[1]
    weights = {
        'wc1': tf.compat.v1.get_variable(name='wc1', dtype=tf.dtypes.float64, initializer=np.load(args[2])),
        'wc2': tf.compat.v1.get_variable(name='wc2', dtype=tf.dtypes.float64, initializer=np.load(args[3])),
        'wc3': tf.compat.v1.get_variable(name='wc3', dtype=tf.dtypes.float64, initializer=np.load(args[4])),
        'wc4': tf.compat.v1.get_variable(name='wc4', dtype=tf.dtypes.float64, initializer=np.load(args[5])),
        'wc5': tf.compat.v1.get_variable(name='wc5', dtype=tf.dtypes.float64, initializer=np.load(args[6])),
        'wc6': tf.compat.v1.get_variable(name='wc6', dtype=tf.dtypes.float64, initializer=np.load(args[7]))}

    biases = {
        'bc1': tf.compat.v1.get_variable(name='bc1', dtype=tf.dtypes.float64, initializer=np.load(args[8])),
        'bc2': tf.compat.v1.get_variable(name='bc2', dtype=tf.dtypes.float64, initializer=np.load(args[9])),
        'bc3': tf.compat.v1.get_variable(name='bc3', dtype=tf.dtypes.float64, initializer=np.load(args[10])),
        'bc4': tf.compat.v1.get_variable(name='bc4', dtype=tf.dtypes.float64, initializer=np.load(args[11])),
        'bc5': tf.compat.v1.get_variable(name='bc5', dtype=tf.dtypes.float64, initializer=np.load(args[12])),
        'bc6': tf.compat.v1.get_variable(name='bc6', dtype=tf.dtypes.float64, initializer=np.load(args[13]))}
    
    # read in images
    t1imgs, t1imgpaths = get_images(t1dir)
    numimgs = len(t1imgs)
    
    # parameters (MUST MATCH TRAINING PARAMETERS)
    kernel_size = 3
    num_nodes = 128
    
    def loss(predicted_t2, actual_t2):
        return tf.reduce_mean(tf.pow(actual_t2 - predicted_t2, 2))

    # get t2s
    t2pred = cnn(t1imgs.reshape((numimgs, 208, 300, 320, 1)), weights, biases, numimgs)
    
    # save out images
    for i in range(numimgs):
        # get image affine matrix
        t1img = nib.load(t1imgpaths[i])
        img = nib.Nifti1Image(t2pred[i, :, :, :, :], t1img.affine)
        nib.save(outdir + '/' + os.path.basename(os.path.splitext(t1imgpaths[i])[0]) + '_t2.nii.gz')




if __name__ == "__main__":
    main(sys.argv[1:])

