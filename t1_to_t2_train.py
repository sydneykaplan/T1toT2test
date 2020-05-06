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
    return imgs


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
        print("Usage:\t./t1_to_t2_train.py <t1dir> <t2dir> <outdir>")
        print("Usage:\t./t1_to_t2_train.py <t1dir> <t2dir> <outdir> <wc1.npy> <wc2.npy> <wc3.npy> <wc4.npy> <wc5.npy> <wc6.npy> <bc1.npy> <bc2.npy> <bc3.npy> <bc4.npy> <bc5.npy> <bc6.npy>")
        exit()
    
    # get input arguments
    t1dir = args[0]
    t2dir = args[1]
    outdir = args[2]
    initfile = 0
    # preset weights if file provided
    if len(args) > 3:
        initfile = 1
        weights = {
            'wc1': tf.compat.v1.get_variable(name='wc1', dtype=tf.dtypes.float64, initializer=np.load(args[3])),
            'wc2': tf.compat.v1.get_variable(name='wc2', dtype=tf.dtypes.float64, initializer=np.load(args[4])),
            'wc3': tf.compat.v1.get_variable(name='wc3', dtype=tf.dtypes.float64, initializer=np.load(args[5])),
            'wc4': tf.compat.v1.get_variable(name='wc4', dtype=tf.dtypes.float64, initializer=np.load(args[6])),
            'wc5': tf.compat.v1.get_variable(name='wc5', dtype=tf.dtypes.float64, initializer=np.load(args[7])),
            'wc6': tf.compat.v1.get_variable(name='wc6', dtype=tf.dtypes.float64, initializer=np.load(args[8]))}

        biases = {
            'bc1': tf.compat.v1.get_variable(name='bc1', dtype=tf.dtypes.float64, initializer=np.load(args[9])),
            'bc2': tf.compat.v1.get_variable(name='bc2', dtype=tf.dtypes.float64, initializer=np.load(args[10])),
            'bc3': tf.compat.v1.get_variable(name='bc3', dtype=tf.dtypes.float64, initializer=np.load(args[11])),
            'bc4': tf.compat.v1.get_variable(name='bc4', dtype=tf.dtypes.float64, initializer=np.load(args[12])),
            'bc5': tf.compat.v1.get_variable(name='bc5', dtype=tf.dtypes.float64, initializer=np.load(args[13])),
            'bc6': tf.compat.v1.get_variable(name='bc6', dtype=tf.dtypes.float64, initializer=np.load(args[14]))}
    
    # read in images
    t1imgs = get_images(t1dir)
    t2imgs = get_images(t2dir)
    numimgs = t1imgs.shape[4]
    
    # training parameters
    learning_rate = 0.001
    batch_size = 2
    num_steps = math.ceil(numimgs / batch_size) * 100 # roughly 100 epochs
    display_step = 100
    kernel_size = 3
    num_nodes = 128
    
    # if weights not provided
    if initfile == 0:
        # initialize model
        weights = {
            'wc1': tf.Variable(tf.random.truncated_normal([kernel_size, kernel_size, kernel_size, 1, num_nodes], dtype=tf.dtypes.float64), name='wc1'),
            'wc2': tf.Variable(tf.random.truncated_normal([kernel_size, kernel_size, kernel_size, num_nodes, num_nodes], dtype=tf.dtypes.float64), name='wc2'),
            'wc3': tf.Variable(tf.random.truncated_normal([kernel_size, kernel_size, kernel_size, num_nodes, num_nodes], dtype=tf.dtypes.float64), name='wc3'),
            'wc4': tf.Variable(tf.random.truncated_normal([kernel_size, kernel_size, kernel_size, num_nodes, num_nodes], dtype=tf.dtypes.float64), name='wc4'),
            'wc5': tf.Variable(tf.random.truncated_normal([kernel_size, kernel_size, kernel_size, num_nodes, num_nodes], dtype=tf.dtypes.float64), name='wc5'),
            'wc6': tf.Variable(tf.random.truncated_normal([kernel_size, kernel_size, kernel_size, 1, num_nodes], dtype=tf.dtypes.float64), name='wc6')}

        biases = {
            'bc1': tf.Variable(tf.random.truncated_normal([num_nodes], dtype=tf.dtypes.float64), name='bc1'),
            'bc2': tf.Variable(tf.random.truncated_normal([num_nodes], dtype=tf.dtypes.float64), name='bc2'),
            'bc3': tf.Variable(tf.random.truncated_normal([num_nodes], dtype=tf.dtypes.float64), name='bc3'),
            'bc4': tf.Variable(tf.random.truncated_normal([num_nodes], dtype=tf.dtypes.float64), name='bc4'),
            'bc5': tf.Variable(tf.random.truncated_normal([num_nodes], dtype=tf.dtypes.float64), name='bc5'),
            'bc6': tf.Variable(tf.random.truncated_normal([1], dtype=tf.dtypes.float64), name='bc6')}

    # setup training
    var_list = [weights['wc1'], weights['wc2'], weights['wc3'], weights['wc4'], weights['wc5'], weights['wc6'], biases['bc1'], biases['bc2'], biases['bc3'], biases['bc4'], biases['bc5'], biases['bc6']]
    optimizer = tf.optimizers.Adam(learning_rate)

    def loss(predicted_t2, actual_t2):
        return tf.reduce_mean(tf.pow(actual_t2 - predicted_t2, 2))

    @tf.function
    def train(model, t1in, t2in, var_list):
        with tf.GradientTape() as tape:
            current_loss = loss(cnn(t1in, weights, biases, batch_size, num_nodes), t2in)
        grads = tape.gradient(current_loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))
        return current_loss
    
    # start tensorflow session
    for i in range(1, num_steps+1):
        idxs = np.random.permutation(numimgs)[0:batch_size]
        batch_t1 = t1imgs[:, :, :, :, idxs]
        batch_t2 = t2imgs[:, :, :, :, idxs]
        # reshape array
        batch_t1 = batch_t1.reshape((batch_size, 208, 300, 320, 1))
        batch_t2 = batch_t2.reshape((batch_size, 208, 300, 320, 1))
        # training
        lossval = train(cnn, batch_t1, batch_t2, var_list)
        if i % display_step == 0 or i == 1:
            print('Step %i: batch loss %f' % (i, lossval))
            np.save(outdir + '/wc1', weights['wc1'].numpy())
            np.save(outdir + '/wc2', weights['wc2'].numpy())
            np.save(outdir + '/wc3', weights['wc3'].numpy())
            np.save(outdir + '/wc4', weights['wc4'].numpy())
            np.save(outdir + '/wc5', weights['wc5'].numpy())
            np.save(outdir + '/wc6', weights['wc6'].numpy())
            np.save(outdir + '/bc1', biases['bc1'].numpy())
            np.save(outdir + '/bc2', biases['bc2'].numpy())
            np.save(outdir + '/bc3', biases['bc3'].numpy())
            np.save(outdir + '/bc4', biases['bc4'].numpy())
            np.save(outdir + '/bc5', biases['bc5'].numpy())
            np.save(outdir + '/bc6', biases['bc6'].numpy())
            
    # save out weights
    np.save(outdir + '/wc1', weights['wc1'].numpy())
    np.save(outdir + '/wc2', weights['wc2'].numpy())
    np.save(outdir + '/wc3', weights['wc3'].numpy())
    np.save(outdir + '/wc4', weights['wc4'].numpy())
    np.save(outdir + '/wc5', weights['wc5'].numpy())
    np.save(outdir + '/wc6', weights['wc6'].numpy())
    np.save(outdir + '/bc1', biases['bc1'].numpy())
    np.save(outdir + '/bc2', biases['bc2'].numpy())
    np.save(outdir + '/bc3', biases['bc3'].numpy())
    np.save(outdir + '/bc4', biases['bc4'].numpy())
    np.save(outdir + '/bc5', biases['bc5'].numpy())
    np.save(outdir + '/bc6', biases['bc6'].numpy())


if __name__ == "__main__":
    main(sys.argv[1:])

