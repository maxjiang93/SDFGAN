"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import os
import pprint
import scipy.misc
import numpy as np
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def inverse_transform(images):
    return (images + 1.) / 2.


def create_samples(sess, sdfgan):
    # pick batch of training set shapes
    data = glob(os.path.join(sdfgan.dataset_dir, sdfgan.dataset_name, sdfgan.input_fname_pattern))
    np.random.shuffle(data)
    batch_files = data[:sdfgan.sample_num]
    batch = [
        np.load(batch_file)[0, :, :, :] for batch_file in batch_files]
    batch_images = np.array(batch).astype(np.float32)[:, :, :, :, None]
    batch_z = np.random.normal(0, 1, [sdfgan.batch_size, sdfgan.z_dim]) \
        .astype(np.float32)
    train_shapes, dec_shapes, rand_shapes = sess.run([sdfgan.train_shapes, sdfgan.dec_shapes, sdfgan.rand_shapes]
                                                     , feed_dict={sdfgan.inputs: batch_images,
                                                                  sdfgan.z: batch_z})
    fname_0 = os.path.join(sdfgan.sample_dir, "training_samples.npy")
    fname_1 = os.path.join(sdfgan.sample_dir, "decoder_samples.npy")
    fname_2 = os.path.join(sdfgan.sample_dir, "random_samples.npy")
    np.save(fname_0, train_shapes)
    np.save(fname_1, dec_shapes)
    np.save(fname_2, rand_shapes)


