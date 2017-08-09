"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import os
import pprint
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def create_sdfgan_samples(sess, sdfgan, config):
    z_sample = np.random.uniform(-1, 1, size=(config.sample_num,sdfgan.z_dim))
    samples = sess.run(sdfgan.sampler, feed_dict={sdfgan.z: z_sample})
    fname = os.path.join(config.sample_dir, "sdfgan_final_samples.npy")
    np.save(fname, samples)
    
    return fname

def create_pix2pix_samples(sess, pix2pix, config):
    sample_in = np.load(config.test_input_path).astype(np.float32)
    if len(sample_in.shape) == 4:
        sample_in = np.expand_dims(sample_in, axis=-1)
    samples = sess.run(pix2pix.sampler, feed_dict={pix2pix.sample_inputs: sample_in})
    fname = os.path.join(config.sample_dir, 'test_{:05d}.npy'.format(counter))
    print("Writing test results to " + fname)
    np.save(fname, samples)
