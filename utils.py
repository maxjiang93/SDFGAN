"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import os
import pprint
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def create_samples(sess, pix2pix, config):
    z_sample = np.random.uniform(-1, 1, size=(config.batch_size,pix2pix.z_dim))
    samples = sess.run(pix2pix.sampler, feed_dict={pix2pix.z: z_sample})
    fname = os.path.join(config.sample_dir, "samples.npy")
    np.save(fname, samples)
