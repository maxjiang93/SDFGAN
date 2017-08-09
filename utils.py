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
    z_sample = np.random.uniform(-1, 1, size=(config.batch_size,sdfgan.z_dim))
    samples = sess.run(sdfgan.sampler, feed_dict={sdfgan.z: z_sample})
    fname = os.path.join(config.sample_dir, "samples.npy")
    np.save(fname, samples)
