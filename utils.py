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


def create_samples(sess, pix2pix, config, counter):
    sample_in = np.load(config.test_input_path).astype(np.float32)
    if len(sample_in.shape) == 4:
        sample_in = np.expand_dims(sample_in, axis=-1)
    samples = sess.run(pix2pix.sampler, feed_dict={pix2pix.sample_inputs: sample_in})
    fname = os.path.join(config.sample_dir, 'test_{:05d}.npy'.format(counter))
    print("Writing test results to " + fname)
    np.save(fname, samples)
