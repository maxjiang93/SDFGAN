import os
import numpy as np

from model import Pix2Pix
from utils import pp, show_all_variables, create_samples
import shutil

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("d_learning_rate", 0.0002, "Learning rate of discrim for adam [0.0002]")
flags.DEFINE_float("g_learning_rate", 0.0005, "Learning rate of gen for adam [0.0005]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_depth", 64, "The size of sdf field to use (will be center cropped). [64]")
flags.DEFINE_integer("input_height", None,
                     "The size of sdf to use (will be center cropped). If None, same value as input_depth [None]")
flags.DEFINE_integer("input_width", None,
                     "The size of sdf to use (will be center cropped). If None, same value as input_depth [None]")
flags.DEFINE_integer("output_depth", 64, "The size of the output sdf to produce [64]")
flags.DEFINE_integer("output_height", None,
                     "The size of the output images to produce. If None, same value as output_depth [None]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_depth [None]")
flags.DEFINE_integer("c_dim", 1, "Dimension of sdf. [1]")
flags.DEFINE_integer("sample_num", 16, "Number of samples. [16]")
flags.DEFINE_string("test_input", None, "Path to test inputs. [None]")
flags.DEFINE_integer("gan_weight", 1, "GAN weight in generator loss function. [1]")
flags.DEFINE_integer("l1_weight", 100, "L1 weight in generator loss function. [100]")
flags.DEFINE_string("dataset", "shapenet_freqsplit", "The name of dataset [shapenet]")
flags.DEFINE_string("input_fname_pattern", "*.npy", "Glob pattern of filename of input sdf [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("dataset_dir", "data", "Directory name to read the input training data [data]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save the log files [logs]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_new", False, "True for training from scratch, deleting original file [False]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("random_seed", 1, "Random seed. [1]")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs to use [1]")
flags.DEFINE_float("field_constraint", 0.1, "Coefficient for field constraint error [0.1]")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_height is None:
        FLAGS.input_height = FLAGS.input_depth
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_depth

    if FLAGS.output_height is None:
        FLAGS.output_height = FLAGS.output_depth
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_depth

    if FLAGS.is_new:
        if os.path.exists(FLAGS.checkpoint_dir):
            shutil.rmtree(FLAGS.checkpoint_dir)
        if os.path.exists(FLAGS.sample_dir):
            shutil.rmtree(FLAGS.sample_dir)
        if os.path.exists(FLAGS.log_dir):
            shutil.rmtree(FLAGS.log_dir)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    # set random seed
    tf.set_random_seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    with tf.Session(config=run_config) as sess:

        pix2pix = Pix2Pix(
            sess,
            input_depth=FLAGS.input_depth,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_depth=FLAGS.output_depth,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.sample_num,
            gan_weight=FLAGS.gan_weight,
            l1_weight=FLAGS.l1_weight,
            c_dim=FLAGS.c_dim,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            checkpoint_dir=FLAGS.checkpoint_dir,
            dataset_dir=FLAGS.dataset_dir,
            log_dir=FLAGS.log_dir,
            sample_dir=FLAGS.sample_dir,
            num_gpus=FLAGS.num_gpus)

        show_all_variables()
        if FLAGS.is_train:
            pix2pix.train(FLAGS)
        else:
            if not pix2pix.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")
            create_samples(sess, pix2pix, FLAGS)

if __name__ == '__main__':
    tf.app.run()
