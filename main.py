from model_sdfgan import SDFGAN
from model_pix2pix import Pix2Pix
from utils import *
from ops import batch_lowpass
import shutil

import tensorflow as tf

# common training flags
flags = tf.app.flags
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_string("model", "sdfgan", "Model to train. Choice of 'sdfgan' or 'pix2pix' [sdfgan]")
flags.DEFINE_boolean("is_new", False, "True for training from scratch, deleting original file [False]")
flags.DEFINE_integer("epoch", 100, "Epoch to train [100]")
flags.DEFINE_float("d_learning_rate", 0.0002, "Learning rate of discrim for adam [0.0002]")
flags.DEFINE_float("g_learning_rate", 0.0005, "Learning rate of gen for adam [0.0005]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("image_depth", 64, "The size of sdf field to use. [64]")
flags.DEFINE_integer("image_height", None, "The size of sdf to use. If None, same value as image_depth [None]")
flags.DEFINE_integer("image_width", None, "The size of sdf to use. If None, same value as image_depth [None]")
flags.DEFINE_integer("c_dim", 1, "Number of channels. [1]")
flags.DEFINE_string("dataset", "shapenet", "The name of dataset [shapenet]")
flags.DEFINE_string("input_fname_pattern", "*.npy", "Glob pattern of filename of input sdf [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("dataset_dir", "data", "Directory name to read the input training data [data]")
flags.DEFINE_string("log_dir", "logs", "Directory name to save the log files [logs]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs to use [1]")
flags.DEFINE_integer("random_seed", 1, "Random seed [1]")
flags.DEFINE_integer("save_inverval", 200, "Interval in steps for saving model and samples [200]")

# pix2pix specific training flags
flags.DEFINE_integer("gan_weight", 1, "GAN weight in generator loss function. [1]")
flags.DEFINE_integer("l1_weight", 100, "L1 weight in generator loss function. [100]")

# testing flags
flags.DEFINE_integer("sample_num", 16, "Number of samples. [16]")
flags.DEFINE_string("test_from_input_path", None, "Path to test inputs. [None]")

FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.image_height is None:
        FLAGS.image_height = FLAGS.image_depth
    if FLAGS.image_width is None:
        FLAGS.image_width = FLAGS.image_depth

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
        if not (FLAGS.model == 'pix2pix' and FLAGS.is_train):
            sdfgan = SDFGAN(
                sess,
                image_depth=FLAGS.image_depth,
                image_height=FLAGS.image_height,
                image_width=FLAGS.image_width,
                batch_size=FLAGS.batch_size,
                sample_num=FLAGS.batch_size,
                c_dim=FLAGS.c_dim,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                checkpoint_dir=FLAGS.checkpoint_dir,
                dataset_dir=FLAGS.dataset_dir,
                log_dir=FLAGS.log_dir,
                sample_dir=FLAGS.sample_dir,
                num_gpus=FLAGS.num_gpus,
                save_interval=FLAGS.save_interval)
       
            show_all_variables()
        
        if not (FLAGS.model == 'sdfgan' and FLAGS.is_train):
            pix2pix = Pix2Pix(
                sess,
                image_depth=FLAGS.image_depth,
                image_height=FLAGS.image_height,
                image_width=FLAGS.image_width,
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
                num_gpus=FLAGS.num_gpus,
                save_interval=FLAGS.save_interval)

            show_all_variables()
                
        if FLAGS.model == 'sdfgan' and FLAGS.is_train:
            os.makedirs(os.path.join(FLAGS.sample_dir, "sdfgan_sample"))
            os.makedirs(os.path.join(FLAGS.log_dir, "sdfgan_log"))    
            sdfgan.train(FLAGS)
                
        elif FLAGS.model == 'pix2pix' and FLAGS.is_train:
            os.makedirs(os.path.join(FLAGS.sample_dir, "pix2pix_sample"))
            os.makedirs(os.path.join(FLAGS.log_dir, "pix2pix_log"))
            pix2pix.train(FLAGS)
                
        elif not FLAGS.is_train:  # test mode
            # load pix2pix network
            if not pix2pix.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Could not load Pix2Pix model. Train first, then run test mode.")
                
            # load sdfgan network if necessary
            if FLAGS.test_from_input_path is None:
                # generate a sample from sdfgan first
                if not sdfgan.load(FLAGS.checkpoint_dir):
                    raise Exception("[!] Could not load SDFGAN model. Train first, then run test mode.")
                FLAGS.test_from_input_path = create_sdfgan_samples(sess, sdfgan, FLAGS)
            
            # postprocess samples (low-pass filter)
            sdf = np.squeeze(np.load(FLAGS.test_from_input_path), axis=-1)
            sdf_lf = batch_lowpass(sdf)

            # create and save final samples
            sdf_hf = create_pix2pix_samples(sess, pix2pix, sdf_lf)
            sdf_all = sdf_lf + sdf_hf
            sdf_save = np.array([sdf_all, sdf_lf, sdf_hf])
            fname = os.path.join(FLAGS.sample_dir, "full_final_samples.npy")
            np.save(fname, sdf_save)
            print("Saving final full samples to {0}, "
                  "shape: {1} (combine_freq/low_freq/hi_freq, sample_num, dim0, dim1, dim2)"
                  .format(fname, sdf_save.shape))

        else:
            raise Exception("[!] Model must be 'sdfgan' or 'pix2pix'.")

                
if __name__ == '__main__':
    tf.app.run()
