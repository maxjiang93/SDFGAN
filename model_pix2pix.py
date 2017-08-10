from __future__ import division
import time
from glob import glob
import math

from ops import *
from utils import *

EPS = 1e-12


class Pix2Pix(object):
    def __init__(self, sess, image_depth=64, image_height=64, image_width=64,
                 batch_size=64, sample_num=64, gf_dim=64, df_dim=64, gan_weight=1, l1_weight=100,
                 c_dim=1, dataset_name='shapenet_freqsplit', num_gpus=1, save_interval=200,
                 input_fname_pattern='*.npy', checkpoint_dir=None, dataset_dir=None, log_dir=None, sample_dir=None):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.image_depth = image_depth
        self.image_height = image_height
        self.image_width = image_width

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.c_dim = c_dim
        self.save_inverval = save_interval

        self.gan_weight = gan_weight
        self.l1_weight = l1_weight

        self.num_gpus = num_gpus
        self.glob_batch_size = self.num_gpus * self.batch_size

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):

        image_dims = [self.image_depth, self.image_height, self.image_width, self.c_dim]

        # input placeholders
        self.inputs = tf.placeholder(
            tf.float32, [self.glob_batch_size] + image_dims, name='input_images')
        self.targets = tf.placeholder(
            tf.float32, [self.glob_batch_size] + image_dims, name='target_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        self.n_eff = tf.placeholder(tf.int32, name='n_eff')  # overall number of effective data points

        # initialize global lists
        self.G = [None] * self.num_gpus
        self.D = [None] * self.num_gpus
        self.D_ = [None] * self.num_gpus
        self.d_losses = [None] * self.num_gpus
        self.g_losses = [None] * self.num_gpus
        self.g_losses_gan = [None] * self.num_gpus
        self.g_losses_l1 = [None] * self.num_gpus
        self.n_effs = [None] * self.num_gpus

        # compute using multiple gpus
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for gpuid in xrange(self.num_gpus):
                with tf.device('/gpu:%d' % gpuid):

                    # range of data for this gpu
                    gpu_start = gpuid * self.batch_size
                    gpu_end = (gpuid + 1) * self.batch_size

                    # number of effective data points
                    gpu_n_eff = tf.reduce_min([tf.reduce_max([0, self.n_eff - gpu_start]), self.batch_size])

                    # create examples and pass through discriminator
                    gpu_G = self.generator(self.inputs[gpu_start:gpu_end])
                    gpu_D = self.discriminator(self.inputs[gpu_start:gpu_end], self.targets[gpu_start:gpu_end])  # real
                    gpu_D_ = self.discriminator(self.inputs[gpu_start:gpu_end], gpu_G, reuse=True)  # fake pairs

                    # compute discriminator loss
                    gpu_d_loss = tf.reduce_mean(-(tf.log(gpu_D[:gpu_n_eff] + EPS)
                                                  + tf.log(1 - gpu_D_[:gpu_n_eff] + EPS)))

                    # compute generator loss
                    gpu_g_loss_gan = tf.reduce_mean(-tf.log(gpu_D[:gpu_n_eff] + EPS))
                    gpu_g_loss_l1 = tf.reduce_mean(tf.abs(self.targets[gpu_start:gpu_end] - gpu_G))
                    gpu_g_loss = gpu_g_loss_gan * self.gan_weight + gpu_g_loss_l1 * self.l1_weight

                    # add gpu-wise data to global list
                    self.G[gpuid] = gpu_G
                    self.D[gpuid] = gpu_D
                    self.D_[gpuid] = gpu_D_
                    self.d_losses[gpuid] = gpu_d_loss
                    self.g_losses[gpuid] = gpu_g_loss
                    self.g_losses_gan[gpuid] = gpu_g_loss_gan
                    self.g_losses_l1[gpuid] = gpu_g_loss_l1
                    self.n_effs[gpuid] = gpu_n_eff

                    # Reuse variables for the next gpu
                    tf.get_variable_scope().reuse_variables()

        # concatenate across GPUs
        self.D = tf.concat(self.D, axis=0)
        self.D_ = tf.concat(self.D_, axis=0)
        self.G = tf.concat(self.G, axis=0)
        weighted_d_loss = [self.d_losses[j] * tf.cast(self.n_effs[j], tf.float32)
                           / tf.cast(self.n_eff, tf.float32) for j in range(self.num_gpus)]
        weighted_g_loss = [self.g_losses[j] * tf.cast(self.n_effs[j], tf.float32)
                           / tf.cast(self.n_eff, tf.float32) for j in range(self.num_gpus)]
        weighted_g_loss_gan = [tf.cast(self.g_losses_gan[j], tf.float32) * tf.cast(self.n_effs[j], tf.float32)
                                 / tf.cast(self.n_eff, tf.float32) for j in range(self.num_gpus)]
        weighted_g_loss_l1 = [tf.cast(self.g_losses_l1[j], tf.float32) * tf.cast(self.n_effs[j], tf.float32)
                                 / tf.cast(self.n_eff, tf.float32) for j in range(self.num_gpus)]

        self.d_loss = tf.reduce_sum(weighted_d_loss, axis=0)
        self.g_loss = tf.reduce_sum(weighted_g_loss, axis=0)
        self.g_loss_gan = tf.reduce_sum(weighted_g_loss_gan, axis=0)
        self.g_loss_l1 = tf.reduce_sum(weighted_g_loss_l1, axis=0)

        # summarize variables
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.g_sum = image_summary("G", self.G[:, int(self.image_depth / 2), :, :])

        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.g_loss_gan_sum = scalar_summary("g_loss_gan", self.g_loss_gan)
        self.g_loss_l1_sum = scalar_summary("g_loss_l1", self.g_loss_l1)

        # define trainable variables for generator and discriminator
        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.sampler = self.sampler(self.sample_inputs)
        self.saver = tf.train.Saver()

    def train(self, config):
        """Train SFDGAN"""
        data = glob(os.path.join(self.dataset_dir, config.dataset, self.input_fname_pattern))
        np.random.shuffle(data)

        # define optimization operation
        d_opt = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1)
        g_opt = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1)

        # create list of grads from different gpus
        global_d_grads_vars = [None] * self.num_gpus
        global_g_grads_vars = [None] * self.num_gpus

        # compute d gradients
        with tf.variable_scope(tf.get_variable_scope()):
            for gpuid in xrange(self.num_gpus):
                with tf.device('/gpu:%d' % gpuid):
                    gpu_d_grads_vars = d_opt.compute_gradients(loss=self.d_losses[gpuid], var_list=self.d_vars)
                    global_d_grads_vars[gpuid] = gpu_d_grads_vars

        # compute g gradients
        with tf.variable_scope(tf.get_variable_scope()):
            for gpuid in xrange(self.num_gpus):
                with tf.device('/gpu:%d' % gpuid):
                    gpu_g_grads_vars = g_opt.compute_gradients(loss=self.g_losses[gpuid], var_list=self.g_vars)
                    global_g_grads_vars[gpuid] = gpu_g_grads_vars

        # average gradients across gpus and apply gradients
        d_grads_vars = average_gradients(global_d_grads_vars)
        g_grads_vars = average_gradients(global_g_grads_vars)
        d_optim = d_opt.apply_gradients(d_grads_vars)
        g_optim = g_opt.apply_gradients(g_grads_vars)

        # compatibility across tf versions
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.d__sum, self.g_loss_gan_sum, self.g_loss_l1_sum, self.g_loss_sum, self.g_sum])
        self.d_sum = merge_summary([self.d_sum, self.d_loss_sum])

        self.writer = SummaryWriter(self.log_dir, self.sess.graph)
        sample_files = data[0:self.sample_num]

        sample_inputs = [np.load(sample_file)[0, :, :, :] for sample_file in sample_files]
        sample_targets = [np.load(sample_file)[1, :, :, :] for sample_file in sample_files]
        sample_in = np.array(sample_inputs).astype(np.float32)[:, :, :, :, None]
        sample_tg = np.array(sample_targets).astype(np.float32)[:, :, :, :, None]

        counter = 0
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        batch_idxs = int(math.ceil(min(len(data), config.train_size) / self.glob_batch_size)) - 1
        total_steps = config.epoch * batch_idxs
        prev_time = -np.inf

        for epoch in xrange(config.epoch):
            # shuffle data before training in each epoch
            np.random.shuffle(data)
            for idx in xrange(0, batch_idxs):
                glob_batch_files = data[idx * self.glob_batch_size:(idx + 1) * self.glob_batch_size]
                glob_batch_inputs = [
                    np.load(batch_file)[0, :, :, :] for batch_file in glob_batch_files]
                glob_batch_targets = [
                    np.load(batch_file)[1, :, :, :] for batch_file in glob_batch_files]
                glob_batch_in = np.array(glob_batch_inputs).astype(np.float32)[:, :, :, :, None]
                glob_batch_tg = np.array(glob_batch_targets).astype(np.float32)[:, :, :, :, None]

                n_eff = len(glob_batch_files)

                # Pad zeros if effective batch size is smaller than global batch size
                if n_eff != self.glob_batch_size:
                    glob_batch_in = pad_glob_batch(glob_batch_in, self.glob_batch_size)
                    glob_batch_tg = pad_glob_batch(glob_batch_tg, self.glob_batch_size)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.inputs: glob_batch_in,
                                                          self.targets: glob_batch_tg,
                                                          self.n_eff: n_eff})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.inputs: glob_batch_in,
                                                          self.targets: glob_batch_tg,
                                                          self.n_eff: n_eff})
                self.writer.add_summary(summary_str, counter)

                # Compute last batch accuracy and losses
                lossD, lossG = self.sess.run([self.d_loss, self.g_loss],
                                             feed_dict={self.inputs: glob_batch_in,
                                                        self.targets: glob_batch_tg,
                                                        self.n_eff: n_eff})

                # get time
                now_time = time.time()
                time_per_iter = now_time - prev_time
                prev_time = now_time
                eta = (total_steps - counter + checkpoint_counter) * time_per_iter
                counter += 1

                try:
                    timestr = time.strftime("%H:%M:%S", time.gmtime(eta))
                except:
                    timestr = '?:?:?'

                print("Epoch:[%3d] [%3d/%3d] Iter:[%5d] eta(h:m:s): %s, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, idx, batch_idxs, counter, timestr, lossD, lossG))

                # save checkpoint and samples every save_interval steps
                if np.mod(counter, self.save_inverval) == 1:
                    sample_gen = self.sess.run(self.sampler, feed_dict={self.sample_inputs: sample_in})
                    sample = np.concatenate((np.expand_dims(sample_in, axis=0),  # sample_num x 64 x 64 x 64 x 1
                                             np.expand_dims(sample_tg, axis=0),  # sample_num x 64 x 64 x 64 x 1
                                             np.expand_dims(sample_gen, axis=0)), axis=0)  # sample_num x 64 x 64 x 64 x 1
                    np.save(self.sample_dir+'/sample_{:05d}.npy'
                            .format(counter), sample)
                    print("[Sample] Iter {0}, saving sample size of {1}, saving checkpoint."
                          .format(counter, self.sample_num))
                    self.save(config.checkpoint_dir, counter)

        # save last checkpoint
        sample_gen = self.sess.run(self.sampler, feed_dict={self.sample_inputs: sample_in})
        sample = np.concatenate((np.expand_dims(sample_in[:, :, :, :, 0], axis=0),  # sample_num x 64 x 64 x 64
                                 np.expand_dims(sample_tg[:, :, :, :, 0], axis=0),  # sample_num x 64 x 64 x 64
                                 np.expand_dims(sample_gen[:, :, :, :, 0], axis=0)), axis=0)  # sample_num x 64 x 64 x 64
        np.save(self.sample_dir+'/sample_{:05d}.npy'
                .format(counter), sample)
        print("[Sample] Iter {0}, saving sample size of {1}, saving checkpoint."
              .format(counter, self.sample_num))
        self.save(config.checkpoint_dir, counter)
        
        print("[!] Training of Pix2Pix Network Complete.")

    def discriminator(self, discrim_inputs, discrim_targets, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            n_layers = 3
            layers = []

            # 2x [batch, depth, height, width, in_channels] => [batch, depth, height, width, in_channels * 2]
            input = tf.concat([discrim_inputs, discrim_targets], axis=-1)

            # layer_1: [batch, 64, 64, 64, in_channels * 2] => [batch, 32, 32, 32, df_dim]
            convolved = conv3d(input, self.df_dim, name='d_h0_conv')
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

            # layer_2: [batch, 32, 32, 32, df_dim] => [batch, 16, 16, 16, df_dim * 2]
            # layer_3: [batch, 16, 16, 16, df_dim * 2] => [batch, 8, 8, 8, df_dim * 4]
            # layer_4: [batch, 8, 8, 8, df_dim * 4] => [batch, 7, 7, 7, df_dim * 8]
            for i in range(n_layers):
                # with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = self.df_dim * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv3d(layers[-1], out_channels,
                                   d_d=stride, d_h=stride, d_w=stride, name='d_h%d_conv' % (len(layers)))
                normalized = batchnorm(convolved, name='d_h%d_bn' % (len(layers)))
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

            # layer_5: [batch, 7, 7, df_dim * 8] => [batch, 6, 6, 1]
            # with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv3d(rectified, output_dim=1, d_d=1, d_h=1, d_w=1, name='d_h%d_conv' % (len(layers)))
            output = tf.sigmoid(convolved)
            layers.append(output)

            return layers[-1]

    def generator(self, generator_inputs):
        with tf.variable_scope("generator") as scope:
            layers = []

            # encoder_1: [batch, 64, 64, 64, in_channels] => [batch, 32, 32, gf_dim]
            output = conv3d(generator_inputs, self.gf_dim, name="g_enc_conv_0")
            layers.append(output)

            layer_specs = [
                self.gf_dim * 2,  # encoder_2: [batch, 32, 32, 32, gf_dim] => [batch, 16, 16, 16, gf_dim * 2]
                self.gf_dim * 4,  # encoder_3: [batch, 16, 16, 16, gf_dim * 2] => [batch, 8, 8, 8, gf_dim * 4]
                self.gf_dim * 8,  # encoder_4: [batch, 8, 8, 8, gf_dim * 2] => [batch, 4, 4, 4, gf_dim * 4]
                self.gf_dim * 8,  # encoder_5: [batch, 4, 4, 4, gf_dim * 8] => [batch, 2, 2, 2, gf_dim * 8]
                self.gf_dim * 8,  # encoder_6: [batch, 2, 2, 2, gf_dim * 8] => [batch, 1, 1, 1, gf_dim * 8]
            ]

            for out_channels in layer_specs:
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv3d(rectified, out_channels, name="g_enc_cov_%d" % (len(layers)))
                output = batchnorm(convolved, name="g_enc_bn_%d" % (len(layers)))
                layers.append(output)

            layer_specs = [
                (self.gf_dim * 8, 0.5),  # decoder_6: [batch,1,1,1,gf_dim * 8] => [batch,2,2,2,gf_dim * 8 * 2]
                (self.gf_dim * 8, 0.5),  # decoder_5: [batch,2,2,2,gf_dim * 8 * 2] => [batch,4 4,4,gf_dim * 8 * 2]
                (self.gf_dim * 4, 0.5),  # decoder_4: [batch,4,4,4,gf_dim * 8 * 2] => [batch,8,8,8,gf_dim * 4 * 2]
                (self.gf_dim * 2, 0.0),  # decoder_3: [batch,8,8,8,gf_dim * 4 * 2] => [batch,16,16,16,gf_dim * 2 * 2]
                (self.gf_dim * 1, 0.0),  # decoder_2: [batch,16,16,16,gf_dim * 2 * 2] => [batch,32,32,32,gf_dim * 2]
            ]

            num_encoder_layers = len(layers)
            for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=-1)

                rectified = tf.nn.relu(input)
                # [batch, in_depth, in_height, in_width, in_channels]
                # => [batch, in_depth, in_height*2, in_width*2, out_channels]
                in_dims = rectified.get_shape().as_list()
                out_dims = [in_dims[0], in_dims[1] * 2, in_dims[2] * 2, in_dims[3] * 2, out_channels]
                output = deconv3d(rectified, out_dims, name="g_dec_conv_%d" % skip_layer)
                output = batchnorm(output, name="g_dec_bn_%d" % skip_layer)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

            # decoder_1: [batch, 32, 32, 32, gf_dim * 2] => [batch, 64, 64, 64, 1]
            input = tf.concat([layers[-1], layers[0]], axis=-1)
            rectified = tf.nn.relu(input)
            in_dims = rectified.get_shape().as_list()
            out_dims = [in_dims[0], in_dims[1] * 2, in_dims[2] * 2, in_dims[3] * 2, 1]
            output = deconv3d(rectified, out_dims, name="g_dec_conv_0")
            output = tf.tanh(output)
            layers.append(output)

            return layers[-1]

    def sampler(self, generator_inputs):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            layers = []

            # encoder_1: [batch, 64, 64, 64, in_channels] => [batch, 32, 32, gf_dim]
            output = conv3d(generator_inputs, self.gf_dim, name="g_enc_conv_0")
            layers.append(output)

            layer_specs = [
                self.gf_dim * 2,  # encoder_2: [batch, 32, 32, 32, gf_dim] => [batch, 16, 16, 16, gf_dim * 2]
                self.gf_dim * 4,  # encoder_3: [batch, 16, 16, 16, gf_dim * 2] => [batch, 8, 8, 8, gf_dim * 4]
                self.gf_dim * 8,  # encoder_4: [batch, 8, 8, 8, gf_dim * 2] => [batch, 4, 4, 4, gf_dim * 4]
                self.gf_dim * 8,  # encoder_5: [batch, 4, 4, 4, gf_dim * 8] => [batch, 2, 2, 2, gf_dim * 8]
                self.gf_dim * 8,  # encoder_6: [batch, 2, 2, 2, gf_dim * 8] => [batch, 1, 1, 1, gf_dim * 8]
            ]

            for out_channels in layer_specs:
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv3d(rectified, out_channels, name="g_enc_cov_%d" % (len(layers)))
                output = batchnorm(convolved, name="g_enc_bn_%d" % (len(layers)))
                layers.append(output)

            layer_specs = [
                (self.gf_dim * 8, 0.5),  # decoder_6: [batch,1,1,1,gf_dim * 8] => [batch,2,2,2,gf_dim * 8 * 2]
                (self.gf_dim * 8, 0.5),  # decoder_5: [batch,2,2,2,gf_dim * 8 * 2] => [batch,4 4,4,gf_dim * 8 * 2]
                (self.gf_dim * 4, 0.5),  # decoder_4: [batch,4,4,4,gf_dim * 8 * 2] => [batch,8,8,8,gf_dim * 4 * 2]
                (self.gf_dim * 2, 0.0),  # decoder_3: [batch,8,8,8,gf_dim * 4 * 2] => [batch,16,16,16,gf_dim * 2 * 2]
                (self.gf_dim * 1, 0.0),  # decoder_2: [batch,16,16,16,gf_dim * 2 * 2] => [batch,32,32,32,gf_dim * 2]
            ]

            num_encoder_layers = len(layers)
            for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
                skip_layer = num_encoder_layers - decoder_layer - 1
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=-1)

                rectified = tf.nn.relu(input)
                # [batch, in_depth, in_height, in_width, in_channels]
                # => [batch, in_depth, in_height*2, in_width*2, out_channels]
                in_dims = rectified.get_shape().as_list()
                out_dims = [in_dims[0], in_dims[1] * 2, in_dims[2] * 2, in_dims[3] * 2, out_channels]
                output = deconv3d(rectified, out_dims, name="g_dec_conv_%d" % skip_layer)
                output = batchnorm(output, name="g_dec_bn_%d" % skip_layer)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

            # decoder_1: [batch, 32, 32, 32, gf_dim * 2] => [batch, 64, 64, 64, 1]
            input = tf.concat([layers[-1], layers[0]], axis=-1)
            rectified = tf.nn.relu(input)
            in_dims = rectified.get_shape().as_list()
            out_dims = [in_dims[0], in_dims[1] * 2, in_dims[2] * 2, in_dims[3] * 2, 1]
            output = deconv3d(rectified, out_dims, name="g_dec_conv_0")
            output = tf.tanh(output)
            layers.append(output)

            return layers[-1]

    @property
    def model_dir(self):
        return "Pix2Pix" + "{}_{}_{}_{}_{}".format(
            self.dataset_name.replace("_freqsplit", ""), self.batch_size,
            self.image_depth, self.image_height, self.image_width)

    def save(self, checkpoint_dir, step):
        model_name = "SDFGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
