from __future__ import division
import time

from ops import *
from utils import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class SDFGAN(object):
    def __init__(self, sess, input_depth=64, input_height=64, input_width=64, is_crop=True,
                 batch_size=64, sample_num=64,
                 output_depth=64, output_height=64, output_width=64, z_dim=200, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=1, dataset_name='shapenet',
                 input_fname_pattern='*.npy', checkpoint_dir=None, dataset_dir=None, log_dir=None, sample_dir=None,
                 num_gpus=1, field_constraint=0.1):
        """

        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width

        self.output_depth = output_depth
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        self.num_gpus = num_gpus
        self.glob_batch_size = self.num_gpus * self.batch_size
        self.field_constraint = field_constraint

        # batch normalization : deals with poor initialization helps gradient flow
        self.e_bn1 = batch_norm(name='e_bn1')
        self.e_bn2 = batch_norm(name='e_bn2')
        self.e_bn3 = batch_norm(name='e_bn3')

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):

        if self.is_crop:
            image_dims = [self.output_depth, self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.output_depth, self.input_height, self.input_width, self.c_dim]

        # input placeholders
        self.inputs = tf.placeholder(
            tf.float32, [self.glob_batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='real_samples')
        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.n_eff = tf.placeholder(tf.int32, name='n_eff')  # overall number of effective data points

        self.z_sum = histogram_summary("z", self.z)

        # initialize global lists
        self.I = [None] * self.num_gpus
        self.G = [None] * self.num_gpus
        self.D = [None] * self.num_gpus
        self.EG = [None] * self.num_gpus
        self.D_logits = [None] * self.num_gpus
        self.D_ = [None] * self.num_gpus
        self.D_logits_ = [None] * self.num_gpus
        self.d_loss_real = [None] * self.num_gpus
        self.d_loss_fake = [None] * self.num_gpus
        self.e_losses = [None] * self.num_gpus
        self.e_loss_kl = [None] * self.num_gpus
        self.e_loss_ll = [None] * self.num_gpus
        self.g_losses = [None] * self.num_gpus
        self.d_losses = [None] * self.num_gpus
        self.d_accus = [None] * self.num_gpus
        self.n_effs = [None] * self.num_gpus
        self.g_loss_eik = [None] * self.num_gpus
        self.g_loss_sym = [None] * self.num_gpus
        self.g_loss_gen = [None] * self.num_gpus

        # compute using multiple gpus
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for gpuid in xrange(self.num_gpus):
                with tf.device('/gpu:%d' % gpuid):

                    # range of data for this gpu
                    gpu_start = gpuid * self.batch_size
                    gpu_end = (gpuid + 1) * self.batch_size

                    # number of effective data points
                    gpu_n_eff = tf.reduce_min([tf.reduce_max([0, self.n_eff - gpu_start]), self.batch_size])

                    # encoder
                    eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1)  # normal dist for VAE
                    gpu_I = self.inputs[gpu_start:gpu_end]  # original inputs
                    z_x_mean, z_x_log_sigma_sq = self.encoder(gpu_I)  # get z from the input
                    z_x = tf.add(z_x_mean, tf.multiply(tf.sqrt(tf.exp(z_x_log_sigma_sq)), eps)) # grab our actual latent vec z
                    gpu_EG = self.generator(z_x)  # recover from autoencoder

                    # compatibility across different tf versions
                    def sigmoid_cross_entropy_with_logits(x, y):
                        try:
                            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
                        except:
                            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

                    # compute loss and accuracy
                    # clip values
                    gpu_kl_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + tf.clip_by_value(z_x_log_sigma_sq, -5.0, 5.0)
                                   - tf.square(tf.clip_by_value(z_x_mean, -5.0, 5.0))
                                   - tf.exp(tf.clip_by_value(z_x_log_sigma_sq, -5.0, 5.0)), 1))\
                                   / self.output_depth / self.output_height / self.output_width
                    # L2 Distance Metric
                    gpu_l2_loss = 0.1 * tf.reduce_sum(tf.square(gpu_EG - gpu_I)) \
                                  / self.output_depth / self.output_height / self.output_width

                    gpu_e_loss = gpu_kl_loss + gpu_l2_loss
                    gpu_g_loss = gpu_l2_loss

                    # compute generator field constraint loss
                    # enforce eikonal equation
                    delta_d, delta_h, delta_w = 1 / (np.array(image_dims[:-1]) - 1)
                    gpu_G_d1, gpu_G_d0 = gpu_G[2:, 1:-1, 1:-1], gpu_G[:-2, 1:-1, 1:-1]
                    gpu_G_h1, gpu_G_h0 = gpu_G[1:-1, 2:, 1:-1], gpu_G[1:-1, :-2, 1:-1]
                    gpu_G_w1, gpu_G_w0 = gpu_G[1:-1, 1:-1, 2:], gpu_G[1:-1, 1:-1, :-2]
                    grad_G_d = tf.expand_dims((gpu_G_d1 - gpu_G_d0) / 2 / delta_d, axis=0)
                    grad_G_h = tf.expand_dims((gpu_G_h1 - gpu_G_h0) / 2 / delta_h, axis=0)
                    grad_G_w = tf.expand_dims((gpu_G_w1 - gpu_G_w0) / 2 / delta_w, axis=0)
                    grad_G = tf.concat([grad_G_d, grad_G_h, grad_G_w], axis=0)
                    grad_G_norms = tf.norm(grad_G, axis=0)
                    grad_G_diff2 = tf.square(tf.reduce_mean(grad_G_norms)-tf.ones_like(grad_G_norms))
                    gpu_g_loss_eik = tf.abs(tf.reduce_mean(grad_G_diff2))

                    # enforce symmetry (along dim=2)
                    gpu_left = gpu_G[:, :, :int(image_dims[2] / 2)]
                    gpu_rite = tf.reverse(gpu_G[:, :, int(image_dims[2] / 2):], axis=[2])
                    gpu_g_loss_sym = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(gpu_left, gpu_rite))))

                    # combine losses for encoder, generator and discriminator)
                    gpu_e_loss = gpu_ll_loss + gpu_kl_loss
                    gpu_g_loss = gpu_g_loss_gen + (gpu_g_loss_eik + gpu_g_loss_sym) * self.field_constraint \
                                 + gpu_ll_loss
                    gpu_d_loss = gpu_d_loss

                    # add gpu-wise data to global list
                    self.I[gpuid] = gpu_I
                    self.EG[gpuid] = gpu_EG
                    self.G[gpuid] = gpu_G
                    self.D[gpuid] = gpu_D
                    self.D_[gpuid] = gpu_D_
                    self.D_logits[gpuid] = gpu_D_logits
                    self.D_logits_[gpuid] = gpu_D_logits_
                    self.e_losses[gpuid] = gpu_e_loss
                    self.e_loss_kl[gpuid] = gpu_kl_loss
                    self.e_loss_ll[gpuid] = gpu_ll_loss
                    self.d_loss_real[gpuid] = gpu_d_loss_real
                    self.d_loss_fake[gpuid] = gpu_d_loss_fake
                    self.d_losses[gpuid] = gpu_d_loss
                    self.g_loss_eik[gpuid] = gpu_g_loss_eik
                    self.g_loss_sym[gpuid] = gpu_g_loss_sym
                    self.g_loss_gen[gpuid] = gpu_g_loss_gen
                    self.g_losses[gpuid] = gpu_g_loss
                    self.d_accus[gpuid] = gpu_d_accu
                    self.n_effs[gpuid] = gpu_n_eff

                    # Reuse variables for the next gpu
                    tf.get_variable_scope().reuse_variables()

        # concatenate across GPUs
        self.I = tf.concat(self.I, axis=0)
        self.EG = tf.concat(self.EG, axis=0)
        self.D = tf.concat(self.D, axis=0)
        self.D_ = tf.concat(self.D_, axis=0)
        self.G = tf.concat(self.G, axis=0)

        def weighted_avg(quantity, cast=False):
            if cast:
                quantity = [tf.cast(quantity[j], tf.float32) for j in range(self.num_gpus)]
            result = [quantity[j] * tf.cast(self.n_effs[j], tf.float32)
                                / tf.cast(self.n_eff, tf.float32) for j in range(self.num_gpus)]
            return result

        self.e_loss = tf.reduce_sum(weighted_avg(self.e_losses), axis=0)
        self.e_loss_kl = tf.reduce_sum(weighted_avg(self.e_loss_kl), axis=0)
        self.e_loss_ll = tf.reduce_sum(weighted_avg(self.e_loss_ll), axis=0)
        self.d_loss_real = tf.reduce_sum(weighted_avg(self.d_loss_real), axis=0)
        self.d_loss_fake = tf.reduce_sum(weighted_avg(self.d_loss_fake), axis=0)
        self.d_loss = tf.reduce_sum(weighted_avg(self.d_losses), axis=0)
        self.g_loss_eik = tf.reduce_sum(weighted_avg(self.g_loss_eik, cast=True), axis=0)
        self.g_loss_sym = tf.reduce_sum(weighted_avg(self.g_loss_sym, cast=True), axis=0)
        self.g_loss_gen = tf.reduce_sum(weighted_avg(self.g_loss_gen, cast=True), axis=0)
        self.g_loss = tf.reduce_sum(weighted_avg(self.g_losses), axis=0)
        self.d_accu = tf.reduce_sum(weighted_avg(self.d_accus, cast=True), axis=0)

        # summarize variables
        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.g_sum = image_summary("G", self.G[:, 32, :, :])
        self.eg_sum = image_summary("EG", self.EG[:, 32, :, :])
        self.i_sum = image_summary("I", self.I[:, 32, :, :])

        self.e_loss_sum = scalar_summary("e_loss", self.e_loss)
        self.e_loss_kl_sum = scalar_summary("e_loss_kl", self.e_loss_kl)
        self.e_loss_ll_sum = scalar_summary("e_loss_ll", self.e_loss_ll)

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        self.g_loss_eik_sum = scalar_summary("g_loss_eik", self.g_loss_eik)
        self.g_loss_sym_sum = scalar_summary("g_loss_sym", self.g_loss_sym)
        self.g_loss_gen_sum = scalar_summary("g_loss_gen", self.g_loss_gen)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_accu_sum = scalar_summary("d_accu", self.d_accu)

        # define trainable variables for generator and discriminator
        t_vars = tf.trainable_variables()
        self.e_vars = [var for var in t_vars if 'e_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        self.saver = tf.train.Saver()

        # sampler
        self.train_shapes, self.dec_shapes, self.rand_shapes \
            = self.sampler(self.sample_inputs[:self.sample_num], self.z)

    def train(self, config):
        """Train SFDGAN"""
        data = glob(os.path.join(self.dataset_dir, config.dataset, self.input_fname_pattern))
        np.random.shuffle(data)

        d_accu = tf.placeholder(tf.float32, name='d_accu')

        # def sigmoid(x, shift, mult):
        #     """
        #     Using this sigmoid to discourage one network overpowering the other
        #     """
        #     return 1 / (1 + tf.exp(-(x + shift) * mult))

        # define optimization operation
        e_opt = tf.train.AdamOptimizer(config.e_learning_rate, beta1=config.beta1)
        g_opt = tf.train.AdamOptimizer(config.g_learning_rate, beta1=config.beta1)
        d_opt = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1)

        # create list of grads from different gpus
        global_e_grads_vars = [None] * self.num_gpus
        global_g_grads_vars = [None] * self.num_gpus
        global_d_grads_vars = [None] * self.num_gpus

        # compute e, g, d gradients
        with tf.variable_scope(tf.get_variable_scope()):
            for gpuid in xrange(self.num_gpus):
                with tf.device('/gpu:%d' % gpuid):
                    global_e_grads_vars[gpuid] = \
                        e_opt.compute_gradients(loss=self.e_losses[gpuid], var_list=self.e_vars)
                    global_g_grads_vars[gpuid] = \
                        g_opt.compute_gradients(loss=self.g_losses[gpuid], var_list=self.g_vars)
                    global_d_grads_vars[gpuid] = \
                        d_opt.compute_gradients(loss=self.d_losses[gpuid], var_list=self.d_vars)

        # average gradients across gpus and apply gradients
        e_grads_vars = average_gradients(global_e_grads_vars)
        g_grads_vars = average_gradients(global_g_grads_vars)
        d_grads_vars = average_gradients(global_d_grads_vars)

        # clip gradients
        e_grads_vars = [(tf.clip_by_value(grad, -.5, .5), var) for grad, var in e_grads_vars]
        g_grads_vars = [(tf.clip_by_value(grad, -.5, .5), var) for grad, var in g_grads_vars]
        d_grads_vars = [(tf.clip_by_value(grad, -.5, .5), var) for grad, var in d_grads_vars]

        e_optim = e_opt.apply_gradients(e_grads_vars)
        g_optim = g_opt.apply_gradients(g_grads_vars)
        d_optim = d_opt.apply_gradients(d_grads_vars)

        # compatibility across tf versions
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.e_sum = merge_summary([self.eg_sum, self.i_sum, self.e_loss_sum, self.e_loss_kl_sum, self.e_loss_ll_sum])
        self.g_sum = merge_summary([self.z_sum, self.d__sum, self.g_loss_eik_sum, self.g_loss_sym_sum,
                                    self.g_loss_gen_sum, self.g_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.d_sum, self.d_loss_real_sum, self.d_loss_sum])

        self.writer = SummaryWriter(self.log_dir, self.sess.graph)

        sample_z = np.random.normal(0, 1, size=(self.sample_num, self.z_dim))
        sample_files = data[0:self.sample_num]
        sample = [np.load(sample_file)[0, :, :, :] for sample_file in sample_files]

        if self.is_grayscale:
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, :, None]

        counter = 0
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        d_accu_last_batch = .5
        batch_idxs = int(math.ceil(min(len(data), config.train_size) / self.glob_batch_size))
        total_steps = config.epoch * (batch_idxs - 1)
        prev_time = -np.inf

        for epoch in xrange(config.epoch):
            # shuffle data before training in each epoch
            np.random.shuffle(data)
            for idx in xrange(0, batch_idxs - 1):
                glob_batch_files = data[idx * self.glob_batch_size:(idx + 1) * self.glob_batch_size]
                glob_batch = [
                    np.load(batch_file)[0, :, :, :] for batch_file in glob_batch_files]
                glob_batch_images = np.array(glob_batch).astype(np.float32)[:, :, :, :, None]

                glob_batch_z = np.random.normal(0, 1, [self.glob_batch_size, self.z_dim]) \
                    .astype(np.float32)
                n_eff = len(glob_batch_files)

                # Pad zeros if effective batch size is smaller than global batch size
                if n_eff != self.glob_batch_size:
                    glob_batch_images = pad_glob_batch(glob_batch_images, self.glob_batch_size)

                # Update E network
                _, summary_str = self.sess.run([e_optim, self.e_sum],
                                               feed_dict={self.inputs: glob_batch_images,
                                                          self.z: glob_batch_z,
                                                          self.n_eff: n_eff,
                                                          d_accu: d_accu_last_batch})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.inputs: glob_batch_images,
                                                          self.z: glob_batch_z,
                                                          self.n_eff: n_eff,
                                                          d_accu: d_accu_last_batch})
                self.writer.add_summary(summary_str, counter)

                # Update D network if accuracy in last batch <= 80%
                if d_accu_last_batch < .8:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.inputs: glob_batch_images,
                                                              self.z: glob_batch_z,
                                                              self.n_eff: n_eff,
                                                              d_accu: d_accu_last_batch})
                    self.writer.add_summary(summary_str, counter)

                # Compute last batch accuracy and losses
                d_accu_last_batch, errD_fake, errD_real, errG, errE \
                    = self.sess.run([self.d_accu, self.d_loss_fake, self.d_loss_real, self.g_loss, self.e_loss],
                                    feed_dict={self.inputs: glob_batch_images,
                                               self.z: glob_batch_z,
                                               self.n_eff: n_eff})
                self.writer.add_summary(summary_str, counter)

                # get time
                now_time = time.time()
                time_per_iter = now_time - prev_time
                prev_time = now_time
                eta = (total_steps - counter + checkpoint_counter) * time_per_iter
                counter += 1

                print(eta, total_steps, counter)
                try:
                    timestr = time.strftime("%H:%M:%S", time.gmtime(eta))
                except:
                    timestr = '?:?:?'

                print("Epoch:[%3d] [%3d/%3d] Iter:[%5d] eta(h:m:s): %s, e_loss: %.8f, d_loss: %.8f, g_loss: %.8f, d_accu: %.4f"
                      % (epoch, idx, batch_idxs, counter,
                         timestr, errE,
                         errD_fake + errD_real, errG, d_accu_last_batch))

                if np.mod(counter, 200) == 1:
                    train_shapes, dec_shapes, rand_shapes \
                        = self.sess.run([self.train_shapes, self.dec_shapes, self.rand_shapes],
                                        feed_dict={self.z: sample_z,
                                                   self.sample_inputs: sample_inputs})
                    np.save(self.sample_dir+'/sample_{:05d}_train.npy'
                                .format(counter), train_shapes)
                    np.save(self.sample_dir + '/sample_{:05d}_dec.npy'
                            .format(counter), dec_shapes)
                    np.save(self.sample_dir + '/sample_{:05d}_rand.npy'
                            .format(counter), rand_shapes)
                    print("[Sample] Iter {0}, saving sample size of {1}, saving checkpoint."
                          .format(counter, self.sample_num))
                    self.save(config.checkpoint_dir, counter)

        # save last checkpoint
        train_shapes, dec_shapes, rand_shapes \
            = self.sess.run([self.train_shapes, self.dec_shapes, self.rand_shapes],
                            feed_dict={self.z: sample_z,
                                       self.sample_inputs: sample_inputs})
        np.save(self.sample_dir + '/sample_final_train.npy', train_shapes)
        np.save(self.sample_dir + '/sample_final_dec.npy', dec_shapes)
        np.save(self.sample_dir + '/sample_final_rand.npy', rand_shapes)
        print("[Sample] Iter {0}, saving sample size of {1f}, saving checkpoint.".format(counter, self.sample_num))
        self.save(config.checkpoint_dir, counter)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv3d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv3d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv3d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv3d(h2, self.df_dim * 8, name='d_h3_conv')))
            h3_flat = tf.contrib.layers.flatten(h3)
            lth_layer = linear(h3_flat, 1024, name='d_ll_lin')
            fin_logit = linear(lth_layer, 1, name='d_fl_lin')
            fin_result = tf.nn.sigmoid(fin_logit)

            return fin_result, fin_logit, lth_layer

    def encoder(self, image, reuse=False):
        with tf.variable_scope("encoder") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv3d(image, self.df_dim, name='e_h0_conv'))
            h1 = lrelu(self.e_bn1(conv3d(h0, self.df_dim * 2, name='e_h1_conv')))
            h2 = lrelu(self.e_bn2(conv3d(h1, self.df_dim * 4, name='e_h2_conv')))
            h3 = lrelu(self.e_bn3(conv3d(h2, self.df_dim * 8, name='e_h3_conv')))
            # flatten
            h3_flat = tf.contrib.layers.flatten(h3)
            z_mean = linear(h3_flat, self.z_dim, name='e_h3_lin_mean')
            z_log_sigma_sq = linear(h3_flat, self.z_dim, name='e_h3_lin_sig')

            return z_mean, z_log_sigma_sq
            
    def generator(self, z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            s_d, s_h, s_w = self.output_depth, self.output_height, self.output_width
            s_d2, s_h2, s_w2 = conv_out_size_same(s_d, 2), conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_d4, s_h4, s_w4 = conv_out_size_same(s_d2, 2), conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_d8, s_h8, s_w8 = conv_out_size_same(s_d4, 2), conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_d16, s_h16, s_w16 = conv_out_size_same(s_d8, 2), conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_d16 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s_d16, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv3d(
                h0, [self.batch_size, s_d8, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv3d(
                h1, [self.batch_size, s_d4, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv3d(
                h2, [self.batch_size, s_d2, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv3d(
                h3, [self.batch_size, s_d, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, images, z):
        enc_z_mean, enc_z_log_sigma_sq = self.encoder(images, reuse=True)
        eps = tf.random_normal((self.batch_size, self.z_dim), 0, 1)  # normal dist for VAE
        enc_z = tf.add(enc_z_mean,
                       tf.multiply(tf.sqrt(tf.exp(enc_z_log_sigma_sq)), eps))  # grab our actual latent vec z
        # generate shapes
        rand_shapes = self.generator(z, reuse=True)
        dec_shapes = self.generator(enc_z, reuse=True)
        train_shapes = images

        return train_shapes, dec_shapes, rand_shapes

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_depth, self.output_height, self.output_width)

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
