import tensorflow as tf
import numpy as np
import os
import datetime
from utils import *
from ops import *
from networks import *
import time


class my_gan:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg
        self.images = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus, 250, 250, 3],
                                     name='real_image')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus, 10000],
                                     name='id')
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.cfg.z_dim], name='noise_z')

        my_gan_model = m4_gan_network(self.cfg)
        my_gan_model.build_model(self.images, self.labels, self.z)
        self.g_optim = my_gan_model.g_optim
        self.d_optim = my_gan_model.d_optim
        self.g_loss = my_gan_model.g_loss
        self.d_loss = my_gan_model.d_loss
        self.image_fake_sum = my_gan_model.image_fake_sum
        self.g_loss_sum = my_gan_model.g_loss_sum
        self.d_loss_sum = my_gan_model.d_loss_sum
        self.global_step = my_gan_model.global_step
        self.sampler = my_gan_model.global_step

    def train(self):
        saver = tf.train.Saver()
        try:
            tf.global_variables_initializer().run()
        except:

            tf.initialize_all_variables().run()

        self.writer = tf.summary.FileWriter(
            '{}/{}'.format(self.cfg.log_dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))),
            self.sess.graph)
        # self.writer = tf.summary.FileWriter('{}'.format(self.cfg.log_dir), self.sess.graph)
        # self.writer.close()
        merged = tf.summary.merge_all()

        names = np.loadtxt(os.path.join(self.cfg.datalabel_dir, self.cfg.datalabel_name), dtype=np.str)
        dataset_size = names.shape[0]
        names = names.tolist()
        names = m4_get_open_image_name(names, os.path.join(self.cfg.dataset_dir, self.cfg.dataset_name))

        filenames = tf.constant(names)
        try:
            dataset = tf.data.Dataset.from_tensor_slices((filenames, filenames))
        except:
            dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, filenames))

        dataset = dataset.map(m4_parse_function)
        dataset = dataset.shuffle(buffer_size=1000).batch(self.cfg.batch_size * self.cfg.num_gpus).repeat(
            self.cfg.epoch)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        batch_idxs = dataset_size // (self.cfg.batch_size * self.cfg.num_gpus)

        # try:
        for epoch in range(self.cfg.epoch):
            for idx in range(batch_idxs):

                starttime = datetime.datetime.now()
                batch_images, batch_labels = self.sess.run(one_element)
                batch_z = np.random.uniform(-1, 1, [self.cfg.batch_size * self.cfg.num_gpus, self.cfg.z_dim]).astype(np.float32)
                if batch_images.shape[0] < self.cfg.batch_size * self.cfg.num_gpus:
                    continue

                # Upadate D network
                _, d_loss, summary_str,counter,merged_ = self.sess.run(
                    [self.d_optim, self.d_loss, self.d_loss_sum,self.global_step,merged],
                    feed_dict={self.images: batch_images,
                               self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, g_loss, summary_str, samplers = self.sess.run(
                    [self.g_optim, self.g_loss, self.g_loss_sum, self.image_fake_sum],
                    feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                self.writer.add_summary(samplers)

                endtime = datetime.datetime.now()

                timediff = (endtime - starttime).total_seconds()
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, self.cfg.epoch, idx, batch_idxs, timediff, d_loss, g_loss))
                try:
                    if counter % self.cfg.saveimage_period == 0:
                        samples = self.sess.run([self.sampler], feed_dict={self.z: batch_z})
                        m4_image_save_cv(samples[0],
                                         '{}/train_{}_{}.jpg'.format(self.cfg.sampel_save_dir, epoch, counter))
                except:
                    print('oen pic error')
            #
        # except:
        #     print('Mission complete!')
        #
