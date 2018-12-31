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


        self.images = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size* self.cfg.num_gpus, 250, 250, 3],
                                     name='real_image')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size* self.cfg.num_gpus, 10000], name='id')
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.cfg.z_dim], name='noise_z')
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[1], name='learning_rate')
        self.beta1 = tf.placeholder(dtype=tf.float32, shape=[1], name='beta1')
        self.beta2 = tf.placeholder(dtype=tf.float32, shape=[1], name='beta2')
        self.g_optim, self.d_optim = m4_gan_network(self.cfg). \
            build_model(self.images, self.labels, self.z, self.learning_rate, self.beta1, self.beta2)

    def train(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        # self.writer = tf.summary.FileWriter(
        #     '{}/{}'.format(self.cfg.log_dir, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))),
        #     self.sess.graph)
        self.writer = tf.summary.FileWriter('{}'.format(self.cfg.log_dir),self.sess.graph)
        # self.writer.close()
        merged = tf.summary.merge_all()

        names = np.loadtxt(os.path.join(self.cfg.datalabel_dir, self.cfg.datalabel_name), dtype=str)
        names = names.tolist()
        filenames = tf.constant(names)
        try:
            dataset = tf.data.Dataset.from_tensor_slices((filenames,filenames))
        except:
            dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames,filenames))

        dataset = dataset.map(m4_parse_function)
        dataset = dataset.shuffle(buffer_size=1000).batch(self.cfg.batch_size * self.cfg.num_gpus).repeat(40)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        for i in range(10):

            self.sess.run(one_element)
