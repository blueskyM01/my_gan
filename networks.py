import tensorflow as tf
import numpy as np
import os
from ops import *
from utils import *

class m4_gan_network:
    def __init__(self,cfg):
        self.cfg = cfg

    def build_model(self,images,labels,z,learning_rate,beta1,beta2):
        with tf.device('/cpu:0'):


            self.op_g = tf.train.AdamOptimizer(self.cfg.learning_rate,beta1=self.cfg.beta1,beta2=self.cfg.beta2)
            self.op_d = tf.train.AdamOptimizer(self.cfg.learning_rate, beta1=self.cfg.beta1, beta2=self.cfg.beta2)

            grads_g = []
            grads_d = []
            grads_c = []

            for i in range(self.cfg.num_gpus):
                images_on_one_gpu = images[self.cfg.batch_size * i:self.cfg.batch_size * (i+1)]
                labels_on_one_gpu = labels[self.cfg.batch_size * i:self.cfg.batch_size * (i+1)]
                z_on_one_gpu = z[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
                with tf.device("/gpu:{}".format(i)):
                    with tf.variable_scope("GPU:0") as scope:
                        if i != 0:
                            scope.reuse_variables()

                        self.loss_g = 1
                        self.loss_d = 2

                        t_vars = tf.trainable_variables()

                        self.g_vars = [var for var in t_vars if 'g_' in var.name]
                        self.d_vars = [var for var in t_vars if 'd_' in var.name]


                        grad_g = self.op_g.compute_gradients(self.loss_g, var_list=self.g_vars)
                        grads_g.append(grad_g)
                        grad_d = self.op_d.compute_gradients(self.loss_d, var_list=self.d_vars)
                        grads_d.append(grad_d)

            mean_grad_g = m4_average_grads(grads_g)
            mean_grad_d = m4_average_grads(grads_d)
            self.g_optim = self.op_g.apply_gradients(mean_grad_g,global_step=self.global_step)
            self.d_optim = self.op_d.apply_gradients(mean_grad_d)

        return self.g_optim,self.d_optim

    def m4_generator(self,z,cfg,reuse=False):
        with tf.name_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            z = tf.reshape(z,[cfg.batch_size,1,1,cfg.z_dim])

            






