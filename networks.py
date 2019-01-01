import tensorflow as tf
import numpy as np
import os
from ops import *
from utils import *


class m4_gan_network:
    def __init__(self, cfg):
        self.cfg = cfg
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def build_model(self, images, labels, z):
        with tf.device('/cpu:0'):

            self.op_g = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate, beta1=self.cfg.beta1, beta2=self.cfg.beta2)
            self.op_d = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate, beta1=self.cfg.beta1, beta2=self.cfg.beta2)

            grads_g = []
            grads_d = []
            grads_c = []

            for i in range(self.cfg.num_gpus):
                images_on_one_gpu = images[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
                labels_on_one_gpu = labels[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
                z_on_one_gpu = z[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
                with tf.device("/gpu:{}".format(i)):
                    with tf.variable_scope("GPU_0") as scope:
                        if i != 0:
                            scope.reuse_variables()

                        fake_image = self.m4_generator(z_on_one_gpu, self.cfg, reuse=False)
                        if i == 0:
                            self.sampler = self.m4_generator(z_on_one_gpu, self.cfg, reuse=True)

                        D_fake = self.m4_discriminator(fake_image,self.cfg,reuse=False)
                        D_real = self.m4_discriminator(images_on_one_gpu,self.cfg,reuse=True)

                        self.d_loss,self.g_loss = m4_wgan_loss(D_real,D_fake)


                        # Gradient penalty
                        lambda_gp = 10.
                        gamma_gp = 1.
                        batch_size = self.cfg.batch_size
                        input_shape = images_on_one_gpu.get_shape().as_list()
                        alpha = tf.random_uniform(shape=input_shape, minval=0., maxval=1.)
                        differences = fake_image - images_on_one_gpu
                        interpolates = images_on_one_gpu + alpha * differences
                        gradients = tf.gradients(
                            self.m4_discriminator(interpolates, self.cfg,reuse=True),
                            [interpolates, ])[0]
                        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
                        gradient_penalty = \
                            lambda_gp * tf.reduce_mean((slopes / gamma_gp - 1.) ** 2)
                        self.d_loss += gradient_penalty
                        

                        # drift
                        eps = 0.001
                        drift_loss = eps * tf.reduce_mean(tf.nn.l2_loss(D_real))
                        self.d_loss += drift_loss


                        self.image_fake_sum = tf.summary.image('image_fake',fake_image)
                        self.g_loss_sum = tf.summary.scalar('g_loss',self.g_loss)
                        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

                        t_vars = tf.trainable_variables()
                        self.g_vars = [var for var in t_vars if 'generator' in var.name]
                        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
                        grad_g = self.op_g.compute_gradients(self.g_loss, var_list=self.g_vars)
                        grads_g.append(grad_g)
                        grad_d = self.op_d.compute_gradients(self.d_loss, var_list=self.d_vars)
                        grads_d.append(grad_d)
                print('Init GPU:{} finshed'.format(i))
            mean_grad_g = m4_average_grads(grads_g)
            mean_grad_d = m4_average_grads(grads_d)
            self.g_optim = self.op_g.apply_gradients(mean_grad_g)
            self.d_optim = self.op_d.apply_gradients(mean_grad_d,global_step=self.global_step)


    def m4_generator(self, z, cfg, reuse=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()

            # z = tf.reshape(z,[cfg.batch_size,1,1,cfg.z_dim])
            conn1 = m4_linear(z, 4 * 4 * cfg.z_dim, name='conn1')
            conn1_reshape = tf.reshape(conn1, [-1, 4, 4, cfg.z_dim])
            deconv1 = m4_deconv_moudel(conn1_reshape, [cfg.batch_size, 8, 8, cfg.g_feats], padding="SAME",
                                       active_function='leak_relu', norm='batch_norm', is_trainable=True,
                                       do_active=True, name='deconv1')
            deconv2 = m4_deconv_moudel(deconv1, [cfg.batch_size, 16, 16, cfg.g_feats], padding="SAME",
                                       active_function='leak_relu', norm='batch_norm', is_trainable=True,
                                       do_active=True, name='deconv2')
            deconv3 = m4_deconv_moudel(deconv2, [cfg.batch_size, 32, 32, cfg.g_feats], padding="SAME",
                                       active_function='leak_relu', norm='batch_norm', is_trainable=True,
                                       do_active=True, name='deconv3')
            deconv4 = m4_deconv_moudel(deconv3, [cfg.batch_size, 64, 64, cfg.g_feats // 2], padding="SAME",
                                       active_function='leak_relu', norm='batch_norm', is_trainable=True,
                                       do_active=True, name='deconv4')
            resnet = deconv4
            for i in range(6):
                resnet = m4_res_block(resnet, [cfg.g_feats // 2, cfg.g_feats // 2], [3, 3], [1, 1],
                                   active_function='leak_relu', name='resnet{}'.format(i))

            deconv5 = m4_deconv_moudel(resnet, [cfg.batch_size, 128, 128, cfg.g_feats // 4], padding="SAME",
                                       active_function='leak_relu', norm='batch_norm', is_trainable=True,
                                       do_active=True, name='deconv5') # 256


            deconv6 = m4_deconv_moudel(deconv5, [cfg.batch_size, 256, 256, cfg.g_feats // 8], padding="SAME",
                                       active_function='leak_relu', norm='batch_norm', is_trainable=True,
                                       do_active=True, name='deconv6')

            conv1 = m4_conv_moudel(deconv6, 3, 7, 7, 1, 1, padding='VALID', active_function='leak_relu',
                                   is_trainable=True, do_active=False, name='conv1')  # 250x250x8

            output = tf.nn.tanh(conv1,name='tanh')

            return output

    def m4_discriminator(self, input_, cfg, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            # feat, d_out = m4_resnet_18(input_,name='resnet_18')
            feat, d_out = m4_VGG(input_,name='resnet_18')
            return d_out
