import tensorflow as tf
import numpy as np
import os
import datetime
from utils import *
from ops import *
from networks import *

class my_gan:
    def __init__(self,sess,cfg):
        self.cfg = cfg
        self.global_step = tf.Variable(0, name='global_step', trainable=False)



        self.g_optim,self.d_optim = m4_gan_network(cfg).build_model()




    def train(self):
        a = 1