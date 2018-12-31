from __future__ import division, print_function, absolute_import
import os
import argparse
import tensorflow as tf
import param
from model import my_gan
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'   #指定第一块GPU可用

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default=param.dataset_dir,type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default=param.dataset_name,type=str, help="Train data set name")
parser.add_argument("--datalabel_dir", default=param.datalabel_dir,type=str, help="Train data label dir")
parser.add_argument("--datalabel_name", default=param.datalabel_name,type=str, help="Train data label name")
parser.add_argument("--log_dir", default=param.log_dir,type=str, help="Train data label name")


parser.add_argument("--num_gpus", default=param.num_gpus,type=int, help="num of gpu")
parser.add_argument("--batch_size", default=param.batch_size,type=int, help="batch size for one gpus")
parser.add_argument("--z_dim", default=param.z_dim,type=int, help="dim of noise")
parser.add_argument("--learning_rate", default=param.learning_rate,type=float, help="learn_rate")
parser.add_argument("--beta1", default=param.beta1,type=float, help="beta1")
parser.add_argument("--beta2", default=param.beta2,type=float, help="beta2")
parser.add_argument("--g_feats", default=param.g_feats,type=int, help="g_feats")
cfg = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)

    config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        my_gan = my_gan(sess,cfg)
        my_gan.train()