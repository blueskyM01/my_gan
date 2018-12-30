import os
import argparse
import tensorflow as tf
import param

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default=param.dataset_dir,type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default=param.dataset_name,type=str, help="Train data set name")
parser.add_argument("--datalabel_dir", default=param.datalabel_dir,type=str, help="Train data label dir")
parser.add_argument("--datalabel_name", default=param.datalabel_name,type=str, help="Train data label name")
parser.add_argument("--num_gpus", default=param.num_gpus,type=int, help="num of gpu")
parser.add_argument("--batch_size", default=param.batch_size,type=int, help="batch size for one gpus")
parser.add_argument("--z_dim", default=param.z_dim,type=int, help="dim of noise")

cfg = parser.parse_args()



config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    a = 1
