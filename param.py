import tensorflow as tf

dataset_dir = '/media/yang/F/DataSet/Face'
dataset_name = 'lfw-deepfunneled'
datalabel_dir = '/media/yang/F/DataSet/Face/Label'
datalabel_name = 'pair_FGLFW.txt'
log_dir = './logs'
num_gpus = 2
learning_rate = 0.0002
beta1 = 0.
beta2 = 0.99
batch_size = 64
z_dim = 128
g_feats = 64


