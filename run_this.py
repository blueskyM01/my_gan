from __future__ import division, print_function, absolute_import
import os
import argparse
import tensorflow as tf
import param
from model import my_gan

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # 指定第  块GPU可用

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
# TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
# TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
# TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息


parser = argparse.ArgumentParser()
'''
#-----------------------------m4_gan_network-----------------------------
parser.add_argument("--dataset_dir", default=param.dataset_dir, type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default=param.dataset_name, type=str, help="Train data set name")
parser.add_argument("--datalabel_dir", default=param.datalabel_dir, type=str, help="Train data label dir")
parser.add_argument("--datalabel_name", default=param.datalabel_name, type=str, help="Train data label name")
parser.add_argument("--log_dir", default=param.log_dir, type=str, help="Train data label name")
parser.add_argument("--sampel_save_dir", default=param.sampel_save_dir, type=str, help="sampel save dir")
parser.add_argument("--num_gpus", default=param.num_gpus, type=int, help="num of gpu")
parser.add_argument("--epoch", default=param.epoch, type=int, help="epoch")
parser.add_argument("--batch_size", default=param.batch_size, type=int, help="batch size for one gpus")
parser.add_argument("--z_dim", default=param.z_dim, type=int, help="dim of noise")
parser.add_argument("--g_feats", default=param.g_feats, type=int, help="feats for generator")
parser.add_argument("--learning_rate", default=param.learning_rate, type=float, help="learn_rate")
parser.add_argument("--beta1", default=param.beta1, type=float, help="beta1")
parser.add_argument("--beta2", default=param.beta2, type=float, help="beta2")
parser.add_argument("--saveimage_period", default=param.saveimage_period, type=int, help="saveimage_period")
parser.add_argument("--savemodel_period", default=param.savemodel_period, type=int, help="savemodel_period")
#-----------------------------m4_gan_network-----------------------------
'''

# -----------------------------m4_BE_GAN_network-----------------------------
parser.add_argument("--dataset_dir", default=param.dataset_dir, type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default=param.dataset_name, type=str, help="Train data set name")
parser.add_argument("--datalabel_dir", default=param.datalabel_dir, type=str, help="Train data label dir")
parser.add_argument("--datalabel_name", default=param.datalabel_name, type=str, help="Train data label name")
parser.add_argument("--log_dir", default=param.log_dir, type=str, help="Train data label name")
parser.add_argument("--sampel_save_dir", default=param.sampel_save_dir, type=str, help="sampel save dir")
parser.add_argument("--num_gpus", default=param.num_gpus, type=int, help="num of gpu")
parser.add_argument("--epoch", default=param.epoch, type=int, help="epoch")
parser.add_argument("--batch_size", default=param.batch_size, type=int, help="batch size for one gpus")
parser.add_argument("--z_dim", default=param.z_dim, type=int, choices=[64, 128], help="dim of noise")
parser.add_argument("--conv_hidden_num", default=param.conv_hidden_num, type=int, choices=[64, 128],
                    help="conv_hidden_num")
parser.add_argument("--data_format", default=param.data_format, type=str, help="data_format")
parser.add_argument("--g_lr", default=param.g_lr, type=float, help="learning rate of G")
parser.add_argument("--d_lr", default=param.d_lr, type=float, help="learning rate of D")
parser.add_argument("--gamma", default=param.gamma, type=float, help="gamma")
parser.add_argument("--lambda_k", default=param.lambda_k, type=float, help="lambda_k")
parser.add_argument("--saveimage_period", default=param.saveimage_period, type=int, help="saveimage_period")
parser.add_argument("--savemodel_period", default=param.savemodel_period, type=int, help="savemodel_period")
# -----------------------------m4_BE_GAN_network-----------------------------


cfg = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(cfg.sampel_save_dir):
        os.makedirs(cfg.sampel_save_dir)
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        my_gan = my_gan(sess, cfg)
        my_gan.train()
