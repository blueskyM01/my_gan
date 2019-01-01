import cv2
import numpy as np
import os
from utils import *
# import tensorflow as tf

data_dir = '/media/yang/F/DataSet/Face'
data_set_name = 'lfw-deepfunneled'
label_dir = '/media/yang/F/DataSet/Face/Label'
label_name = 'pair_FGLFW.txt'

# names = np.loadtxt(os.path.join(label_dir,label_name),dtype=str)
#
# names = names[0:16]
# images = []
# for i in names:
#
#     image = cv2.imread(os.path.join(data_dir,data_set_name,i))
#     images.append(image)
#
#
# if not os.path.exists('./samples'):
#     os.makedirs('./samples')
#
# img = (np.array(images,np.float32)- 127.5) / 255.0
# m4_image_save_cv(img,savepath='./samples/{}.jpg'.format(2),rows=2,cols=8,zero_mean=True)



a = np.array([['han'],['yu']],dtype=np.str)
b = a.tolist()
print(b)

for i in range(10):
    if i==4:
        continue
    print(i)




