"""使用MNIST测试数据生成PROJECTOR所需要的两个文件：sprite图像和一个tsv文件给出每张图片对应的真实标签"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

# PROJECT需要额日志文件名和地址相关参数
LOG_DIR = 'log'
SPRITE_FILE = 'mnist_sprite'
META_FILE = "mnist_meta.tsv"


# 使用给出的MNIST图片列表生成sprite图像
def create_sprite_image(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    m = int(np.ceil(np.sqrt(images.shape[0])))

    # 使用全1来初始化最终的大图片
    sprite_image = np.ones((img_h*m, img_w*m))

    for i in range(m):
        for j in range(m):
            # 计算当前图片的编号
            cur = i * m + j
            if cur < images.shape[0]:
                sprite_image[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w] = images[cur]
    return sprite_image


# 加载MNIST数据
mnist = input_data.read_data_sets("MNIST_data",one_hot=False)

# 生成sprite图像
to_visualise = 1 - np.reshape(mnist.test.images, (-1, 28, 28))
sprite_image = create_sprite_image(to_visualise)

# 将生成sprite图像放到相应的日志目录下
path_for_mnist_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')

# 生成每张图片对应的标签文件并写到相应的日志目录下
path_for_mnist_metadata = os.path.join(LOG_DIR, META_FILE)
with open(path_for_mnist_metadata, 'w') as f:
    f.write("Index\tLable\n")
    for index, label in enumerate(mnist.test.labels):
        f.write("%d\t%d\n" % (index, label))