from PIL import Image
import numpy as np
import os


# 将图像（灰度图）以矩阵（数字）的的形式输出
img = Image.open("mnist_test_3.png")
img_array = np.asarray(img)
print("输出图像数组：", img_array)
print("数组形状：", img_array.shape)

# 批量：图像（灰度图）以矩阵（数字）的的形式输出

list_dir = os.listdir("train-images/0/")
for image in list_dir:
    im = Image.open("train-images/0/"+image)
    print("输出图像数组：", np.asarray(im))




