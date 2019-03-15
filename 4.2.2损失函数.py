# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:36:30 2019

@author: HGD
"""
"""使用定义的损失函数，讲解损失函数对模型训练结果的影响"""
"""实现了一个拥有两个输入节点、一个输出节点，没有隐藏层的神经网络"""

import tensorflow as tf
#Numpy是一个科学计算的工具包，通过Numpy工具包生成模拟数据集
from numpy.random import RandomState

#定义训练数据batch的大小
batch_size = 8

#定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#两个输入节点
x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
#回归问题一般只有一个输出节点
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")

#定义神经网络前向传播的过程，这里就是简单加权和
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y,y_),(y - y_) * loss_more,(y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
#设置回归的正确值为两个输入的和加上一个随机量，随机量是为了加入不可预测的噪音，设置为-0.05~0.05的随机数
Y = [[x1 + x2 +rdm.rand()/10.0-0.05] for (x1,x2) in X]
 
#训练神经网络
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)    
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size(8)个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size,dataset_size)
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
    print(sess.run(w1))

#不同的损失函数对训练得到的模型产生重要影响