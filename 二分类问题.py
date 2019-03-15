# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:55:23 2019

@author: HGD
"""
"""完整的程序来训练神经网络解决二分类问题"""

import tensorflow as tf
#Numpy是一个科学计算的工具包，通过Numpy工具包生成模拟数据集
from numpy.random import RandomState

#定义训练数据batch的大小
batch_size = 8

#定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#使用None可以方便使用不同的batch大小，仅限少量数据，否则会导致内存溢出
x = tf.placeholder(tf.float32,shape=(None,2),name="x-input")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y-input")

#定义神经网络前向传播的过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数和反向传播的算法
#使用sigmoid函数将y转换为0~1之间的数值,转换后y代表预测是正样本的概率，1-y是负样本概率
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
        y_ * tf.log(tf.clip_by_value(y,1e-10,1.0))
        +(1-y_) * tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
#定义规则来给出样本的标签，x1+x2<1的样例被认为是正样本，否则为负样本，0：负样本，1：正样本
Y = [[int(x1+x2 < 1)] for (x1,x2) in X]
 
#创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    """输入训练之前神经网络参数的值"""
    print (sess.run(w1))
    print (sess.run(w2))
    
    
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size(8)个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size,dataset_size)
        
        #通过选取的样本训练神经网络并更新参数
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 1000 == 0:
            #每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy,feed_dict={x: X,y_: Y})
            print("After %d training step(s),cross entropy on all data is %g" % (i,total_cross_entropy))
            """通过这个结果可以发现随着训练的进行，交叉熵逐渐变小。交叉熵越小，说明预测值的结果和真实的结果差距越小"""
    print (sess.run(w1))
    print (sess.run(w2))
    """训练之后的神经网络参数的值，两个参数发生了变化，这就是训练的结果，它使得这个神经网络能更好地拟合提供的训练数据"""        