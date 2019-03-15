# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:15:46 2019

@author: HGD
"""
"""通过变量实现神经网络的参数并实现前向传播的过程"""
import tensorflow as tf

#通过seed参数设定了随机种子，保证每次运行得到的结果都一样
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.constant([[0.7,0.9]])

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
#sess.run(w1.initializer)
#sess.run(w2.initializer)
init =tf.global_variables_initializer()
sess.run(init)
print(sess.run(y))
sess.close()