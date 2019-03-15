# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:38:42 2019

@author: HGD
"""
"""通过placeholder实现前向传播算法"""

import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#定义placeholder作为存放输入数据的地方，维度不一定要定义
x = tf.placeholder(tf.float32,shape=(1,2),name="input")
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#SyntaxError: invalid character in identifier可能是某个符号使用了中文输入法
#feed_dict={}是指定x的取值，是一个字典(map),这个用了placeholder之后必须使用
print(sess.run(y,feed_dict={x: [[0.7,0.9]]}))