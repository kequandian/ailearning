# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:03:16 2019

@author: Ace
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print("Example training data: \n", mnist.train.images[0])
print("************************************************************")
print("Example training data label: ", mnist.train.labels[0])
