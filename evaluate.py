#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 21:06:25 2018

Evaluate the train directory image

@author: caozhang
"""

# 导入python3.*的一些特性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os 
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import input_data
import model

BATCH_SIZE = 1

def get_one_image(train_data):
    """
    Randomly pick one image from test data
    Args:
        data: the train image list 
    Returns:
        image arrary
    """
    n = len(train_data)   # 列表长度
    index = np.random.randint(0, n)  # 随机得到图片索引, np.random.randint(0, n)得到0～n的一个随机数
    image_dir = train_data[index]    # 通过图片索引得到图片的路径
    image = Image.open(image_dir)
    plt.imshow(image)
    
    image_resize = image.resize([300, 300])
    image_arrary = np.array(image_resize)     # 将图片转化成numpy数组
    
    return image_arrary   
    
def evaluate_one_image():
    """
    Test one image against the saved models and parameters
    """
    data_dir = '/home/caozhang/spyder_projects/cats_vs_dogs/my_test_data/'
    log_dir = '/home/caozhang/spyder_projects/cats_vs_dogs/logs'
    image_list, label_list = input_data.read_files(data_dir) 
    image_arrary = get_one_image(image_list)
    
    with tf.Graph().as_default() as g:
        image_arrary = get_one_image(image_list)
        image = tf.cast(image_arrary, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, shape=[1, 300, 300, 3])
        logit = model.inference(image, batch_size=BATCH_SIZE)
        
        logit = tf.nn.softmax(logit)  #  softmax模型可以用来给不同的对象分配概率
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            print ('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print ('Loading success! global step is: %s' % global_step)
            else:
                print ('Not find chekcpoint!')
                return
        
            prediction = sess.run(logit)
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])
                                   
def evaluate_50_images():
    """
    Test 50 image against the saved models and parameters
    """
    data_dir = '/home/caozhang/spyder_projects/cats_vs_dogs/my_test_data/'
    log_dir = '/home/caozhang/spyder_projects/cats_vs_dogs/logs'
    # image_list, label_list = input_data.read_files(data_dir)
    images_list = []
    for file in os.listdir(data_dir):
        images_list.append(data_dir + file)
    
    with tf.Graph().as_default() as g:
        for i in np.arange(0, len(images_list)):
            image_dir = images_list[i]
            image = Image.open(image_dir)
            plt.imshow(image)
            plt.show()
            
            image_resize = image.resize([300, 300])
            image_arrary = np.array(image_resize)
            
            image = tf.cast(image_arrary, tf.float32)
            image = tf.image.per_image_standardization(image)
            image = tf.reshape(image, shape=[1, 300, 300, 3])
            logit = model.inference(image, batch_size=BATCH_SIZE)
            
            logit = tf.nn.softmax(logit)  #  softmax模型可以用来给不同的对象分配概率
            saver = tf.train.Saver()
            
            with tf.Session() as sess:
                # 测试多张图片，我们模型的参数需要重复使用，所以我们需要告诉TF允许复用参数，加上下行代码
                tf.get_variable_scope().reuse_variables()
                print ('Reading checkpoints...')
                ckpt = tf.train.get_checkpoint_state(log_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print ('Loading success! global step is: %s' % global_step)
                else:
                    print ('Not find checkpoint!')
                    return
            
                prediction = sess.run(logit)
                max_index = np.argmax(prediction) 
                if max_index==0:
                    print('This is a cat with possibility %.6f' %prediction[:, 0])
                else:
                    print('This is a dog with possibility %.6f' %prediction[:, 1])
            
                
    
