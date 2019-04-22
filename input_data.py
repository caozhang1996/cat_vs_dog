#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 14:36:44 2018

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

BATCH_SIZE = 10

def read_files(file_dir):
    """
    Arg:
        file_dir: f ile directory
    Return:
        list of images and labels
    """
    cats = []
    cats_label = []
    dogs = []
    dogs_label = []
    
    for file in os.listdir(file_dir):
        name = file.split('.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            cats_label.append(0)        # label of cat is 0
        else:
            dogs.append(file_dir + file)
            dogs_label.append(1)        # label of cat is 1
    
    # 数组和标签的整合      
    image_array = np.hstack([cats, dogs])
    label_array = np.hstack([cats_label, dogs_label])
    
    # 将图像和标签整合为一个数组,维度是2维
    temp = np.array([image_array, label_array])
    temp = temp.transpose()
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]   # trans string to int 
    return image_list, label_list
    
def get_batch(image, label, batch_size, image_width=300, image_height=300, capacity=1500):
    """
    Args:
        image: image returned from read_files()
        label: label returned from read_files()
        batch_size: the batch size
        image_width； image width, default is 208
        image_height: image height, default is 208
     Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """
    with tf.name_scope('input'):
        image = tf.cast(image, tf.string)
        label = tf.cast(label, tf.int32)
        
        input_queue = tf.train.slice_input_producer([image, label], shuffle=True)
        label = input_queue[1]
        image_raw = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image_raw, channels=3)
        
        # cut or pad the image, show the raw image, you can comment it 
        image = tf.image.resize_image_with_crop_or_pad(image, image_height, image_width)
        # data data argumentation
    #         image = tf.image.random_flip_left_right(image)    # 随机地水平翻转图像
    #         image = tf.image.random_brightness(image, max_delta=65)  # 随机改变亮度
    #         image = tf.image.random_contrast(image, lower=0.1, upper=2.0) # 随机改变对比度
        
        # image standardization for training
        image = tf.cast(image, tf.float32)  # for training, you can comment it to show image
        image = tf.image.per_image_standardization(image)   
        
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=32,
                                                  capacity=capacity)
        label_batch = tf.reshape(label_batch, [batch_size])
        
    return image_batch, label_batch


# show the image(test the abbove code)
if __name__ == "__main__":
    data_dir = '/home/caozhang/spyder_projects/cats_vs_dogs/data/train/'
    i=0
    
    image_list, label_list = read_files(data_dir)
    image_batch, label_batch = get_batch(image_list, label_list, batch_size=BATCH_SIZE)
    
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            while i<1:
                img, label = sess.run([image_batch, label_batch])
                for j in np.arange(BATCH_SIZE):
                    print ('label: % d' % label[j])
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                    
                    i += 1

        except tf.errors.OutOfRangeError:
            print ('done!')
        finally:
            coord.request_stop()
            coord.join(threads)