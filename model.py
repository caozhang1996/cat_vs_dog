#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 15:24:25 2018

@author: caozhang
"""

# 导入python3.*的一些特性 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import numpy as np

import input_data

N_CLASSES = 2
BATCH_SIZE = 16
MAX_STEP = 10000        # with current parameters, it is suggested to use MAX_STEP>10k
LEARNING_RATE = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001
data_dir = '/home/caozhang/spyder_projects/cats_vs_dogs/data/train/'

def inference(images, batch_size):
    """
    Args:
        images: images: 4D tensor [batch_size, img_width, img_height, img_channel]
        batch_size: batch size
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    """
    
    # conv1 
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        
    # pool1 and norm1
    with tf.variable_scope('pooling_lrn1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3,1], strides=[1, 2, 2, 1],padding='SAME', name='pooling1')
        # norm1 lrn: Local Response Normalization (局部响应归一化)
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')
    
    # conv2
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights', 
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
    
    # pool2 and norm2
    with tf.variable_scope('pooling_lrn2') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3,1], strides=[1, 1, 1, 1],padding='SAME', name='pooling2')
       
    # local3
    with tf.variable_scope('local3') as scope:
        # flatten, 这里shape=[batch_size, -1]是因为images批次大小为batch_size
        # reshape 的值类似于:
#        [[1, 2 ... 4]
#         [2, 3 ... 6]
#         [3, 4 ... 4]
#         [4, 5 ... 1]
#        ...
#         [1, 3 ... 4]
#         [4, 6 ... 7]]
        reshape = tf.reshape(pool2, shape=[batch_size, -1])  # flatten, 变成列向量，维度为batch_size 
        dim = reshape.get_shape()[1].value     # # 0代表第一维， 1代表第二维
        
        weights = tf.get_variable('weights', 
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        
    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights', 
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    
    # softmax_linear
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights', 
                                  shape=[128, N_CLASSES],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[N_CLASSES], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32))
        softmax_linear = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
        
    return softmax_linear

def loss(logits, labels):
    """
    Args:
        logits: logits returned from inference()
        labels: labels_batch returned from input_data.get_batch()
    Returns:
        model loss
    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name=scope.name)
        tf.summary.scalar(scope.name, loss)
    return loss

def accuracy(logits, labels):
    """
    calculate the accuracy of model
    Args:
        logits: logits returned from inference()
        labels: labels_batch returned from input_data.get_batch()
    Returns:
        model accuracy
    """
    with tf.variable_scope('accuracy') as scope:
        correct_pred = tf.nn.in_top_k(logits, labels, 1)
        correct_pred = tf.cast(correct_pred, tf.float32)
        accuracy = tf.reduce_mean(correct_pred)
        tf.summary.scalar(scope.name, accuracy)
    return accuracy

def train():
    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    data_dir = '/home/caozhang/spyder_projects/cats_vs_dogs/data/train/'
    log_dir = '/home/caozhang/spyder_projects/cats_vs_dogs/logs/'
    tensorboard_dir = '/home/caozhang/spyder_projects/cats_vs_dogs/tensorboard_files/'
        
    image_list, label_list = input_data.read_files(data_dir)
    image_batch, label_batch = input_data.get_batch(image_list, label_list, batch_size=BATCH_SIZE)           
    train_logits = inference(image_batch, batch_size=BATCH_SIZE)
    train_loss = loss(train_logits, label_batch)
    train_accuracy = accuracy(train_logits, label_batch)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(train_loss, global_step=my_global_step)

    # model saver and tensorboard event saver
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                _, loss_value, accuracy_vale = sess.run([train_op, train_loss, train_accuracy])
                
                if step % 50 == 0:
                    print ('step: %d, loss: %f, accuracy: %f' % (step, loss_value, accuracy_vale))
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                     
                if step % 2000 == 0 or (step+1) == MAX_STEP:
                    checkpoint_path = os.path.join(log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print ('Done training!')
        finally:
            coord.request_stop()
            coord.join(threads)
            
