# -*- coding: utf-8 -*-
"""
Hyperparamaters for the CNN

@author: Sazzad
"""
directory = 'birds'
transform_variables = {'vertical_flip_probability': 0.5, 
                       'horizontal_flip_probability': 0.5,
                       'degrees': (0,20),
                       'Normalize_mean': [0.485, 0.456, 0.406],
                       'Normalize_std': [0.229, 0.224, 0.225]}
batch_size = 32
epoch = 100
learning_rate = 0.001
weight_decay = 0.01
dropout_rate = 0.4
momentum = 0.9


input_channel = 3
number_of_filters = [32, 64, 96, 128, 256]
filter_size = [3, 3, 5, 5, 7]
conv_padding = 1
conv_stride = 1
maxpool_filter_size = 2
maxpool_stride = 2
parameters_after_flatten = 18432
Linear_layers = [1024, 512, 100]
number_of_outputs = 10

#testing

test_batch_size = 50