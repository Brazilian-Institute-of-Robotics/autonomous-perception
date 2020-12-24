#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:15:43 2019

@author: nelson
"""


import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Nadam, Adamax
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import AvgPool2D, Conv2D, BatchNormalization, ReLU, Lambda, Add
from tensorflow.keras.models import Model

import math
#from imgaug import augmenters as iaa
# import imgaug as ia
from PIL import Image
import numpy as np
import glob
import json

import sys
import os
import logging





from tensorflow.keras.applications.resnet import ResNet101, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from matplotlib import pyplot as plt






#SNR = mean^2/std^2
#SNRdb = 10Log10(SNR) -> 10^(SNRdb/10) = SNR -> 10^(SNRdb/10) = mean^2/std^2
#std^2 = mean^2/10^(SNRdb/10) -> std = sqrt(mean^2/(10^(SNRdb/10)))
a = 1
SNRdb = 10


N = 100000
F = 10
n = np.array(range(N))
y = np.sin(2*np.pi*F*n/N)
plt.plot(y)

# std = np.sqrt(np.mean(y**2)/(10**(SNRdb/10)))

for intensity in [.05, .1, .15, .2, .25]:
    std = np.sqrt(np.mean(y**2))*intensity
    
    
    noise = np.random.normal(loc=0, scale=std, size=N)
    y_ = y + noise
    plt.plot(y_)
    
    print("SNR noise: "+ str(np.sqrt(np.mean(noise**2))))
    print("SNR signal: "+ str(np.sqrt(np.mean(y**2))))

    
    SNRdb_ = 10*np.log10(np.mean(y**2)/np.mean((y-y_)**2))
    print("SNRdb_: "+str(SNRdb_))
    
    SNRdb = 20*np.log10(1/intensity)
    print("SNRdb : "+str(SNRdb))
# extracted_path = "/home/nelson/projects/da_art_perception/data/dataset"
# label_suffix = '*label_raw.png'
# image_suffix = '.jpg'

# paths = ['/off-road/evening/cimatec-industrial/small.txt',
#  '/off-road/rain/cimatec-industrial/small.txt',
#  '/unpaved/rain/jaua/small.txt',
#  '/unpaved/rain/praia-do-forte/small.txt',
#  '/unpaved/rain/estrada-dos-tropeiros/small.txt',
#  '/unpaved/day/jaua/small.txt',
#  '/unpaved/day/praia-do-forte/small.txt',
#  '/unpaved/day/estrada-dos-tropeiros/small.txt']

# offroad_paths = ['/night_offroad_clean-test_subset.txt',
# '/day_offroad_clean-test_subset.txt',
# '/night_offroad_dusty-test_subset.txt',
# '/day_offroad_dusty-test_subset.txt']


# img_list = {"evening":[], "rain":[], "day":[], "day_offroad_clean":[],
#             "day_offroad_dusty":[], "night_offroad_clean":[], 
#             "night_offroad_dusty":[]}
# lbl_list = {"evening":[],"rain":[],"day":[], "day_offroad_clean":[],
#             "day_offroad_dusty":[], "night_offroad_clean":[], 
#             "night_offroad_dusty":[]}


# for path in paths:
#     print(path)
#     lineList = open(extracted_path + path).readlines()
#     for name in lineList:
#         search_name_path = path[:path.rfind('/')]+'/'+name.replace('\n', '')
#         full_name_path = tf.io.gfile.glob(extracted_path + search_name_path + label_suffix)[0]
#         if 'test' in full_name_path:
#             img_list[path.split("/")[2]].append(extracted_path + search_name_path + image_suffix)
#             lbl_list[path.split("/")[2]].append(full_name_path)

# for path in offroad_paths:
#     print(path)
#     lineList = open(extracted_path + path).readlines()
#     for name in lineList:
#         search_name_path = path[:path.rfind('/')]+'/'+name.replace('.jpg\n', '')
#         full_name_path = tf.io.gfile.glob(extracted_path + search_name_path + label_suffix)[0]
#         if 'test' in full_name_path:
#             img_list[path.replace('/', '').split("-")[0]].append(extracted_path + search_name_path + image_suffix)
#             lbl_list[path.replace('/', '').split("-")[0]].append(full_name_path)

            
            
# split = round(len(img_list)*.10)
# random.Random(0).shuffle(img_list)
# random.Random(0).shuffle(lbl_list)
# val_img_list = img_list[:split]
# val_lbl_list = lbl_list[:split]
# train_img_list = img_list[split:]
# train_lbl_list = lbl_list[split:]

# model = ResNet50(weights='imagenet')
# modelr101 = ResNet101(weights='imagenet')
# modelmbnet = MobileNetV2(weights='imagenet')
# vgg16 = VGG16(weights='imagenet', include_top=False)


# # ###############################################################################
# # #########################Create Backbone resnet##################################
# logging.basicConfig(level=logging.INFO)

# # resnet 50
# # conv4_block1_1_conv (Conv2D)    (None, 14, 14, 256)  131328      conv3_block4_out[0][0]            <-- os8
# # conv5_block1_1_conv (Conv2D)    (None, 7, 7, 512)    524800      conv4_block6_out[0][0]            <-- 0s16
# # self.strideOutput32LayerName = conv5_block3_out
# # self.strideOutput16LayerName = conv4_block6_out
# # self.strideOutput8LayerName = conv3_block4_out
# # self.inputLayerName = resnet50.layers[0].name
# dl_input_shape=(None, 483,769,3)
# dl_input_shape=(None, 224,224,3)

# resnet101 = ResNet101(weights='imagenet',  input_shape=(dl_input_shape[1], dl_input_shape[2], 3), include_top=False)

# logger = logging.getLogger('perception.models.CMSNet')
# mobile_config = resnet101.get_config()
# mobile_weights = resnet101.get_weights()

# output_stride = 8
# dilatation = 1
# stride_enable = False
# for layer in mobile_config['layers']:
#     if layer['name'] == 'input_1':
#         layer['config']['batch_input_shape'] = (None, dl_input_shape[-3], dl_input_shape[-2], dl_input_shape[-1])
#         logger.info(layer['name']+', '+str(layer['config']['batch_input_shape']))
#     if output_stride == 8 and (layer['name'] == 'conv4_block1_1_conv'  or
#                                layer['name'] == 'conv4_block1_0_conv'):
#         layer['config']['strides'] = (1, 1)
#         logger.info(layer['name']+', strides='+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
#         stride_enable = True
#         if  layer['name'] == 'conv4_block1_1_conv':
#             dilatation = dilatation*2 #Replace stride for dilatation

#     elif output_stride in [8, 16] :
#         if layer['name'] == 'conv5_block1_1_conv' or layer['name'] == 'conv5_block1_0_conv':
#             logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
#             layer['config']['strides'] = (1, 1)
#             logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
#             stride_enable = True
#             if layer['name'] == 'conv5_block1_1_conv':
#                 dilatation = dilatation*2 #Replace stride for dilatation
#         elif stride_enable and ('_conv' in layer['name']):
#             if layer['config']['kernel_size']!=(1,1):
#                 layer['config']['dilation_rate'] = (dilatation, dilatation)
#                 logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
#             else:
#                 logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))



# backbone = Model.from_config(mobile_config)
# mobile_weights[0] = np.resize(mobile_weights[0], [mobile_weights[0].shape[0], mobile_weights[0].shape[1], dl_input_shape[-1], mobile_weights[0].shape[-1]]) #update input to suport 4 channels
# backbone.set_weights(mobile_weights)


# strideOutput32LayerName = 'conv5_block3_out'
# strideOutput16LayerName = 'conv4_block6_out'
# strideOutput8LayerName = 'conv3_block4_out'
# inputLayerName = resnet101.layers[0].name
# # ###############################################################################



# ###############################################################################
# #########################Create Backbone VGG16##################################
# resnet 101
# conv4_block1_1_conv (Conv2D)    (None, 14, 14, 256)  131328      conv3_block4_out[0][0]            <-- os8
# conv5_block1_1_conv (Conv2D)    (None, 7, 7, 512)    524800      conv4_block23_out[0][0]            <-- os16
# self.strideOutput32LayerName = conv5_block3_out
# self.strideOutput16LayerName = conv4_block23_out
# self.strideOutput8LayerName = conv3_block4_out
# self.inputLayerName = resnet101.layers[0].name
# dl_input_shape=(None, 483,769,3)
# dl_input_shape=(None, 224,224,3)

# resnet101 = ResNet101(weights='imagenet',  input_shape=(dl_input_shape[1], dl_input_shape[2], 3), include_top=False)

# logger = logging.getLogger('perception.models.CMSNet')
# mobile_config = resnet101.get_config()
# mobile_weights = resnet101.get_weights()

# output_stride = 8
# dilatation = 1
# stride_enable = False
# for layer in mobile_config['layers']:
#     if layer['name'] == 'input_1':
#         layer['config']['batch_input_shape'] = (None, dl_input_shape[-3], dl_input_shape[-2], dl_input_shape[-1])
#         logger.info(layer['name']+', '+str(layer['config']['batch_input_shape']))
#     if output_stride == 8 and (layer['name'] == 'conv4_block1_1_conv'  or
#                                layer['name'] == 'conv4_block1_0_conv'):
#         layer['config']['strides'] = (1, 1)
#         logger.info(layer['name']+', strides='+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
#         stride_enable = True
#         if  layer['name'] == 'conv4_block1_1_conv':
#             dilatation = dilatation*2 #Replace stride for dilatation

#     elif output_stride in [8, 16] :
#         if layer['name'] == 'conv5_block1_1_conv' or layer['name'] == 'conv5_block1_0_conv':
#             logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
#             layer['config']['strides'] = (1, 1)
#             logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
#             stride_enable = True
#             if layer['name'] == 'conv5_block1_1_conv':
#                 dilatation = dilatation*2 #Replace stride for dilatation
#         elif stride_enable and ('_conv' in layer['name']):
#             if layer['config']['kernel_size']!=(1,1):
#                 layer['config']['dilation_rate'] = (dilatation, dilatation)
#                 logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
#             else:
#                 logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))



# backbone = Model.from_config(mobile_config)
# mobile_weights[0] = np.resize(mobile_weights[0], [mobile_weights[0].shape[0], mobile_weights[0].shape[1], dl_input_shape[-1], mobile_weights[0].shape[-1]]) #update input to suport 4 channels
# backbone.set_weights(mobile_weights)


# strideOutput32LayerName = 'conv5_block3_out'
# strideOutput16LayerName = 'conv4_block23_out'
# strideOutput8LayerName = 'conv3_block4_out'
# inputLayerName = resnet101.layers[0].name
# ###############################################################################


# ###############################################################################
# #########################Create Backbone VGG16##################################
#vgg16
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0            <-- os8
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0            <-- os16
# self.strideOutput32LayerName = block5_pool
# self.strideOutput16LayerName = block5_conv3
# self.strideOutput8LayerName = block4_conv3
# self.inputLayerName = vgg16.layers[0].name

# dl_input_shape=(None, 483,769,3)

# vgg16 = VGG16(weights='imagenet',  input_shape=(dl_input_shape[1], dl_input_shape[2], 3), include_top=False)

# logger = logging.getLogger('perception.models.CMSNet')
# mobile_config = vgg16.get_config()
# mobile_weights = vgg16.get_weights()

# output_stride = 16
# dilatation = 1
# stride_enable = False
# for layer in mobile_config['layers']:
#     if layer['name'] == 'input_1':
#         layer['config']['batch_input_shape'] = (None, dl_input_shape[-3], dl_input_shape[-2], dl_input_shape[-1])
#         logger.info(layer['name']+', '+str(layer['config']['batch_input_shape']))
#     if output_stride == 8 :
#         if layer['name'] == 'block4_pool':
#             layer['config']['strides'] = (1, 1)
#             layer['config']['pool_size'] = (1, 1)
#             dilatation = dilatation*2 #Replace stride for dilatation 
#             logger.info(layer['name']+', strides='+str(layer['config']['strides'])+', '+str(layer['config']['pool_size']))
#             stride_enable = True
#         # if layer['name'] == 'block_7_depthwise':
#         #     layer['config']['dilation_rate'] = (dilatation, dilatation)
#         #     logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))

#     if output_stride in [8, 16] :
#         if layer['name'] == 'block5_pool':
#             layer['config']['strides'] = (1, 1)
#             layer['config']['pool_size'] = (1, 1)
#             dilatation = dilatation*2 #Replace stride for dilatation 
#             logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['pool_size']))
#             stride_enable = True
#         if 'conv' in layer['name'] and stride_enable:
#             layer['config']['dilation_rate'] = (dilatation, dilatation)
#             logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))


# new_vgg16 = Model.from_config(mobile_config)
# mobile_weights[0] = np.resize(mobile_weights[0], [mobile_weights[0].shape[0], mobile_weights[0].shape[1], dl_input_shape[-1], mobile_weights[0].shape[-1]]) #update input to suport 4 channels
# new_vgg16.set_weights(mobile_weights)


# x = new_vgg16.output
# x = Conv2D(filters=512, kernel_size=3, name='block_6_conv1', padding="same", dilation_rate=dilatation, activation='relu')(x)
# x = Conv2D(filters=512, kernel_size=3, name='block_6_conv2', padding="same", dilation_rate=dilatation, activation='relu')(x)
# x = Conv2D(filters=512, kernel_size=3, name='block_6_conv3', padding="same", dilation_rate=dilatation, activation='relu')(x)



# backbone = Model(inputs=new_vgg16.inputs, outputs=x)

# strideOutput32LayerName = 'block6_conv3'
# strideOutput16LayerName = 'block5_conv3'
# strideOutput8LayerName = 'block4_conv3'
# inputLayerName = vgg16.layers[0].name
# ###############################################################################

###############################################################################
# # # ###########################count total train and test######################################
# extracted_path = "/home/nelson/projects/da_art_perception/data/dataset"
# label_suffix = '*label_raw.png'
# train =sum(['train' in path for path in tf.io.gfile.glob(extracted_path+'/**/**/**/'+label_suffix)])
# test = sum(['test' in path for path in tf.io.gfile.glob(extracted_path+'/**/**/**/'+label_suffix)])

# print('General.................................')
# print('Total:' + str(train+test))
# print('Train:' + str(train))
# print('Test:' + str(test))
# print('Test set is :' + str(test/(train+test)) + '%') 

# count = 0
# for location in tf.io.gfile.glob(extracted_path+'/**/*'):
#     count += len(tf.io.gfile.glob(location+'/**/*'+label_suffix))
#     print(location.split('/')[-2]+', '+location.split('/')[-1]+': '+str(len(tf.io.gfile.glob(location+'/**/*'+label_suffix))))
 

# json_paths = [json_path  
#         for list_path in tf.io.gfile.glob(extracted_path+'/**/**/**/*small.txt') 
#         for name in open(list_path).readlines()
#         for json_path in tf.io.gfile.glob(list_path[:list_path.rfind('/')]+'/'+name.replace('\n', '') +label_suffix)]


# train =sum(['train' in path for path in json_paths])
# test = sum(['test' in path for path in json_paths])
# print('Small.................................')
# print('Total:' + str(train+test))
# print('Train:' + str(train))
# print('Test:' + str(test))
# print('Test set is :' + str(test/(train+test)) + '%')  

# count = 0
# train_total=0
# val_total=0
# test_total=0
# for condition in ['/rain/', 'off-road/day/','/night/','/evening/',
#                   'unpaved/day/']:
#     train = sum([('train' in path and condition in path) for path in json_paths])
#     test = sum([('tes' in path and condition in path) for path in json_paths])
#     val = round(train*0.1)
#     train = train - val
#     total = train+test+val
#     print(condition)
#     print('Total:' + str(total))
#     print('Train:' + str(train))
#     print('Val:'   + str(val))
#     print('Test:'  + str(test))
#     print('train set is :' + str(round(1000*train/total)/10) + '%')  
#     print('val set is   :' + str(round(1000*val/total)/10) + '%')  
#     print('test set is  :' + str(round(1000*test/total)/10) + '%')  
#     count += total
#     train_total += train
#     val_total += val
#     test_total += test
# print('...........................................')
# print('Total :' + str(count))
# print('Train:' + str(train_total))
# print('Val:'   + str(val_total))
# print('Test:'  + str(test_total))
# print('train set is :' + str(round(1000*train_total/count)/10) + '%')  
# print('val set is   :' + str(round(1000*val_total/count)/10) + '%')  
# print('test set is  :' + str(round(1000*test_total/count)/10) + '%')  
# print('If 0 it''s OK :' + str(count-len(json_paths)))



########################### Split set into subset ######################################



# ###########################count train and test on small######################################
# extracted_path = "/home/nelson/projects/da_art_perception/data/dataset"
# label_suffix = '*label_raw.png'

# for list_path in tf.io.gfile.glob(extracted_path+'/**/**/**/*small.txt'):
#     print(list_path)
#     lineList = open(list_path).readlines()
#     for name in lineList:
#         search_name_path = list_path[:list_path.rfind('/')]+'/'+name.replace('\n', '')
#         for json_path in tf.io.gfile.glob(list_path[:list_path.rfind('/')]+'/'+name.replace('\n', '') +label_suffix):
            
###############################################################################

# # ###########################count class on geral######################################
# extracted_path = "/home/nelson/projects/da_art_perception/data/dataset"
# label_suffix = '*label_raw.png'
# image_suffix = '.jpg'
# json_suffix = '.json'


# countable_classes = ['animal', 'bike', 'bus', 'car', 'cone', 'moto', 'person', 'truck', 'dog']
# count = {}
# for c_class in countable_classes:
#     count[c_class] = {'train':0,'test':0}


# for json_path in tf.io.gfile.glob(extracted_path+'/**/**/**/*.json'):
#     for label in [shape['label'] for shape in json.load(open(json_path))['shapes']]:
#         #search ocurrence for each countble class
#         for c_class in count:
#             if c_class in label:
#                 if 'test' in tf.io.gfile.glob(json_path.split('.json')[0] + label_suffix)[0]:
#                     count[c_class]['test'] +=1
#                 else:
#                     count[c_class]['train'] +=1
#                 if c_class == 'dog':
#                     print(json_path)
           
       
# # #Total
# # {'animal': {'train': 22, 'test': 5},
# #   'bike': {'train': 32, 'test': 9},
# #   'bus': {'train': 84, 'test': 17},
# #   'car': {'train': 3281, 'test': 888},
# #   'cone': {'train': 106, 'test': 23},
# #   'dog': {'train': 0, 'test': 0},
# #   'moto': {'train': 85, 'test': 29},
# #   'person': {'train': 1495, 'test': 390},
# #   'truck': {'train': 126, 'test': 28}}
# # ###############################################################################


# ###############################################################################
# #########################count class on small##################################
# extracted_path = "/home/nelson/projects/da_art_perception/data/dataset"
# label_suffix = '*label_raw.png'
# image_suffix = '.jpg'
# json_suffix = '.json'


# countable_classes = ['animal', 'bike', 'bus', 'car', 'cone', 'moto', 'person', 'truck']
# count = {}
# for c_class in countable_classes:
#     count[c_class] = {'train':0,'test':0}


# for list_path in tf.io.gfile.glob(extracted_path+'/**/**/**/*small.txt'):
#     print(list_path)
#     lineList = open(list_path).readlines()
#     for name in lineList:
#         search_name_path = list_path[:list_path.rfind('/')]+'/'+name.replace('\n', '')
#         for json_path in tf.io.gfile.glob(search_name_path + json_suffix):
#             for label in [shape['label'] for shape in json.load(open(json_path))['shapes']]:
#                 #search ocurrence for each countble class
#                 for c_class in count:
#                     if c_class in label:
#                         if 'test' in tf.io.gfile.glob(search_name_path + label_suffix)[0]:
#                             count[c_class]['test'] +=1
#                         else:
#                             count[c_class]['train'] +=1

# print('train:' + str(
#     sum([ count[key]['train'] for key in count]))
#     )
# print('test:' + str(
#     sum([ count[key]['test'] for key in count]))
#     )
# print('test set is :' + str(
#     sum([ count[key]['test'] for key in count])/sum([ count[key]['train'] for key in count])
#     ) + '%')
# # #new
# # {'animal': {'train': 16, 'test': 3},
# #  'bike': {'train': 12, 'test': 2},
# #  'bus': {'train': 7, 'test': 2},
# #  'car': {'train': 1257, 'test': 265},
# #  'cone': {'train': 2, 'test': 2},
# #  'moto': {'train': 15, 'test': 4},
# #  'person': {'train': 506, 'test': 140},
# #  'truck': {'train': 27, 'test': 13}}
# # #old
# # {'animal': {'train': 17, 'test': 2},
# #  'bike': {'train': 11, 'test': 3},
# #  'bus': {'train': 8, 'test': 1},
# #  'car': {'train': 1596, 'test': 381},
# #  'cone': {'train': 74, 'test': 26},
# #  'moto': {'train': 17, 'test': 2},
# #  'person': {'train': 1189, 'test': 269},
# #  'truck': {'train': 30, 'test': 10}}
# ###############################################################################

# ###############################################################################
# batch_size = 1
# height = 2
# width = 2

# class MIoU(MeanIoU):
#   def __init__(self, num_classes, name=None, dtype=None):
#     super(MIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

#   def update_state(self, y_true, y_pred, sample_weight=None):
#     print(y_true)
#     print(y_pred)
#     y_pred_max=tf.math.argmax(input=y_pred, axis=-1, output_type=tf.dtypes.int64)

#     sample_weight = tf.where(tf.equal(y_pred_max, 2), 0, 1)
#     sample_weight_vector = tf.reshape(sample_weight, (batch_size*height*width))

#     result = super(MIoU, self).update_state(
#             y_true=y_true, 
#             y_pred=y_pred_max,
#             sample_weight=sample_weight_vector)

# #m = MIoU(num_classes=3)
# #m.update_state([[0, 2], [1, 1]], [[[0, 1, 0], [0, 0, 1]], [[0,1,0], [0,1,0]]])

# # path = root_path + '/unpaved/rain/jaua/'
# # search_fragment = 'c0_'
# # count += generate_list(path, search_fragment)

# #print(m.result().numpy())
# ###############################################################################



# ###############################################################################
# extracted_path = "/home/nelson/projects/da_art_perception/data/dataset"
# label_suffix = '*label_raw.png'
# image_suffix = '.jpg'

# train_img_list = []
# train_lbl_list = []
# test_img_list = []
# test_lbl_list = []


# for list_path in tf.io.gfile.glob(extracted_path+'/**/**/**/*small.txt'):
#     print(list_path)
#     lineList = open(list_path).readlines()
#     for name in lineList:
#         search_name_path = list_path[:list_path.rfind('/')]+'/'+name.replace('\n', '')
#         full_name_path = tf.io.gfile.glob(search_name_path + label_suffix)[0]
#         if 'train' in full_name_path:
#             train_img_list.append(search_name_path + image_suffix)
#             train_lbl_list.append(full_name_path)
#         elif 'test' in full_name_path:
#             test_img_list.append(search_name_path + image_suffix)
#             test_lbl_list.append(full_name_path)
# ###############################################################################



###############################################################################
# def generate_list(path, search_fragment):
#     my_list = glob.glob(path+'**.json')
#     count=0
#     with open(path+'small.txt', 'w') as f:
#         for item in my_list:
#             if search_fragment in item:
#                 f.write("%s\n" % item[item.rfind('/')+1:-5])
#                 count +=1
#             elif  'car-0' in [shape['label'] for shape in json.load(open(item))['shapes']] or 'person-0' in [shape['label'] for shape in json.load(open(item))['shapes']]:
#                 f.write("%s" % item[item.rfind('/')+1:-5])
#                 count +=1
#     return count

# root_path = '/home/nelson/projects/da_art_perception/data/dataset'
# count = 0


# path = root_path + '/off-road/day/cimatec-industrial/'
# search_fragment = '001_'
# count += generate_list(path, search_fragment)

# path = root_path + '/off-road/evening/cimatec-industrial/'
# search_fragment = '001cc'
# count += generate_list(path, search_fragment)

# path = root_path + '/off-road/rain/cimatec-industrial/'
# search_fragment = 'cim_'
# count += generate_list(path, search_fragment)



# path = root_path + '/unpaved/day/estrada-dos-tropeiros/'
# search_fragment = '001_'
# count += generate_list(path, search_fragment)

# path = root_path + '/unpaved/day/jaua/'
# search_fragment = 'jt_'
# count += generate_list(path, search_fragment)

# path = root_path + '/unpaved/day/praia-do-forte/'
# search_fragment = 'k1_'
# count += generate_list(path, search_fragment)



# path = root_path + '/unpaved/rain/estrada-dos-tropeiros/'
# search_fragment = 'c0_'
# count += generate_list(path, search_fragment)

# path = root_path + '/unpaved/rain/jaua/'
# search_fragment = 'c0_'
# count += generate_list(path, search_fragment)

# path = root_path + '/unpaved/rain/praia-do-forte/'
# search_fragment = 'd1_'
# count += generate_list(path, search_fragment)
###############################################################################