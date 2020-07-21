#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:51:42 2020

@author: nelson
"""
import tensorflow as tf







# # ###########################count total train and test######################################
extracted_path = "/home/nelson/projects/da_art_perception/data/dataset"
label_suffix = '*label_raw.png'
train =sum(['train' in path for path in tf.io.gfile.glob(extracted_path+'/**/**/**/'+label_suffix)])
test = sum(['test' in path for path in tf.io.gfile.glob(extracted_path+'/**/**/**/'+label_suffix)])

print('General.................................')
print('Total:' + str(train+test))
print('Train:' + str(train))
print('Test:' + str(test))
print('Test set is :' + str(test/(train+test)) + '%') 

count = 0
for location in tf.io.gfile.glob(extracted_path+'/**/*'):
    count += len(tf.io.gfile.glob(location+'/**/*'+label_suffix))
    print(location.split('/')[-2]+', '+location.split('/')[-1]+': '+str(len(tf.io.gfile.glob(location+'/**/*'+label_suffix))))
 

json_paths = [json_path  
        for list_path in tf.io.gfile.glob(extracted_path+'/**/**/**/*small.txt') 
        for name in open(list_path).readlines()
        for json_path in tf.io.gfile.glob(list_path[:list_path.rfind('/')]+'/'+name.replace('\n', '') +label_suffix)]


train =sum(['train' in path for path in json_paths])
test = sum(['test' in path for path in json_paths])
print('Small.................................')
print('Total:' + str(train+test))
print('Train:' + str(train))
print('Test:' + str(test))
print('Test set is :' + str(test/(train+test)) + '%')  

count = 0
train_total=0
val_total=0
test_total=0
for condition in ['/rain/', 'off-road/day/','/night/','/evening/',
                  'unpaved/day/']:
    train = sum([('train' in path and condition in path) for path in json_paths])
    test = sum([('tes' in path and condition in path) for path in json_paths])
    val = round(train*0.1)
    train = train - val
    total = train+test+val
    print(condition)
    print('Total:' + str(total))
    print('Train:' + str(train))
    print('Val:'   + str(val))
    print('Test:'  + str(test))
    print('train set is :' + str(round(1000*train/total)/10) + '%')  
    print('val set is   :' + str(round(1000*val/total)/10) + '%')  
    print('test set is  :' + str(round(1000*test/total)/10) + '%')  
    count += total
    train_total += train
    val_total += val
    test_total += test
print('...........................................')
print('Total :' + str(count))
print('Train:' + str(train_total))
print('Val:'   + str(val_total))
print('Test:'  + str(test_total))
print('train set is :' + str(round(1000*train_total/count)/10) + '%')  
print('val set is   :' + str(round(1000*val_total/count)/10) + '%')  
print('test set is  :' + str(round(1000*test_total/count)/10) + '%')  
print('If 0 it''s OK :' + str(count-len(json_paths)))




###########################count train and test on small######################################
extracted_path = "/home/nelson/projects/da_art_perception/data/dataset"
label_suffix = '*label_raw.png'

for list_path in tf.io.gfile.glob(extracted_path+'/**/**/**/*small.txt'):
    print(list_path)
    lineList = open(list_path).readlines()
    for name in lineList:
        search_name_path = list_path[:list_path.rfind('/')]+'/'+name.replace('\n', '')
        for json_path in tf.io.gfile.glob(list_path[:list_path.rfind('/')]+'/'+name.replace('\n', '') +label_suffix):
            
