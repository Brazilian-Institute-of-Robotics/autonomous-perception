#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:28:18 2019

@author: nelson
"""
import os

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow.compat.v2 as tf
import sys
import os.path

sys.path.append(os.path.dirname(__file__) + '/..')
from evaluation.others import eval_models, print_result
import json

build_path = os.path.dirname(__file__) + '/../../build/'


models = ['mnv2_cityscapes_train', 'pspnet101_cityscape', 'xception71_dpc_cityscapes_trainfine', 'xception65_cityscapes_trainfine']
# results = []
# for model_name in models:
#     #Traing
#     r = eval_models(model_name)
#     results.append(r)
#     tf.keras.backend.clear_session()
    
# with open(build_path+'reslut_model.txt', 'w') as file:
#     file.write(json.dumps(results))


classes_ids = [0, 1, 2, 3]

with open(build_path+'reslut_model.txt', 'r') as file:
    loaded_results = json.loads(file.read())


for result in loaded_results:
    result['confusion_matrix']
    result['classes']
    print(result['name'])
    print("Params: "+str(result['count_params']))
    print_result(result['classes'], np.array(result['confusion_matrix']), classes_ids)