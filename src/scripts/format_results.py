#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 19:14:57 2020
This script is used to manipulated and format the results

@author: nelson
"""
import  json
import numpy as np
import os, sys
sys.path.append(os.path.dirname(__file__) + '/..')
from evaluation.trained import print_result,print_result2


build_path = os.path.dirname(__file__) + '/../../build/'



with open(build_path+'reslut.txt', 'r') as file:
   results = json.loads(file.read())
   
 
# maxiou=0
# for result, i in zip(results, range(len(results))):
#     if result['mIoU: '] > maxiou:
#         maxiou =  (result['class']['road']['IoU']+result['class']['car']['IoU']+result['class']['person']['IoU']+result['class']['truck']['IoU'])/4
#         index = i
        
classes_ids = [1, 2, 3, 4, 0]

with open(build_path+'reslut.txt', 'r') as file:
    loaded_results = json.loads(file.read())


# for result in loaded_results:
#     result['confusion_matrix']
#     result['classes']
#     print(result['name'])
#     print("Params: "+str(result['count_params']))
#     print_result(result['classes'], np.array(result['confusion_matrix']), classes_ids)

for result in loaded_results:
    print(result['name'])
    print("Params: "+str(result['count_params']))
    print_result2(result['classes'], np.array(result['confusion_matrix']), classes_ids)