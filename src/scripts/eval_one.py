#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 23:29:23 2020

@author: nelson
"""

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')
from evaluation.others import eval_one
from evaluation.trained import eval_one as eval_one_trained
from PIL import Image

build_path = os.path.dirname(__file__) + '/../../build/'




# Set up a colormap:
# use copy so that we do not mutate the global colormap instance
def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 2

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map, label, model):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_label = label_to_color_image(label).astype(np.uint8)
  plt.imshow(image)
  plt.imshow(seg_label, alpha=0.6)  
  plt.axis('off')
  plt.title('label overlay')

  plt.subplot(grid_spec[2])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.6)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.savefig(fname=build_path+model+'.jpeg', dpi=300, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
  plt.show()


LABEL_NAMES = np.asarray([
    'ignore', 'road', 'car', 'person', 'truck', 'bus', 'cone',
    'motocycle', 'animal', 'bicycle'
])



FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)





models = ['mnv2_cityscapes_train', 'pspnet101_cityscape', 'xception71_dpc_cityscapes_trainfine', 'xception65_cityscapes_trainfine']
path = "/home/nelson/projects/da_art_perception/data/dataset/off-road/day/cimatec-industrial/cmindtk001_000013.jpg"
path = "/home/nelson/projects/da_art_perception/data/dataset/unpaved/day/jaua/jt_008401.jpg"
path = "/home/nelson/projects/da_art_perception/data/dataset/unpaved/rain/estrada-dos-tropeiros/edtc0_000108.jpg"
#path = "/home/nelson/projects/da_art_perception/data/dataset/unpaved/rain/estrada-dos-tropeiros/edtc0_000218.jpg"
#daytime
path = '/home/nelson/projects/autonomous_perception/data/dataset/off-road/day/cimatec-industrial/cmindtk005_000034.jpg'
path = '/home/nelson/projects/autonomous_perception/data/dataset/off-road/day/cimatec-industrial/cmindtk001_000014.jpg'
#dusty
path = ''
#night
path = '/home/nelson/projects/autonomous_perception/data/dataset/off-road/night/cimatec-industrial/ci03c005cc_004845.jpg'
path = '/home/nelson/projects/autonomous_perception/data/dataset/off-road/night/cimatec-industrial/cmindtk02008_000222.jpg'

#night-dusty
path = '/home/nelson/projects/autonomous_perception/data/dataset/off-road/night/cimatec-industrial/cmindtk02010_000034.jpg'
path = '/home/nelson/projects/autonomous_perception/data/dataset/off-road/night/cimatec-industrial/cmindtk02008_000039.jpg'

#Others
others = False
if others == True:
    for model in models:
        seg_map , image, label = eval_one(model, path)
        vis_segmentation(image, np.array(seg_map, dtype="uint8"), label, model)
        
        
        
        
        

test_cases =[
                {'pooling' : 'aspp',
                  'backbonetype' : 'mobilenetv2',
                'output_stride' : 8,
                'residual_shortcut' : False, 
                'height_image' : 483,
                'width_image' : 769,
                'channels' : 3, 
                'crop_enable' : False, 
                'height_crop' : 483,
                'width_crop' : 769,
    
                'dataset_name' : 'off_road_small',
                'class_imbalance_correction' : False, 
                'data_augmentation' : True, 
                 
                'multigpu_enable' : True,
                'debug_en' : False,
                
                'batch_size' : 2,
                'epochs' : 200,
                'initial_epoch' : -1,
                'continue_traning' : False,
                'fine_tune_last' : False, 
                
                'base_learning_rate' : 0.007, 
                'learning_power' : 0.98, 
                'decay_steps' : 1,
                'learning_rate_decay_step' : 4,
                'decay' : 5**(-4),
                'name': 'CM2'
                },
                ]

t = test_cases[0]

#CMSNet  
cmsnet = True      
if cmsnet == True:
    image = np.array(Image.open(path))


    label = np.array(Image.open(path.split(".jpg")[0]+"-train-label_raw.png")) 
    
    seg_r = eval_one_trained(pooling = t['pooling'], 
              backbonetype=t['backbonetype'], 
              output_stride = t['output_stride'], 
              residual_shortcut = t['residual_shortcut'],
              height_image = t['height_image'], 
              width_image = t['width_image'], 
              channels = t['channels'], 
              crop_enable = t['crop_enable'], 
              height_crop = t['height_crop'], 
              width_crop = t['width_crop'],
              debug_en = t['debug_en'], 
              dataset_name = t['dataset_name'], 
              class_imbalance_correction = t['class_imbalance_correction'],
              data_augmentation = t['data_augmentation'], 
              multigpu_enable = t['multigpu_enable'], 
              batch_size = t['batch_size'], epochs = t['epochs'], 
              initial_epoch = t['initial_epoch'],
              continue_traning = t['continue_traning'], 
              fine_tune_last = t['fine_tune_last'],
              base_learning_rate = t['base_learning_rate'], 
              learning_power = t['learning_power'], 
              decay_steps = t['decay_steps'], 
              learning_rate_decay_step = t['learning_rate_decay_step'],
              decay = t['decay'],
              image = image)
    
    seg_map = np.argmax(seg_r[0],axis=-1)
    seg_map = np.array(Image.fromarray(np.uint8(seg_map)).resize((image.shape[1],image.shape[0])))
    vis_segmentation(image, np.array(seg_map, dtype="uint8"), label, t['name'])
#     cmsnet.summary()
#     cmsnet.load_weights(weights_path)
    
    
    
#     label = np.array(Image.open(image_path.split(".jpg")[0]+"-train-label_raw.png")) 

#     image = tf.cast(tf.image.resize(img, (height_image,width_image), method=tf.image.ResizeMethod.BILINEAR),dtype=tf.uint8)
#     mask = model.run_np(image.numpy())
    
#     mask_cityscape = np.array(Image.fromarray(np.uint8(mask)).resize((img.shape[1],img.shape[0])))
#     #Convert infered result in a offroad mask
#     #Zero is the ignore label. Every thing is considereted ignore
#     mask_offroad = np.zeros(mask_cityscape.shape)
    
#     mask_offroad[mask_cityscape==0]=1  #road
#     mask_offroad[mask_cityscape==11]=3 #person #person
#     mask_offroad[mask_cityscape==12]=3 #rider  #person
#     mask_offroad[mask_cityscape==13]=2 #car
#     mask_offroad[mask_cityscape==14]=4 #truck
#     mask_offroad[mask_cityscape==15]=5 #bus
#     mask_offroad[mask_cityscape==17]=7 #motocycle
#     mask_offroad[mask_cityscape==18]=9 #bicycle
#     return mask_offroad, img, label
    
    
    