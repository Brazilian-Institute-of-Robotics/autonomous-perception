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

for model in models:
    seg_map , image, label = eval_one(model, path)
    vis_segmentation(image, np.array(seg_map, dtype="uint8"), label, model)