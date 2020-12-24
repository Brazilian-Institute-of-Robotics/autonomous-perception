#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:28:18 2019

@author: nelson
"""
import os
import tarfile
from six.moves import urllib

import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds
from tensorflow.keras.metrics import MeanIoU
from google_drive_downloader import GoogleDriveDownloader as gdd
import time



import sys
import os.path

sys.path.append(os.path.dirname(__file__) + '/..')
from datasets import off_road_small




class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.compat.v1.Session(graph=self.graph, config=config)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

  def run_np(self, image):
    """Runs inference on a single image.

    Args:
      image: A numpy raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    start = time.time()
    batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                                  feed_dict={self.INPUT_TENSOR_NAME: [image]})
    inference_time = time.time()-start
    seg_map = batch_seg_map[0]
    return seg_map, inference_time




class PSPNetModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'input_image:0'
  INPUT_H = 'input_h:0'
  INPUT_W = 'input_w:0'
  OUTPUT_TENSOR_NAME = 'ArgMax:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf1.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    config = tf1.ConfigProto()
    # device_count = {'CPU': 0}
    config.gpu_options.allow_growth = True
    self.sess = tf1.Session(graph=self.graph, config=config)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: np.asarray(resized_image),
                   self.INPUT_H: target_size[1],
                   self.INPUT_W: target_size[0]}
        )
    seg_map = batch_seg_map[0]
    return resized_image, seg_map

  def run_np(self, image):
    """Runs inference on a single image.

    Args:
      image: A numpy raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    start = time.time()
    batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                                  feed_dict={self.INPUT_TENSOR_NAME: image,
                                             self.INPUT_H: image.shape[0],
                                             self.INPUT_W: image.shape[1]}
                                  )
    inference_time = time.time()-start
    seg_map = batch_seg_map[0]
    return seg_map, inference_time




#model_name = 'pspnet101_cityscape'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
# model_name = 'xception71_dpc_cityscapes_trainfine'
# model_name = 'xception65_cityscapes_trainfine'


def print_result(classes, confusion_matrix, classes_ids):
    IoU_sum = 0
    IoU_wf = []
    mIoU_wf = 0
    nii_sum = 0
    ti_sum = 0
    IoU = []
    CPacc = []
    for class_id in range(len(confusion_matrix)):
        nii = confusion_matrix[class_id, class_id] #correctly inferred
        ti = confusion_matrix[:, class_id].sum()#ground-truth
        sumjnij = confusion_matrix[class_id, :].sum()#all classes inferred ad truth
        IoU.append(nii/(ti+sumjnij-nii))
        IoU_sum += IoU[-1]
        nii_sum += nii
        ti_sum += ti
        if ti > 0:
            CPacc.append(nii/ti)
            IoU_wf.append(IoU[-1]*(ti/confusion_matrix.sum()))
            mIoU_wf += IoU_wf[-1]
        else:
            CPacc.append(0)
        if class_id in classes_ids:
            print('class: '+ classes[class_id]['name']+'\t IoU: '+
                  str(round(IoU[-1]*10000)/100)+',\t CPacc: '+
                  str(str(round(CPacc[-1]*10000)/100)))

    print('mIoU: '+str(round(np.array(IoU)[classes_ids].mean()*10000)/100)
        + ', FWmIoU: ' + str(round(mIoU_wf*10000)/100)
        + ', mCPacc: ' + str(round(np.array(CPacc)[classes_ids].mean()*10000)/100)+'%'
        + ', Pacc: ' + str(round(nii_sum/ti_sum*10000)/100)+'%')




_MODEL_URLS = {
    'mnv2_cityscapes_train':
        'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz',
    'xception71_dpc_cityscapes_trainfine':
        'deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz',
    'xception65_cityscapes_trainfine':
        'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
    'pspnet101_cityscape':
        '1vZkk9nLvM9NNBCVCuEjnoXms30OMcZ8K',#/pspnet101_cityscape_2018_11_08.tar.gz'
}
_TARBALL_NAME = {
    'mnv2_cityscapes_train':
        'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz',
    'xception71_dpc_cityscapes_trainfine':
        'deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz',
    'xception65_cityscapes_trainfine':
        'deeplabv3_cityscapes_train_2018_02_06.tar.gz',
    'pspnet101_cityscape':
        'pspnet101_cityscape_2018_11_08.tar.gz',#/pspnet101_cityscape_2018_11_08.tar.gz'
}
# _TARBALL_NAME = 'model.tar.gz'

def load_model(model_name):
    if model_name!='pspnet101_cityscape':
        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    else:
        #_DOWNLOAD_URL_PREFIX = 'https://drive.google.com/'
        _DOWNLOAD_URL_PREFIX = ''
    
    
    # model_dir = tempfile.mkdtemp()
    model_dir = "/tmp/tmppjadxrbw"
    tf.io.gfile.makedirs(model_dir)
    
    download_path = os.path.join(model_dir, _TARBALL_NAME[model_name])
    

    if os.path.isfile(download_path):
        print ("File exist in "+download_path)
    else:
        print('downloading model '+model_name +', this might take a while...')
        
        if model_name=='pspnet101_cityscape':
            gdd.download_file_from_google_drive(file_id=_MODEL_URLS[model_name],dest_path=download_path,unzip=False)
        else:
            urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[model_name],  download_path)
        
        print('download completed! loading DeepLab model...')
        
        
    if model_name=='pspnet101_cityscape':
        model = PSPNetModel(download_path)
    else:
        model = DeepLabModel(download_path)
        
    
    print('model loaded successfully!')
    
    if model_name=='pspnet101_cityscape':
        height_image = 736
        width_image  = 1024
    else:
        height_image = 483
        width_image  = 769
    return model, height_image, width_image


    
def eval_models(model_name):

    
    [model, height_image, width_image] = load_model(model_name)
    
    
    
    batch_size=1
    dataset_name = "off_road_small" #"cityscape"#"citysmall"off_road_small freiburg_forest
    classes = off_road_small.classes
    n_classes = len(classes)
    
    #n_classes = 12
    
    # Construct a tf.data.Dataset
    info = tfds.builder(dataset_name).info
    print(info)
    [ds_test] = tfds.load(name=dataset_name, split=["test"], as_supervised=True)

    # Add normalize
    def _normalize_img(image, label):
        image = tf.cast(tf.image.resize(image, (height_image,width_image), method=tf.image.ResizeMethod.BILINEAR),dtype=tf.uint8)
        label = tf.cast(tf.image.resize(label, (height_image,width_image), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),dtype=tf.uint8)
        return (image, label)
    
    ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Build your input pipeline
    ds_test = ds_test.shuffle(124).batch(batch_size).prefetch(10)
    
    
    class MIoU(MeanIoU):
      def __init__(self, num_classes, name=None, dtype=None):
        super(MIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
    
      def update_state(self, y_true, y_pred, sample_weight=None):
        result = super(MIoU, self).update_state(
                y_true=y_true, 
                y_pred=tf.math.argmax(input=y_pred, axis=-1, output_type=tf.dtypes.int64), 
                sample_weight=sample_weight)
    
        return result
    
    
    miou = MeanIoU(num_classes=n_classes)
    
    i = 0
    for image, label in ds_test:
        print(i)
        i +=1;
        
        mask_cityscape, _ = model.run_np(image.numpy()[0])
        
        
        mask_cityscape = np.array(Image.fromarray(np.uint8(mask_cityscape)).resize((image.numpy().shape[2],image.numpy().shape[1])))
        #Convert infered result in a offroad mask
        #Zero is the ignore label. Every thing is considereted ignore
        mask_offroad = np.zeros(mask_cityscape.shape)
        
        mask_offroad[mask_cityscape==0]=1  #road
        mask_offroad[mask_cityscape==11]=3 #person #person
        mask_offroad[mask_cityscape==12]=3 #rider  #person
        mask_offroad[mask_cityscape==13]=2 #car
        mask_offroad[mask_cityscape==14]=4 #truck
        mask_offroad[mask_cityscape==15]=5 #bus
        mask_offroad[mask_cityscape==17]=7 #motocycle
        mask_offroad[mask_cityscape==18]=9 #bicycle
        
        miou.update_state(label[0,...,0], mask_offroad)
    
    
    weights = miou.get_weights()
    result = {'name':model_name}
    result['classes'] = classes
    result['confusion_matrix'] = weights[0].tolist()
    result['count_params'] = 0#cmsnet.count_params()
    return result

def eval_one(model_name, path):
    tf.config.set_visible_devices([], 'GPU')
    [model, height_image, width_image] = load_model(model_name)
    img = np.array(Image.open(path))
    label = np.array(Image.open(path.split(".jpg")[0]+"-train-label_raw.png")) 

    image = tf.cast(tf.image.resize(img, (height_image,width_image), method=tf.image.ResizeMethod.BILINEAR),dtype=tf.uint8)
    mask = model.run_np(image.numpy())
    
    mask_cityscape = np.array(Image.fromarray(np.uint8(mask)).resize((img.shape[1],img.shape[0])))
    #Convert infered result in a offroad mask
    #Zero is the ignore label. Every thing is considereted ignore
    mask_offroad = np.zeros(mask_cityscape.shape)
    
    mask_offroad[mask_cityscape==0]=1  #road
    mask_offroad[mask_cityscape==11]=3 #person #person
    mask_offroad[mask_cityscape==12]=3 #rider  #person
    mask_offroad[mask_cityscape==13]=2 #car
    mask_offroad[mask_cityscape==14]=4 #truck
    mask_offroad[mask_cityscape==15]=5 #bus
    mask_offroad[mask_cityscape==17]=7 #motocycle
    mask_offroad[mask_cityscape==18]=9 #bicycle
    return mask_offroad, img, label


    