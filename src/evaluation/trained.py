# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import MeanIoU


import numpy as np
import glob

import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')
from models.cmsnet import CMSNet
from datasets import citysmall
from datasets import cityscape
from datasets import off_road_small
from datasets import freiburg_forest
from datasets import off_road_testsets
from utils import augmentation as aug
import tensorflow as tf
import imgaug as ia
import imgaug.augmenters as iaa

# build_path = os.path.dirname(__file__) + '/../../build/'
build_path = os.path.dirname(__file__) + '/../../trained/'


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


def print_result2(classes, confusion_matrix, classes_ids):
    for class_id in range(len(confusion_matrix)):
        if class_id in classes_ids:
            print(classes[class_id]['name']+' & ', end = '')
    
    print('')
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
            print(str(round(IoU[-1]*10000)/100)+' & ', end = '')
            
    print('')
    print('mIoU: '+str(round(np.array(IoU)[classes_ids].mean()*10000)/100)
        + ', FWmIoU: ' + str(round(mIoU_wf*10000)/100)
        + ', mCPacc: ' + str(round(np.array(CPacc)[classes_ids].mean()*10000)/100)+'%'
        + ', Pacc: ' + str(round(nii_sum/ti_sum*10000)/100)+'%')


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
        if IoU[-1] != IoU[-1] and nii == 0: #Fix IoU Nan problem
            IoU[-1] = 0
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

    mIoU = round(np.array(IoU)[classes_ids].mean()*10000)/100
    FWmIoU = round(mIoU_wf*10000)/100
    mCPacc = round(np.array(CPacc)[classes_ids].mean()*10000)/100
    Pacc = round(nii_sum/ti_sum*10000)/100
    print('mIoU: '+str(mIoU)
        + ', FWmIoU: ' + str(FWmIoU)
        + ', mCPacc: ' + str(mCPacc)+'%'
        + ', Pacc: ' + str(Pacc)+'%')
    return mIoU, FWmIoU, mCPacc, Pacc


def eval_models(pooling = 'global',
    backbonetype ='mobilenetv2',
    output_stride = 8,
    residual_shortcut = False, 
    height_image = 448, 
    width_image = 448, 
    channels = 3, 
    crop_enable = False,
    height_crop = 448, 
    width_crop = 448, 
    
    debug_en = False, 
    dataset_name = 'freiburg_forest', 
    class_imbalance_correction = False, 
    data_augmentation = True, 
     
    multigpu_enable = True,
    
    batch_size = 32, 
    epochs = 300, 
    initial_epoch = -1, 
    continue_traning = False, 
    fine_tune_last = False, 
    
    base_learning_rate = 0.007, 
    learning_power = 0.98, 
    decay_steps = 1,
    learning_rate_decay_step = 300, 
    
    decay = 5**(-4), 
    ):
    
    
    fold_name = (backbonetype+'s'+str(output_stride)+'_pooling_' + pooling
             +('_residual_shortcut' if residual_shortcut else '')+'_ep'+str(epochs)
             +('_crop_'+str(height_crop)+'x'+str(width_crop) if crop_enable else '_')
             +('from' if crop_enable else '')+str(height_image)+'x'+str(width_image)
             +'_'+('wda_' if data_augmentation else 'nda_')
             +('wcic_' if class_imbalance_correction else 'ncic_')
             +('wft_' if fine_tune_last else 'nft_')+dataset_name+'_b'
             +str(batch_size)+('_n' if multigpu_enable else '_1')+'gpu')
    
    
    
    #Override params for eval
    # multigpu_enable = False
    # batch_size      = 2


    ## Check the number of labels
    if dataset_name == 'cityscape':
        classes = cityscape.classes
    elif dataset_name == 'citysmall':
        classes = citysmall.classes
    elif dataset_name == 'off_road_small':
        classes = off_road_small.classes
    elif dataset_name == 'freiburg_forest':
        classes = freiburg_forest.classes
    
    n_classes = len(classes)
    

    if crop_enable:
        assert(height_crop == width_crop, "When crop is enable height_crop should be equals to width_crop")
        assert(height_crop <= height_image, "When crop is enable height_crop should be less than or equals to height_image")
        assert(width_crop <= width_image, "When crop is enable height_crop should be less than or equals to width_image")
    else:
        height_crop = height_image
        width_crop = width_image
    
    
    # Construct a tf.data.Dataset
    info = tfds.builder(dataset_name).info
    print(info)
    [ds_test] = tfds.load(name=dataset_name, split=["test"], as_supervised=True)
    # Add normalize
    def _normalize_img(image, label):
        image = tf.cast(image, tf.float32)/127.5 - 1
        if crop_enable:
            y1 = tf.random.uniform(shape=[], minval = 0., maxval = (height_image-height_crop)/height_image, dtype=tf.float32)
            x1 = tf.random.uniform(shape=[], minval = 0., maxval = (width_image-width_crop)/width_image, dtype=tf.float32)
    
            y2 = y1 + (height_crop/height_image)
            x2 = x1 + (width_crop/width_image)
    
            boxes = [[y1, x1, y2, x2]]
            image = tf.image.crop_and_resize([image], boxes, box_indices = [0], crop_size = (height_crop, width_crop), method=tf.image.ResizeMethod.BILINEAR)[0]
            label = tf.cast(tf.image.crop_and_resize([label], boxes, box_indices = [0], crop_size = (height_crop, width_crop), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0],dtype=tf.uint8)
        else:
            image = tf.image.resize(image, (height_image,width_image), method=tf.image.ResizeMethod.BILINEAR)
            label = tf.image.resize(label, (height_image,width_image), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return (image, label)
    
    ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.shuffle(124).batch(batch_size).prefetch(10)
    
    test_steps=int(round(info.splits['test'].num_examples/batch_size))
    
    class MIoU(MeanIoU):
        def __init__(self, num_classes, name=None, dtype=None):
            super(MIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
    
        def update_state(self, y_true, y_pred, sample_weight=None):
            return super(MIoU, self).update_state(
                y_true=y_true, 
                y_pred=tf.math.argmax(input=y_pred, axis=-1, output_type=tf.dtypes.int64), 
                sample_weight=sample_weight)
    

    
    if multigpu_enable:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels),
                            num_classes=n_classes, output_stride=output_stride,
                            pooling=pooling, residual_shortcut=residual_shortcut,
                            backbonetype=backbonetype)
            cmsnet.summary()
            #cmsnet.mySummary()
    
            optimizer = SGD(momentum=0.9, nesterov=True)
            miou = MIoU(num_classes=n_classes)
    
            cmsnet.compile(optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode="temporal",
                          metrics=['accuracy', miou]
                          )
    else:
        cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels),
                        num_classes=n_classes, output_stride=output_stride, pooling=pooling,
                        residual_shortcut=residual_shortcut, backbonetype=backbonetype)
        cmsnet.summary()
        
        optimizer = SGD(momentum=0.9, nesterov=True)
    
        miou = MIoU(num_classes=n_classes)
        cmsnet.compile(optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode="temporal",
                      metrics=['accuracy', miou])
    
    
    class_imbalance_correction
    

    
    #fold_name = 's16_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_aspp_20190815-151551'
    
    # Define the Keras TensorBoard callback.
    continue_traning = True
    if continue_traning:
        logdir=build_path+"logs/fit/" + fold_name #Continue
        weights_path = glob.glob(logdir+'*/weights.last*')[0]
        logdir = weights_path[:weights_path.find('/weights.')]
    
        print('Continuing train from '+ weights_path)
        if multigpu_enable:
            with strategy.scope():
                cmsnet.load_weights(weights_path)
        else:
            cmsnet.load_weights(weights_path)
    else:
        logdir=build_path+"logs/fit/" + fold_name+'_'+datetime.now().strftime("%Y%m%d-%H%M%S")
    
    
    
    
    
    result = cmsnet.evaluate(ds_test, use_multiprocessing=True, steps=test_steps)
    
    
    
    if multigpu_enable:
        with strategy.scope():
            weights = miou.get_weights()
    else:
        weights = miou.get_weights()
    

    
    result = {'name':fold_name}
    result['classes'] = classes
    result['confusion_matrix'] = weights[0].tolist()
    result['count_params'] = cmsnet.count_params()
    return result




def eval_models_impairments(pooling = 'global',
    backbonetype ='mobilenetv2',
    output_stride = 8,
    residual_shortcut = False, 
    height_image = 448, 
    width_image = 448, 
    channels = 3, 
    crop_enable = False,
    height_crop = 448, 
    width_crop = 448, 
    
    debug_en = False, 
    dataset_name = 'freiburg_forest', 
    class_imbalance_correction = False, 
    data_augmentation = True, 
     
    multigpu_enable = True,
    
    batch_size = 32, 
    epochs = 300, 
    initial_epoch = -1, 
    continue_traning = False, 
    fine_tune_last = False, 
    
    base_learning_rate = 0.007, 
    learning_power = 0.98, 
    decay_steps = 1,
    learning_rate_decay_step = 300, 
    
    decay = 5**(-4), 
    impairment = "noise", 
    intensity = 20
    ):
    
    
    fold_name = (backbonetype+'s'+str(output_stride)+'_pooling_' + pooling
             +('_residual_shortcut' if residual_shortcut else '')+'_ep'+str(epochs)
             +('_crop_'+str(height_crop)+'x'+str(width_crop) if crop_enable else '_')
             +('from' if crop_enable else '')+str(height_image)+'x'+str(width_image)
             +'_'+('wda_' if data_augmentation else 'nda_')
             +('wcic_' if class_imbalance_correction else 'ncic_')
             +('wft_' if fine_tune_last else 'nft_')+dataset_name+'_b'
             +str(batch_size)+('_n' if multigpu_enable else '_1')+'gpu')
    
    
    
    #Override params for eval
    # multigpu_enable = False
    # batch_size      = 2


    ## Check the number of labels
    if dataset_name == 'cityscape':
        classes = cityscape.classes
    elif dataset_name == 'citysmall':
        classes = citysmall.classes
    elif dataset_name == 'off_road_small':
        classes = off_road_small.classes
    elif dataset_name == 'freiburg_forest':
        classes = freiburg_forest.classes
    
    n_classes = len(classes)
    

    if crop_enable:
        assert(height_crop == width_crop, "When crop is enable height_crop should be equals to width_crop")
        assert(height_crop <= height_image, "When crop is enable height_crop should be less than or equals to height_image")
        assert(width_crop <= width_image, "When crop is enable height_crop should be less than or equals to width_image")
    else:
        height_crop = height_image
        width_crop = width_image
    
    
    # Construct a tf.data.Dataset
    info = tfds.builder(dataset_name).info
    print(info)
    [ds_test] = tfds.load(name=dataset_name, split=["test"], as_supervised=True)
    # Add normalize
    def _normalize_img(image, label):
        image = tf.cast(image, tf.float32)/127.5 - 1
        if crop_enable:
            y1 = tf.random.uniform(shape=[], minval = 0., maxval = (height_image-height_crop)/height_image, dtype=tf.float32)
            x1 = tf.random.uniform(shape=[], minval = 0., maxval = (width_image-width_crop)/width_image, dtype=tf.float32)
    
            y2 = y1 + (height_crop/height_image)
            x2 = x1 + (width_crop/width_image)
    
            boxes = [[y1, x1, y2, x2]]
            image = tf.image.crop_and_resize([image], boxes, box_indices = [0], crop_size = (height_crop, width_crop), method=tf.image.ResizeMethod.BILINEAR)[0]
            label = tf.cast(tf.image.crop_and_resize([label], boxes, box_indices = [0], crop_size = (height_crop, width_crop), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0],dtype=tf.uint8)
        else:
            image = tf.image.resize(image, (height_image,width_image), method=tf.image.ResizeMethod.BILINEAR)
            label = tf.image.resize(label, (height_image,width_image), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return (image, label)
    
    
    def _noise(img, lbl):
        rms = tf.sqrt(tf.math.reduce_mean(tf.pow(img,2.0)))
        std = rms*intensity/100
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=std, dtype=tf.float32)
        img = img + noise
        return img, lbl
    
    seq = iaa.Sequential([iaa.Fog()])
    def _aug_img(img):
        img = seq.augment_images([img.numpy()])
        return img[0]
    
    def _fog(img, lbl):
        img = tf.py_function(_aug_img, [img], tf.uint8)
        img.set_shape((1208, 1920, 3))
        return img, lbl
    
    
    if impairment == "noise":
        ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #SNR = mean^2/std^2
        #SNRdb = 10Log10(SNR) -> 10^(SNRdb/10) = SNR -> 10^(SNRdb/10) = mean^2/std^2
        #std^2 = mean^2/10^(SNRdb/10) -> std = sqrt(mean^2/(10^(SNRdb/10)))
        ds_test = ds_test.map(_noise, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    elif impairment == "fog":
        ds_test = ds_test.map(lambda img, lbl: tf.cond(tf.random.uniform([], 0, 1) < intensity/100, 
                                                      lambda: _fog(img, lbl), 
                                                      lambda: (img, lbl)),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
        ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if debug_en:
        aug.plot_images(ds_test, n_images=8, samples_per_image=3, classes = classes)
    
            
    ds_test = ds_test.shuffle(124).batch(batch_size).prefetch(10)
    test_steps=int(round(info.splits['test'].num_examples/batch_size))
    
    class MIoU(MeanIoU):
        def __init__(self, num_classes, name=None, dtype=None):
            super(MIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
    
        def update_state(self, y_true, y_pred, sample_weight=None):
            return super(MIoU, self).update_state(
                y_true=y_true, 
                y_pred=tf.math.argmax(input=y_pred, axis=-1, output_type=tf.dtypes.int64), 
                sample_weight=sample_weight)
    

    
    if multigpu_enable:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels),
                            num_classes=n_classes, output_stride=output_stride,
                            pooling=pooling, residual_shortcut=residual_shortcut,
                            backbonetype=backbonetype)
            cmsnet.summary()
            #cmsnet.mySummary()
    
            optimizer = SGD(momentum=0.9, nesterov=True)
            miou = MIoU(num_classes=n_classes)
    
            cmsnet.compile(optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode="temporal",
                          metrics=['accuracy', miou]
                          )
    else:
        cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels),
                        num_classes=n_classes, output_stride=output_stride, pooling=pooling,
                        residual_shortcut=residual_shortcut, backbonetype=backbonetype)
        cmsnet.summary()
        
        optimizer = SGD(momentum=0.9, nesterov=True)
    
        miou = MIoU(num_classes=n_classes)
        cmsnet.compile(optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode="temporal",
                      metrics=['accuracy', miou])
    
    
    class_imbalance_correction
    

    
    #fold_name = 's16_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_aspp_20190815-151551'
    
    # Define the Keras TensorBoard callback.
    continue_traning = True
    if continue_traning:
        logdir=build_path+"logs/fit/" + fold_name #Continue
        weights_path = glob.glob(logdir+'*/weights.last*')[0]
        logdir = weights_path[:weights_path.find('/weights.')]
    
        print('Continuing train from '+ weights_path)
        if multigpu_enable:
            with strategy.scope():
                cmsnet.load_weights(weights_path)
        else:
            cmsnet.load_weights(weights_path)
    else:
        logdir=build_path+"logs/fit/" + fold_name+'_'+datetime.now().strftime("%Y%m%d-%H%M%S")
    
    
    
    
    
    result = cmsnet.evaluate(ds_test, use_multiprocessing=True, steps=test_steps)
    
    
    
    if multigpu_enable:
        with strategy.scope():
            weights = miou.get_weights()
    else:
        weights = miou.get_weights()
    

    
    result = {'name':fold_name}
    result['classes'] = classes
    result['confusion_matrix'] = weights[0].tolist()
    result['count_params'] = cmsnet.count_params()
    return result



def eval_models_conditions(pooling = 'global',
    backbonetype ='mobilenetv2',
    output_stride = 8,
    residual_shortcut = False, 
    height_image = 448, 
    width_image = 448, 
    channels = 3, 
    crop_enable = False,
    height_crop = 448, 
    width_crop = 448,  
    class_imbalance_correction = False, 
    data_augmentation = True, 
    multigpu_enable = True,
    batch_size = 32, 
    epochs = 300, 
    fine_tune_last = False, 
    splits = ("evening[:100%]"+
              "+rain[:100%]"+
              "+day[:100%]"+
              "+day_offroad_clean[:100%]"+
              "+day_offroad_dusty[:100%]"+
              "+night_offroad_clean[:100%]"+
              "+night_offroad_dusty[:100%]")
    ):
    dataset_name = 'off_road_small'
    
    
    fold_name = (backbonetype+'s'+str(output_stride)+'_pooling_' + pooling
             +('_residual_shortcut' if residual_shortcut else '')+'_ep'+str(epochs)
             +('_crop_'+str(height_crop)+'x'+str(width_crop) if crop_enable else '_')
             +('from' if crop_enable else '')+str(height_image)+'x'+str(width_image)
             +'_'+('wda_' if data_augmentation else 'nda_')
             +('wcic_' if class_imbalance_correction else 'ncic_')
             +('wft_' if fine_tune_last else 'nft_')+dataset_name+'_b'
             +str(batch_size)+('_n' if multigpu_enable else '_1')+'gpu')


    ## Check the number of labels
    dataset_name = 'off_road_testsets'
    classes = off_road_testsets.classes
    n_classes = len(classes)
    

    if crop_enable:
        assert(height_crop == width_crop, "When crop is enable height_crop should be equals to width_crop")
        assert(height_crop <= height_image, "When crop is enable height_crop should be less than or equals to height_image")
        assert(width_crop <= width_image, "When crop is enable height_crop should be less than or equals to width_image")
    else:
        height_crop = height_image
        width_crop = width_image
    
    
    # Construct a tf.data.Dataset
    info = tfds.builder(dataset_name).info
    print(info)
    
    #Create satasets
    ds_test = tfds.load(name=dataset_name, split=splits, as_supervised=True)
    # Add normalize
    def _normalize_img(image, label):
        image = tf.cast(image, tf.float32)/127.5 - 1
        if crop_enable:
            y1 = tf.random.uniform(shape=[], minval = 0., maxval = (height_image-height_crop)/height_image, dtype=tf.float32)
            x1 = tf.random.uniform(shape=[], minval = 0., maxval = (width_image-width_crop)/width_image, dtype=tf.float32)
    
            y2 = y1 + (height_crop/height_image)
            x2 = x1 + (width_crop/width_image)
    
            boxes = [[y1, x1, y2, x2]]
            image = tf.image.crop_and_resize([image], boxes, box_indices = [0], crop_size = (height_crop, width_crop), method=tf.image.ResizeMethod.BILINEAR)[0]
            label = tf.cast(tf.image.crop_and_resize([label], boxes, box_indices = [0], crop_size = (height_crop, width_crop), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0],dtype=tf.uint8)
        else:
            image = tf.image.resize(image, (height_image,width_image), method=tf.image.ResizeMethod.BILINEAR)
            label = tf.image.resize(label, (height_image,width_image), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return (image, label)
    
    ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.repeat().shuffle(124).batch(batch_size).prefetch(10)
    test_steps=int(round(info.splits[splits].num_examples/batch_size))

    class MIoU(MeanIoU):
        def __init__(self, num_classes, name=None, dtype=None):
            super(MIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
    
        def update_state(self, y_true, y_pred, sample_weight=None):
            return super(MIoU, self).update_state(
                y_true=y_true, 
                y_pred=tf.math.argmax(input=y_pred, axis=-1, output_type=tf.dtypes.int64), 
                sample_weight=sample_weight)
    
    if multigpu_enable:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels),
                            num_classes=n_classes, output_stride=output_stride,
                            pooling=pooling, residual_shortcut=residual_shortcut,
                            backbonetype=backbonetype)
            cmsnet.summary()
            #cmsnet.mySummary()
    
            optimizer = SGD(momentum=0.9, nesterov=True)
            miou = MIoU(num_classes=n_classes)
    
            cmsnet.compile(optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode="temporal",
                          metrics=['accuracy', miou]
                          )
    else:
        cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels),
                        num_classes=n_classes, output_stride=output_stride, pooling=pooling,
                        residual_shortcut=residual_shortcut, backbonetype=backbonetype)
        cmsnet.summary()
        
        optimizer = SGD(momentum=0.9, nesterov=True)
    
        miou = MIoU(num_classes=n_classes)
        cmsnet.compile(optimizer, loss='sparse_categorical_crossentropy', sample_weight_mode="temporal",
                      metrics=['accuracy', miou])
    
    
    # Define the Keras TensorBoard callback.
    continue_traning = True
    if continue_traning:
        logdir=build_path+"logs/fit/" + fold_name #Continue
        weights_path = glob.glob(logdir+'*/weights.last*')[0]
        logdir = weights_path[:weights_path.find('/weights.')]
    
        print('Continuing train from '+ weights_path)
        if multigpu_enable:
            with strategy.scope():
                cmsnet.load_weights(weights_path)
        else:
            cmsnet.load_weights(weights_path)
    else:
        logdir=build_path+"logs/fit/" + fold_name+'_'+datetime.now().strftime("%Y%m%d-%H%M%S")
    
    result = cmsnet.evaluate(ds_test, use_multiprocessing=True, steps=test_steps)
    
    if multigpu_enable:
        with strategy.scope():
            weights = miou.get_weights()
    else:
        weights = miou.get_weights()
    
    result = {'name':fold_name}
    result['classes'] = classes
    result['confusion_matrix'] = weights[0].tolist()
    result['count_params'] = cmsnet.count_params()
    return result


def eval_models2(pooling = 'global',
    backbonetype ='mobilenetv2',
    output_stride = 8,
    residual_shortcut = False, 
    height_image = 448, 
    width_image = 448, 
    channels = 3, 
    crop_enable = False,
    height_crop = 448, 
    width_crop = 448, 
    
    debug_en = False, 
    dataset_name = 'freiburg_forest', 
    class_imbalance_correction = False, 
    data_augmentation = True, 
     
    multigpu_enable = True,
    
    batch_size = 32, 
    epochs = 300, 
    initial_epoch = -1, 
    continue_traning = False, 
    fine_tune_last = False, 
    
    base_learning_rate = 0.007, 
    learning_power = 0.98, 
    decay_steps = 1,
    learning_rate_decay_step = 300, 
    
    decay = 5**(-4), 
    ):
    
    

    
    
    
    #Override params for eval
    # multigpu_enable = False
    # batch_size      = 2


    ## Check the number of labels
    if dataset_name == 'cityscape':
        classes = cityscape.classes
    elif dataset_name == 'citysmall':
        classes = citysmall.classes
    elif dataset_name == 'off_road_small':
        classes = off_road_small.classes
    elif dataset_name == 'freiburg_forest':
        classes = freiburg_forest.classes
    
    n_classes = len(classes)
    

    if crop_enable:
        assert(height_crop == width_crop, "When crop is enable height_crop should be equals to width_crop")
        assert(height_crop <= height_image, "When crop is enable height_crop should be less than or equals to height_image")
        assert(width_crop <= width_image, "When crop is enable height_crop should be less than or equals to width_image")
    else:
        height_crop = height_image
        width_crop = width_image
    
    
    # Construct a tf.data.Dataset
    info = tfds.builder(dataset_name).info
    print(info)
    [ds_test] = tfds.load(name=dataset_name, split=["test"], as_supervised=True)
    # Add normalize
    def _normalize_img(image, label):
        image = tf.cast(image, tf.float32)/127.5 - 1
        if crop_enable:
            y1 = tf.random.uniform(shape=[], minval = 0., maxval = (height_image-height_crop)/height_image, dtype=tf.float32)
            x1 = tf.random.uniform(shape=[], minval = 0., maxval = (width_image-width_crop)/width_image, dtype=tf.float32)
    
            y2 = y1 + (height_crop/height_image)
            x2 = x1 + (width_crop/width_image)
    
            boxes = [[y1, x1, y2, x2]]
            image = tf.image.crop_and_resize([image], boxes, box_indices = [0], crop_size = (height_crop, width_crop), method=tf.image.ResizeMethod.BILINEAR)[0]
            label = tf.cast(tf.image.crop_and_resize([label], boxes, box_indices = [0], crop_size = (height_crop, width_crop), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0],dtype=tf.uint8)
        else:
            image = tf.image.resize(image, (height_image,width_image), method=tf.image.ResizeMethod.BILINEAR)
            label = tf.image.resize(label, (height_image,width_image), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return (image, label)
    
    ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.shuffle(124).batch(batch_size).prefetch(10)
    
    test_steps=int(round(info.splits['test'].num_examples/batch_size))
    
    class MIoU(MeanIoU):
        def __init__(self, num_classes, name=None, dtype=None):
            super(MIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
    
        def update_state(self, y_true, y_pred, sample_weight=None):
            return super(MIoU, self).update_state(
                y_true=y_true, 
                y_pred=tf.math.argmax(input=y_pred, axis=-1, output_type=tf.dtypes.int64), 
                sample_weight=sample_weight)
        
        
        
    # class MIoU(MeanIoU):
    #     def __init__(self, num_classes, name=None, dtype=None):
    #         super(MIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
    
    #     def update_state(self, y_true, y_pred, sample_weight=None):
    #         result = super(MIoU, self).update_state(
    #             y_true=y_true, 
    #             y_pred=tf.math.argmax(input=y_pred, axis=-1, output_type=tf.dtypes.int64), 
    #             sample_weight=sample_weight)
    #         return result

    

    cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels),
                    num_classes=n_classes, output_stride=output_stride, 
                    pooling=pooling, residual_shortcut=residual_shortcut, 
                    backbonetype=backbonetype)
    cmsnet.summary()
    

    miou = MIoU(num_classes=n_classes)
    #miou = MeanIoU(num_classes=n_classes)
    # cmsnet.compile(optimizer, loss='sparse_categorical_crossentropy', 
    #                sample_weight_mode="temporal", metrics=['accuracy', miou])
    

    

    
    #fold_name = 's16_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_aspp_20190815-151551'
    fold_name = (backbonetype+'s'+str(output_stride)+'_pooling_' + pooling
         +('_residual_shortcut' if residual_shortcut else '')+'_ep'+str(epochs)
         +('_crop_'+str(height_crop)+'x'+str(width_crop) if crop_enable else '_')
         +('from' if crop_enable else '')+str(height_image)+'x'+str(width_image)
         +'_'+('wda_' if data_augmentation else 'nda_')
         +('wcic_' if class_imbalance_correction else 'ncic_')
         +('wft_' if fine_tune_last else 'nft_')+dataset_name+'_b'
         +str(batch_size)+('_n' if multigpu_enable else '_1')+'gpu')
    # Define the Keras TensorBoard callback.
    continue_traning = True
    if continue_traning:
        logdir=build_path+"logs/fit/" + fold_name #Continue
        weights_path = glob.glob(logdir+'*/weights.last*')[0]
        logdir = weights_path[:weights_path.find('/weights.')]
    
        print('Continuing train from '+ weights_path)
        cmsnet.load_weights(weights_path)
    else:
        logdir=build_path+"logs/fit/" + fold_name+'_'+datetime.now().strftime("%Y%m%d-%H%M%S")
    
    
    
    
    
    # result = cmsnet.evaluate(ds_test, use_multiprocessing=True, steps=test_steps)
    i = 0
    for image, label in ds_test:
        print(i)
        i +=1;
        
        # mask_cityscape = model.run_np(image.numpy()[0])
        mask_cityscape = cmsnet.predict(image)
        
        from matplotlib import pyplot as plt
        plt.imshow(image.numpy()[0])
        plt.show()
        plt.imshow(label.numpy()[0][...,0])
        plt.show()
        for i in range(10):
            plt.imshow(mask_cityscape[0][...,i])
            # plt.hist(mask_cityscape[0][...,i])
            plt.show()
        y_pred=tf.math.argmax(input=mask_cityscape[0], axis=-1, output_type=tf.dtypes.int64)
        plt.imshow(y_pred.numpy())
        plt.show()
        # mask_cityscape = np.array(Image.fromarray(np.uint8(mask_cityscape)).resize((image.numpy().shape[2],image.numpy().shape[1])))
        # #Convert infered result in a offroad mask
        # #Zero is the ignore label. Every thing is considereted ignore
        # mask_offroad = np.zeros(mask_cityscape.shape)
        
        # mask_offroad[mask_cityscape==0]=1  #road
        # mask_offroad[mask_cityscape==11]=3 #person #person
        # mask_offroad[mask_cityscape==12]=3 #rider  #person
        # mask_offroad[mask_cityscape==13]=2 #car
        # mask_offroad[mask_cityscape==14]=4 #truck
        # mask_offroad[mask_cityscape==15]=5 #bus
        # mask_offroad[mask_cityscape==17]=7 #motocycle
        # mask_offroad[mask_cityscape==18]=9 #bicycle
        
        # miou.update_state(label[0,...,0], mask_offroad)
        miou.update_state(label, mask_cityscape)
    
    weights = miou.get_weights()
    result = {'name':fold_name}
    result['classes'] = classes
    result['confusion_matrix'] = weights[0].tolist()
    result['count_params'] = cmsnet.count_params()
    return result
    
def eval_one(pooling = 'global',
    backbonetype ='mobilenetv2',
    output_stride = 8,
    residual_shortcut = False, 
    height_image = 448, 
    width_image = 448, 
    channels = 3, 
    crop_enable = False,
    height_crop = 448, 
    width_crop = 448, 
    
    debug_en = False, 
    dataset_name = 'freiburg_forest', 
    class_imbalance_correction = False, 
    data_augmentation = True, 
     
    multigpu_enable = True,
    
    batch_size = 32, 
    epochs = 300, 
    initial_epoch = -1, 
    continue_traning = False, 
    fine_tune_last = False, 
    
    base_learning_rate = 0.007, 
    learning_power = 0.98, 
    decay_steps = 1,
    learning_rate_decay_step = 300, 
    
    decay = 5**(-4),
    image = [],
    ):
    
    
    fold_name = (backbonetype+'s'+str(output_stride)+'_pooling_' + pooling
             +('_residual_shortcut' if residual_shortcut else '')+'_ep'+str(epochs)
             +('_crop_'+str(height_crop)+'x'+str(width_crop) if crop_enable else '_')
             +('from' if crop_enable else '')+str(height_image)+'x'+str(width_image)
             +'_'+('wda_' if data_augmentation else 'nda_')
             +('wcic_' if class_imbalance_correction else 'ncic_')
             +('wft_' if fine_tune_last else 'nft_')+dataset_name+'_b'
             +str(batch_size)+('_n' if multigpu_enable else '_1')+'gpu')
    
    
    
    #Override params for eval
    # multigpu_enable = False
    # batch_size      = 2


    ## Check the number of labels
    if dataset_name == 'cityscape':
        classes = cityscape.classes
    elif dataset_name == 'citysmall':
        classes = citysmall.classes
    elif dataset_name == 'off_road_small':
        classes = off_road_small.classes
    elif dataset_name == 'freiburg_forest':
        classes = freiburg_forest.classes
    
    n_classes = len(classes)
    

    if crop_enable:
        assert(height_crop == width_crop, "When crop is enable height_crop should be equals to width_crop")
        assert(height_crop <= height_image, "When crop is enable height_crop should be less than or equals to height_image")
        assert(width_crop <= width_image, "When crop is enable height_crop should be less than or equals to width_image")
    else:
        height_crop = height_image
        width_crop = width_image
    
    
    # Construct a tf.data.Dataset
    info = tfds.builder(dataset_name).info
    print(info)
    [ds_test] = tfds.load(name=dataset_name, split=["test"], as_supervised=True)
    # Add normalize
    def _normalize_img(image):
        image = tf.cast(image, tf.float32)/127.5 - 1
        if crop_enable:
            y1 = tf.random.uniform(shape=[], minval = 0., maxval = (height_image-height_crop)/height_image, dtype=tf.float32)
            x1 = tf.random.uniform(shape=[], minval = 0., maxval = (width_image-width_crop)/width_image, dtype=tf.float32)
    
            y2 = y1 + (height_crop/height_image)
            x2 = x1 + (width_crop/width_image)
    
            boxes = [[y1, x1, y2, x2]]
            image = tf.image.crop_and_resize([image], boxes, box_indices = [0], crop_size = (height_crop, width_crop), method=tf.image.ResizeMethod.BILINEAR)[0]
        else:
            image = tf.image.resize(image, (height_image,width_image), method=tf.image.ResizeMethod.BILINEAR)
        return (image)
    

    

    cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels),
                    num_classes=n_classes, output_stride=output_stride, pooling=pooling,
                    residual_shortcut=residual_shortcut, backbonetype=backbonetype)
    cmsnet.summary()

    logdir=build_path+"logs/fit/" + fold_name #Continue
    weights_path = glob.glob(logdir+'*/weights.last*')[0]
    logdir = weights_path[:weights_path.find('/weights.')]

    print('Continuing train from '+ weights_path)
    cmsnet.load_weights(weights_path)
    
    image = _normalize_img([image])
    result = cmsnet.predict(image)
    
    return result
# def eval_one(weights_path, image, height_image, width_image, channels, n_classes, 
#              output_stride, pooling, residual_shortcut, backbonetype):
#     cmsnet = CMSNet(dl_input_shape=(None, height_image, width_image, channels),
#                     num_classes=n_classes, output_stride=output_stride, 
#                     pooling=pooling, residual_shortcut=residual_shortcut, 
#                     backbonetype=backbonetype)
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


    
