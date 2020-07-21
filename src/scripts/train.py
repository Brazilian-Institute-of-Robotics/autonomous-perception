# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_datasets as tfds
from datetime import datetime
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Nadam, Adamax
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard

import math
from PIL import Image
import numpy as np
import glob

import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')
from models.cmsnet import CMSNet, LRTensorBoard, polynomial_decay, exponential_decay
from datasets import citysmall
from datasets import cityscape
from datasets import off_road_small
from datasets import freiburg_forest
from utils import augmentation as aug
# from tensorflow.python import debug as tf_debug


# hook = tf_debug.TensorBoardDebugHook("nelson-avell:6007")

build_path = os.path.dirname(__file__) + '/../../build/'





gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = 'false'

def train(pooling = 'global',
    backbonetype='mobilenetv2',
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
    validation_enable=False
    ):



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
    ignore_label = classes[-1]['name']=='ignore'
    ignore_label = False
    
    #n_classes = 12

    if crop_enable:
        assert(height_crop == width_crop, 
               "When crop is enable height_crop should be equals to width_crop")
        assert(height_crop <= height_image, 
               "When crop is enable height_crop should be less than or equals to height_image")
        assert(width_crop <= width_image, 
               "When crop is enable height_crop should be less than or equals to width_image")
    else:
        height_crop = height_image
        width_crop = width_image
    
    # Construct a tf.data.Dataset
    info = tfds.builder(dataset_name).info
    print(info)
    # if validation_enable : #7%
    #     ds_train, ds_val, ds_test = tfds.load(name=dataset_name, split=['train[:-10%]', 'train[-10%:]', 'test'], as_supervised=True)
    # else:
    #     ds_train, ds_test = tfds.load(name=dataset_name, split=["train", 'test'], as_supervised=True)
        
    train, ds_val, ds_test = tfds.load(name=dataset_name, 
                                          split=[tfds.Split.TRAIN, 
                                                 tfds.Split.VALIDATION, 
                                                 tfds.Split.TEST], 
                                          as_supervised=True)

    height_ds = info.features['image'].shape[-3]
    width_ds = info.features['image'].shape[-2]
    # Add normalize
    def _normalize_img(image, label):
        image = tf.cast(image, tf.float32)/127.5 - 1
        if crop_enable:
            y1 = tf.random.uniform(shape=[], minval=0., maxval=(height_image-height_crop)/height_image, dtype=tf.float32)
            x1 = tf.random.uniform(shape=[], minval=0., maxval=(width_image-width_crop)/width_image, dtype=tf.float32)
    
            y2 = y1 + (height_crop/height_image)
            x2 = x1 + (width_crop/width_image)
    
            boxes = [[y1, x1, y2, x2]]
            image = tf.image.crop_and_resize([image], boxes, box_indices=[0], crop_size=(height_crop, width_crop), method=tf.image.ResizeMethod.BILINEAR)[0]
            label = tf.cast(tf.image.crop_and_resize([label], boxes, box_indices=[0], crop_size=(height_crop, width_crop), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0],dtype=tf.uint8)
        else:
            image = tf.image.resize(image, (height_image,width_image), method=tf.image.ResizeMethod.BILINEAR)
            label = tf.image.resize(label, (height_image,width_image), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return (image, label)
    
    # ds_train = ds_train.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # if validation_enable :
    #     ds_val = ds_val.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if validation_enable :
        ds_train = train
        train_size = info.splits['train'].num_examples
    else:
        ds_train = train.concatenate(ds_val)
        train_size = info.splits['train'].num_examples + info.splits['validation'].num_examples
    # ds_train = ds_train.take(100)
    ds_train = ds_train.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    

    ########################################################debug
    # ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # ds_test = ds_test.shuffle(124).batch(batch_size).prefetch(10)
    ########################################################debug
    
    if data_augmentation:
        # Add augmentations
        augmentations = [aug.flip, aug.color, aug.zoom, aug.rotate]
        
        for f in augmentations:
            ds_train = ds_train.map(lambda x, y: tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: f(x, y), lambda: (x, y)),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.map(lambda x, y: (tf.clip_by_value(x, -1, 1), y),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if debug_en:
            aug.plot_images(ds_train, n_images=8, samples_per_image=10, classes = classes)
    
    
    # Build your input pipeline
    ds_train = ds_train.repeat().shuffle(124).batch(batch_size).prefetch(10)
    ds_val = ds_val.shuffle(124).batch(batch_size).prefetch(10)
    # ds_train = ds_train.shuffle(124).batch(batch_size).prefetch(10)
    # if validation_enable :
    #     ds_val = ds_val.shuffle(124).batch(batch_size).prefetch(10)
    # else:
    #     ds_val = None
    
    # validation_steps=int(round(info.splits['test'].num_examples/batch_size))
    steps_per_epoch=int(round(train_size/batch_size))
    step_size = steps_per_epoch
    
    class MIoU(MeanIoU):
      def __init__(self, num_classes, name=None, dtype=None):
        super(MIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)
    
      def update_state(self, y_true, y_pred, sample_weight=None):
        return super(MIoU, self).update_state(
                y_true=y_true, 
                y_pred=tf.math.argmax(input=y_pred, axis=-1, output_type=tf.dtypes.int64), 
                sample_weight=sample_weight)
    
    # class_imbalance_correction
    
    fold_name = (backbonetype+'s'+str(output_stride)+'_pooling_' + pooling
                 +('_residual_shortcut' if residual_shortcut else '')+'_ep'+str(epochs)
                 +('_crop_'+str(height_crop)+'x'+str(width_crop) if crop_enable else '_')
                 +('from' if crop_enable else '')+str(height_image)+'x'+str(width_image)
                 +'_'+('wda_' if data_augmentation else 'nda_')
                 +('wcic_' if class_imbalance_correction else 'ncic_')
                 +('wft_' if fine_tune_last else 'nft_')+dataset_name+'_b'
                 +str(batch_size)+('_n' if multigpu_enable else '_1')+'gpu')
    
    # multigpu_enable = False
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
            #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
            #optimizer = Adadelta(lr=0.008, rho=0.95, epsilon=None, decay=0.0)
            # optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
            miou = MIoU(num_classes=n_classes)
    
            cmsnet.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                          metrics=['accuracy', miou]
                          )
    else:
        cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels),
                        num_classes=n_classes, output_stride=output_stride, pooling=pooling,
                        residual_shortcut=residual_shortcut, backbonetype=backbonetype)
        cmsnet.summary()
        
        optimizer = SGD(momentum=0.9, nesterov=True)
        #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
        #optimizer = Adadelta(lr=0.008, rho=0.95, epsilon=None, decay=0.0)
        # optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
        miou = MIoU(num_classes=n_classes)
        cmsnet.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy', miou])
    
    

    
    #fold_name = 's16_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_aspp_20190815-151551'
    
    # Define the Keras TensorBoard callback.
    if continue_traning:
        logdir=build_path+"logs/fit/" + fold_name #Continue
        if initial_epoch == -1: #get last checkpoint epoch
            names = glob.glob(logdir+'*/weights.*')
            names.sort()
            initial_epoch = int(names[-1].split('.')[-4].split('-')[0])

        #weights_path = glob.glob(logdir+'*/weights.*'+str(initial_epoch)+'-*')[0]
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
        initial_epoch = 0
    
    
    
    
    # if validation_enable : #7%
    #     ckp = ModelCheckpoint(logdir+"/weights.{epoch:03d}-{val_loss:.2f}-{val_m_io_u:.2f}.hdf5",
    #                       monitor='val_m_io_u', mode='max',  verbose=0, save_best_only=True,
    #                       save_weights_only=True, period=1)
    # else:
    # ckp = ModelCheckpoint(logdir+"/weights.{epoch:03d}-{loss:.2f}-{m_io_u:.2f}.hdf5",
    #                   monitor='val_m_io_u', mode='max',  verbose=0, save_best_only=True,
    #                   save_weights_only=True, period=1)
    # ckp = ModelCheckpoint(logdir+"/weights.{epoch:03d}.hdf5",
    #                       mode='max',  verbose=0, save_best_only=True,
    #                       save_weights_only=True, period=1)
    ckp = ModelCheckpoint(logdir+"/weights.{epoch:03d}-{val_loss:.2f}-{val_m_io_u:.2f}.hdf5",
                          monitor='val_m_io_u', mode='max',  verbose=0, save_best_only=True,
                          save_weights_only=True, period=1)
    ckp_last = ModelCheckpoint(logdir+"/weights.last.hdf5", verbose=0, save_best_only=False,
                          save_weights_only=True, period=1)
    
    tensorboard_callback = LRTensorBoard(log_dir=logdir, histogram_freq=0, 
                                         write_graph=True, write_images=True,
                                         update_freq='epoch', profile_batch=2, 
                                         embeddings_freq=0)
    
    #if aplay balance
    # from sklearn.utils import class_weight
    # class_weight = class_weight.compute_class_weight('balanced'
    #                                                ,np.unique(Y_train)
    #                                                ,Y_train)
    
    # class_weight = {cls:1 for cls in range(n_classes)}
    # #ignore the last label
    # if ignore_label:
    #     class_weight[n_classes-1] = 0
    
    if class_imbalance_correction:
        class_weight = np.ones(n_classes) #TODO: insert inbalance equalizer
    elif ignore_label:
        class_weight = np.ones(n_classes)
    else:
        class_weight = None
    #ignore the last label
    if ignore_label:
        class_weight[-1] = 0
    
    class_weight=None
    
    

    
    
    if fine_tune_last:
        base_learning_rate = 0.0001
        cmsnet.setFineTuning(lavel='fromSPPool')
        # learning = lambda epoch: polynomial_decay(epoch, initial_lrate = base_learning_rate,
        #     learning_rate_decay_step=learning_rate_decay_step, learning_power=learning_power, 
        #     end_learning_rate=0.0001, cycle=False)
        learning = lambda epoch: exponential_decay(learning_rate=base_learning_rate, global_step=epoch, decay_steps=decay_steps, decay_rate=learning_power)
    
        lrate = LearningRateScheduler(learning)
        hist1 = cmsnet.fit(ds_train, validation_data=ds_val, epochs=epochs,
                      callbacks=[tensorboard_callback, lrate, ckp],
                      initial_epoch=initial_epoch,  class_weight=class_weight)
    else:
    
        cmsnet.setFineTuning(lavel='fromAll')
        # learning = lambda epoch: polynomial_decay(epoch, initial_lrate = base_learning_rate,
        #         learning_rate_decay_step=learning_rate_decay_step, learning_power=learning_power, 
        #         end_learning_rate=0, cycle=False)
        learning = lambda epoch: exponential_decay(learning_rate=base_learning_rate, global_step=epoch, decay_steps=decay_steps, decay_rate=learning_power)
    
        lrate = LearningRateScheduler(learning)
        cmsnet.fit(ds_train, validation_data = ds_val, 
                   epochs = round(epochs*steps_per_epoch/step_size), 
                   steps_per_epoch = step_size,
                   callbacks = [tensorboard_callback, lrate, ckp, ckp_last],
                   initial_epoch = initial_epoch,  
                   class_weight = class_weight,
                   use_multiprocessing = True)
        
        ########################################################debug
        # result = cmsnet.evaluate(ds_train, use_multiprocessing=True)
        # import json
        # with open(build_path+'reslut0.txt', 'w') as file:
        #     file.write(json.dumps(result))
            
        # classes_ids = [1, 2, 3, 4, 0]
        # from evaluation.trained import print_result2
        # print(result['name'])
        # print("Params: "+str(result['count_params']))
        # print_result2(result['classes'], np.array(result['confusion_matrix']), classes_ids)
        ########################################################debug
    
    # ########################################################debug
    # result = cmsnet.evaluate(ds_test, use_multiprocessing=True)
    # import json
    # with open(build_path+'reslut0.txt', 'w') as file:
    #     file.write(json.dumps(result))
    
    # classes_ids = [1, 2, 3, 4, 0]
    # from evaluation.trained import print_result2
    # print(result['name'])
    # print("Params: "+str(result['count_params']))
    # print_result2(result['classes'], np.array(result['confusion_matrix']), classes_ids)
    # ########################################################debug
    
    cmsnet.save_weights(logdir+"/weights.300-n.nn-n.nn.hdf5")
    # cmsnet.save_weights(logdir+"/weights.300.hdf5")



# decay=5**(-4)
# base_learning_rate = 0.007
# decay_steps = 1
# multigpu_enable=True
# class_imbalance_correction = False
# batch_size=32
# epochs=300
# learning_rate_decay_step =  300 #180/400
# learning_power = 0.98
# debug_en = False
# dataset_name = "freiburg_forest" #"cityscape"#"citysmall"off_road_small 'freiburg_forest'
# channels = 3
# # height_crop = 483
# # width_crop  = 769
# crop_enable=False
# height_image = 448
# width_image  = 448
# height_crop = 448
# width_crop  = 448
# # height_image = 500
# # width_image  = 900
# # height_crop = 500
# # width_crop  = 700





# ['decay' : 5**(-4), 'base_learning_rate' : 0.007, 'decay_steps' : 1, 'multigpu_enable' : True, 'class_imbalance_correction' : False, 'batch_size' : 32, 'epochs' : 300, 'learning_rate_decay_step' : 300, 'learning_power' : 0.98, 'debug_en' : False, 'dataset_name' : 'freiburg_forest', 'channels' : 3, 'crop_enable' : False, 'height_image' : 448, 'width_image' : 448, 'height_crop' : 448, 'width_crop' : 448]






# train_cases =[
                # {'pooling' : 'global',
                #   'backbonetype' : 'resnet101',
                # 'output_stride' : 8,
                # 'residual_shortcut' : False, 
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,
                
                # 'batch_size' : 4,
                # 'epochs' : 300,
                # 'initial_epoch' : 150, 
                # 'continue_traning' : True,
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'spp',
                #   'backbonetype' : 'resnet101',
                # 'output_stride' : 8,
                # 'residual_shortcut' : False, 
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,

                # 'batch_size' : 4,
                # 'epochs' : 300,
                # 'initial_epoch' : 150, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'aspp',
                #   'backbonetype' : 'resnet101',
                # 'output_stride' : 8,
                # 'residual_shortcut' : False, 
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,
                
                # 'batch_size' : 4,
                # 'epochs' : 300,
                # 'initial_epoch' : 150, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'global',
                # 'backbonetype' : 'resnet101',
                # 'output_stride' : 16,
                # 'residual_shortcut' : False, 
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,
                
                # 'batch_size' : 4,
                # 'epochs' : 300,
                # 'initial_epoch' : 150, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'spp',
                #   'backbonetype' : 'resnet101',
                # 'output_stride' : 16,
                # 'residual_shortcut' : False, 
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,

                # 'batch_size' : 4,
                # 'epochs' : 300,
                # 'initial_epoch' : 150, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'aspp',
                #   'backbonetype' : 'resnet101',
                # 'output_stride' : 16,
                # 'residual_shortcut' : False, 
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,
                
                # 'batch_size' : 4,
                # 'epochs' : 300,
                # 'initial_epoch' : 150, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'global',
                # 'backbonetype' : 'resnet101',
                # 'output_stride' : 16,
                # 'residual_shortcut' : True,
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,
                
                # 'batch_size' : 4,
                # 'epochs' : 300,
                # 'initial_epoch' : 150, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'spp',
                #   'backbonetype' : 'resnet101',
                # 'output_stride' : 16,
                # 'residual_shortcut' : True,
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,

                # 'batch_size' : 4,
                # 'epochs' : 300,
                # 'initial_epoch' : 150, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'aspp',
                #   'backbonetype' : 'resnet101',
                # 'output_stride' : 16,
                # 'residual_shortcut' : True,
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False,
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,
                
                # 'batch_size' : 4,
                # 'epochs' : 300,
                # 'initial_epoch' : 150, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                #},
                
                
                
                
#                 ###############################################

                
                
# #                 {'pooling' : 'global',
# #                 'backbonetype' : 'mobilenetv2',
# #                 'output_stride' : 16,
# #                 'residual_shortcut' : False, 
# #                 'height_image' : 483,
# #                 'width_image' : 769,
# #                 'channels' : 3, 
# #                 'crop_enable' : False, 
# #                 'height_crop' : 483,
# #                 'width_crop' : 769,

# #                 'dataset_name' : 'off_road_small',
# #                 'class_imbalance_correction' : False, 
# #                 'data_augmentation' : True, 
                 
# #                 'multigpu_enable' : True,
# #                 'debug_en' : False,
                
# #                 'batch_size' : 2,
# #                 'epochs' : 200,
# #                 'initial_epoch' : -1, 
# #                 'continue_traning' : False, 
# #                 'fine_tune_last' : False, 
                
# #                 'base_learning_rate' : 0.007, 
# #                 'learning_power' : 0.98, 
# #                 'decay_steps' : 1,
# #                 'learning_rate_decay_step' : 4,
# #                 'decay' : 5**(-4)
# #                 },

# #fazer novamente
#                 {'pooling' : 'spp',
#                   'backbonetype' : 'mobilenetv2',
#                 'output_stride' : 16,
#                 'residual_shortcut' : False, 
#                 'height_image' : 483,
#                 'width_image' : 769,
#                 'channels' : 3, 
#                 'crop_enable' : False, 
#                 'height_crop' : 483,
#                 'width_crop' : 769,

#                 'dataset_name' : 'off_road_small',
#                 'class_imbalance_correction' : False, 
#                 'data_augmentation' : True, 
                 
#                 'multigpu_enable' : False,
#                 'debug_en' : False,

#                 'batch_size' : 4,
#                 'epochs' : 200,
#                 'initial_epoch' : -1, 
#                 'continue_traning' : False, 
#                 'fine_tune_last' : False, 
                
#                 'base_learning_rate' : 0.007, 
#                 'learning_power' : 0.98, 
#                 'decay_steps' : 1,
#                 'learning_rate_decay_step' : 4,
#                 'decay' : 5**(-4)
#                 },
                
# #                 {'pooling' : 'aspp',
# #                   'backbonetype' : 'mobilenetv2',
# #                 'output_stride' : 16,
# #                 'residual_shortcut' : False, 
# #                 'height_image' : 483,
# #                 'width_image' : 769,
# #                 'channels' : 3, 
# #                 'crop_enable' : False, 
# #                 'height_crop' : 483,
# #                 'width_crop' : 769,

# #                 'dataset_name' : 'off_road_small',
# #                 'class_imbalance_correction' : False, 
# #                 'data_augmentation' : True, 
                 
# #                 'multigpu_enable' : False,
# #                 'debug_en' : False,
                
# #                 'batch_size' : 2,
# #                 'epochs' : 200,
# #                 'initial_epoch' : -1, 
# #                 'continue_traning' : False, 
# #                 'fine_tune_last' : False, 
                
# #                 'base_learning_rate' : 0.007, 
# #                 'learning_power' : 0.98, 
# #                 'decay_steps' : 1,
# #                 'learning_rate_decay_step' : 4,
# #                 'decay' : 5**(-4)
# #                 },
# #                 {'pooling' : 'global',
# #                 'backbonetype' : 'mobilenetv2',
# #                 'output_stride' : 16,
# #                 'residual_shortcut' : True,
# #                 'height_image' : 483,
# #                 'width_image' : 769,
# #                 'channels' : 3, 
# #                 'crop_enable' : False, 
# #                 'height_crop' : 483,
# #                 'width_crop' : 769,

# #                 'dataset_name' : 'off_road_small',
# #                 'class_imbalance_correction' : False, 
# #                 'data_augmentation' : True, 
                 
# #                 'multigpu_enable' : False,
# #                 'debug_en' : False,
                
# #                 'batch_size' : 4,
# #                 'epochs' : 200,
# #                 'initial_epoch' : -1, 
# #                 'continue_traning' : False, 
# #                 'fine_tune_last' : False, 
                
# #                 'base_learning_rate' : 0.007, 
# #                 'learning_power' : 0.98, 
# #                 'decay_steps' : 1,
# #                 'learning_rate_decay_step' : 4,
# #                 'decay' : 5**(-4)
# #                 },
# #                 {'pooling' : 'spp',
# #                   'backbonetype' : 'mobilenetv2',
# #                 'output_stride' : 16,
# #                 'residual_shortcut' : True,
# #                 'height_image' : 483,
# #                 'width_image' : 769,
# #                 'channels' : 3, 
# #                 'crop_enable' : False, 
# #                 'height_crop' : 483,
# #                 'width_crop' : 769,

# #                 'dataset_name' : 'off_road_small',
# #                 'class_imbalance_correction' : False, 
# #                 'data_augmentation' : True, 
                 
# #                 'multigpu_enable' : False,
# #                 'debug_en' : False,

# #                 'batch_size' : 4,
# #                 'epochs' : 200,
# #                 'initial_epoch' : -1, 
# #                 'continue_traning' : False, 
# #                 'fine_tune_last' : False, 
                
# #                 'base_learning_rate' : 0.007, 
# #                 'learning_power' : 0.98, 
# #                 'decay_steps' : 1,
# #                 'learning_rate_decay_step' : 4,
# #                 'decay' : 5**(-4)
# #                 },
#                 # {'pooling' : 'aspp',
#                 #   'backbonetype' : 'mobilenetv2',
#                 # 'output_stride' : 16,
#                 # 'residual_shortcut' : True,
#                 # 'height_image' : 483,
#                 # 'width_image' : 769,
#                 # 'channels' : 3, 
#                 # 'crop_enable' : False, 
#                 # 'height_crop' : 483,
#                 # 'width_crop' : 769,

#                 # 'dataset_name' : 'off_road_small',
#                 # 'class_imbalance_correction' : False,
#                 # 'data_augmentation' : True, 
                 
#                 # 'multigpu_enable' : False,
#                 # 'debug_en' : False,
                
#                 # 'batch_size' : 2,
#                 # 'epochs' : 200,
#                 # 'initial_epoch' : -1, 
#                 # 'continue_traning' : False, 
#                 # 'fine_tune_last' : False, 
                
#                 # 'base_learning_rate' : 0.007, 
#                 # 'learning_power' : 0.98, 
#                 # 'decay_steps' : 1,
#                 # 'learning_rate_decay_step' : 4,
#                 # 'decay' : 5**(-4)
#                 # },
#                 {'pooling' : 'global',
#                 'backbonetype' : 'mobilenetv2',
#                 'output_stride' : 8,
#                 'residual_shortcut' : False, 
#                 'height_image' : 483,
#                 'width_image' : 769,
#                 'channels' : 3, 
#                 'crop_enable' : False, 
#                 'height_crop' : 483,
#                 'width_crop' : 769,

#                 'dataset_name' : 'off_road_small',
#                 'class_imbalance_correction' : False, 
#                 'data_augmentation' : True, 
                 
#                 'multigpu_enable' : False,
#                 'debug_en' : False,
                
#                 'batch_size' : 2,
#                 'epochs' : 200,
#                 'initial_epoch' : -1, 
#                 'continue_traning' : False,
#                 'fine_tune_last' : False, 
                
#                 'base_learning_rate' : 0.007, 
#                 'learning_power' : 0.98, 
#                 'decay_steps' : 1,
#                 'learning_rate_decay_step' : 4,
#                 'decay' : 5**(-4)
#                 },
#                 {'pooling' : 'spp',
#                  'backbonetype' : 'mobilenetv2',
#                 'output_stride' : 8,
#                 'residual_shortcut' : False, 
#                 'height_image' : 483,
#                 'width_image' : 769,
#                 'channels' : 3, 
#                 'crop_enable' : False, 
#                 'height_crop' : 483,
#                 'width_crop' : 769,

#                 'dataset_name' : 'off_road_small',
#                 'class_imbalance_correction' : False, 
#                 'data_augmentation' : True, 
                 
#                 'multigpu_enable' : False,
#                 'debug_en' : False,

#                 'batch_size' : 2,
#                 'epochs' : 200,
#                 'initial_epoch' : -1, 
#                 'continue_traning' : False, 
#                 'fine_tune_last' : False, 
                
#                 'base_learning_rate' : 0.007, 
#                 'learning_power' : 0.98, 
#                 'decay_steps' : 1,
#                 'learning_rate_decay_step' : 4,
#                 'decay' : 5**(-4)
#                 },
#                 ##################################################
#                 # {'pooling' : 'aspp',
#                 #   'backbonetype' : 'mobilenetv2',
#                 # 'output_stride' : 8,
#                 # 'residual_shortcut' : False, 
#                 # 'height_image' : 483,
#                 # 'width_image' : 769,
#                 # 'channels' : 3, 
#                 # 'crop_enable' : False, 
#                 # 'height_crop' : 483,
#                 # 'width_crop' : 769,

#                 # 'dataset_name' : 'off_road_small',
#                 # 'class_imbalance_correction' : False, 
#                 # 'data_augmentation' : True, 
                 
#                 # 'multigpu_enable' : True,
#                 # 'debug_en' : False,
                
#                 # 'batch_size' : 2,
#                 # 'epochs' : 200,
#                 # 'initial_epoch' : 50,
#                 # 'continue_traning' : True,
#                 # 'fine_tune_last' : False, 
                
#                 # 'base_learning_rate' : 0.007, 
#                 # 'learning_power' : 0.98, 
#                 # 'decay_steps' : 1,
#                 # 'learning_rate_decay_step' : 4,
#                 # 'decay' : 5**(-4)
#                 # },
#                   ]

train_cases =[
                # {'pooling' : 'global',
                # 'backbonetype' : 'mobilenetv2',
                # 'output_stride' : 16,
                # 'residual_shortcut' : False, 
                # 'height_image' : 483,
                # 'width_image' : 769,
                # 'channels' : 3, 
                # 'crop_enable' : False, 
                # 'height_crop' : 483,
                # 'width_crop' : 769,

                # 'dataset_name' : 'off_road_small',
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                 
                # 'multigpu_enable' : False,
                # 'debug_en' : False,
                
                # 'batch_size' : 4,
                # 'epochs' : 200,
                # 'initial_epoch' : -1, 
                # 'continue_traning' : False, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                {'pooling' : 'global',
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
                 
                'multigpu_enable' : False,
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
                'decay' : 5**(-4)
                },
                {'pooling' : 'spp',
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
                 
                'multigpu_enable' : False,
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
                'decay' : 5**(-4)
                },

                ]



for t, i in zip(train_cases, range(len(train_cases)) ):
    train(pooling = t['pooling'], backbonetype=t['backbonetype'], output_stride = t['output_stride'], residual_shortcut = t['residual_shortcut'],
          height_image = t['height_image'], width_image = t['width_image'], channels = t['channels'],
          crop_enable = t['crop_enable'], height_crop = t['height_crop'], width_crop = t['width_crop'],
          debug_en = t['debug_en'], dataset_name = t['dataset_name'], class_imbalance_correction = t['class_imbalance_correction'],
          data_augmentation = t['data_augmentation'],multigpu_enable = t['multigpu_enable'],
          batch_size = t['batch_size'], epochs = t['epochs'], initial_epoch = t['initial_epoch'],
          continue_traning = t['continue_traning'], fine_tune_last = t['fine_tune_last'],
          base_learning_rate = t['base_learning_rate'], learning_power = t['learning_power'],
          decay_steps = t['decay_steps'], learning_rate_decay_step = t['learning_rate_decay_step'],
          decay = t['decay'])
    tf.keras.backend.clear_session()
