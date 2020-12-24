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
# from imgaug import augmenters as iaa
# import imgaug as ia
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
import time

build_path = os.path.dirname(__file__) + '/../../build/'




decay=5**(-4)
base_learning_rate = 0.007
decay_steps = 1
multigpu_enable=False
class_imbalance_correction = False
batch_size=1
epochs=300
learning_rate_decay_step =  300 #180/400
learning_power = 0.98
debug_en = False
dataset_name = "off_road_small" #"cityscape"#"citysmall"off_road_small freiburg_forest
channels = 3
crop_enable=False
height_image = 483
width_image  = 769
height_crop = 483
width_crop  = 769
# height_image = 500
# width_image  = 900
# height_crop = 500
# width_crop  = 700

pooling = 'aspp'
output_stride = 16
residual_shortcut=False


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

#n_classes = 12

initial_epoch = -1
data_augmentation = True
continue_traning = False
fine_tune_last = False




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

ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# if data_augmentation:
#     # Add augmentations
#     augmentations = [aug.flip, aug.color, aug.zoom]
    
#     for f in augmentations:
#         ds_train = ds_train.map(lambda x, y: tf.cond(tf.random.uniform([], 0, 1) > 0.75, lambda: f(x, y), lambda: (x, y)),
#                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     ds_train = ds_train.map(lambda x, y: (tf.clip_by_value(x, -1, 1), y),  num_parallel_calls=tf.data.experimental.AUTOTUNE)
#     aug.plot_images(ds_train, n_images=8, samples_per_image=10, classes = classes)


# Build your input pipeline
ds_test = ds_test.shuffle(124).batch(batch_size).prefetch(10)

test_steps=int(round(info.splits['test'].num_examples/batch_size))

class MIoU(MeanIoU):
  def __init__(self, num_classes, name=None, dtype=None):
    super(MIoU, self).__init__(num_classes=num_classes, name=name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    result = super(MIoU, self).update_state(
            y_true=y_true, 
            y_pred=tf.math.argmax(input=y_pred, axis=-1, output_type=tf.dtypes.int64), 
            sample_weight=sample_weight)

    return result



if multigpu_enable:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels), num_classes=n_classes, output_stride=output_stride, pooling=pooling, residual_shortcut=residual_shortcut)
        cmsnet.summary()
        #cmsnet.mySummary()

        # optimizer = SGD(momentum=0.9, nesterov=True)
        #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
        #optimizer = Adadelta(lr=0.008, rho=0.95, epsilon=None, decay=0.0)
        optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
        #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        miou = MIoU(num_classes=n_classes)

        cmsnet.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                      metrics=['accuracy', miou], sample_weight_mode="temporal"
                      )
else:
    cmsnet = CMSNet(dl_input_shape=(None, height_crop, width_crop, channels), num_classes=n_classes, output_stride=output_stride, pooling=pooling, residual_shortcut=residual_shortcut)
    cmsnet.summary()
    
    # optimizer = SGD(momentum=0.9, nesterov=True)
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
    #optimizer = Adadelta(lr=0.008, rho=0.95, epsilon=None, decay=0.0)
    optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    #optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    miou = MIoU(num_classes=n_classes)
    cmsnet.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', miou], sample_weight_mode="temporal")




fold_name = ('s'+str(output_stride)+'_'+('wda_' if data_augmentation else 'nda_')+('wcic_' if class_imbalance_correction else 'ncic_')
             +('wft_' if fine_tune_last else 'nft_')+dataset_name+'_b'+str(batch_size)
             +('_n' if multigpu_enable else '_1')+'gpu_ep'+str(epochs)+('_crop_'+str(height_crop)+'x'+str(width_crop) if crop_enable else '_')
             +('from' if crop_enable else '')+str(height_image)+'x'+str(width_image)+'_pooling_' + pooling + ('residual_shortcut' if residual_shortcut else ''))



fold_name = 's8_wda_ncic_nft_off_road_small_b8_ngpu_ep500_crop_483x769from483x769_20190807-133552'
fold_name = 's8_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_20190808-085829'
# fold_name = 's8_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_spp_20190809-143224'
# fold_name = 's8_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_aspp_20190812-165338'
#fold_name = 's16_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_asppresidual_shortcut_20190814-125731'
# fold_name = 's16_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_aspp_20190815-151551'
fold_name = 's8_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_spp_20190809-143224'
# Define the Keras TensorBoard callback.
fold_name = 's16_wda_ncic_nft_off_road_small_b16_ngpu_ep300_483x769_pooling_ssp_20190822-102823'
fold_name = 's8_wda_ncic_nft_freiburg_forest_b2_1gpu_ep300_448x448_pooling_spp_20190825-202117'
fold_name = 's16_wda_ncic_nft_off_road_small_b8_ngpu_ep300_483x769_pooling_aspp_20190815-151551'
logdir=build_path+"logs/fit/" + fold_name #Continue
if initial_epoch == -1: #get last checkpoint epoch
    names = glob.glob(logdir+'*/weights.*')
    names.sort()
    initial_epoch = int(names[-1].split('.')[-4].split('-')[0])

weights_path = glob.glob(logdir+'*/weights.*'+str(initial_epoch)+'-*')[0]
print('Continuing train from '+ weights_path)
if multigpu_enable:
    with strategy.scope():
        cmsnet.load_weights(weights_path)
else:
    cmsnet.load_weights(weights_path)





# class_weight = {cls:1 for cls in range(n_classes)}
# class_weight[19] = 0

# class_weight = np.ones(n_classes)
# #ignore the last label
# if ignore_label:
#     class_weight[-1] = 0


# def _add_sample_weight(image, label):
#     sample_weight = tf.where(tf.equal(label, n_classes-1), 0, 1)
#     sample_weight_vector = tf.reshape(sample_weight, (batch_size, height*width))
#     return (image, label, sample_weight_vector)
#     #return class_weight
# ds_test = ds_test.map(_add_sample_weight, num_parallel_calls=tf.data.experimental.AUTOTUNE)



result = cmsnet.evaluate(ds_test, use_multiprocessing=True, steps=test_steps)



if multigpu_enable:
    with strategy.scope():
        weights = miou.get_weights()
        r = miou.result().numpy()
else:
    weights = miou.get_weights()
    r = miou.result().numpy()

# miouig = MIoU(num_classes=n_classes-1)
# miouig.set_weights([weights[0][:-1, :-1]])
classes_ids = [0, 1, 2, 3, 4]
classes_ids = [0, 1, 2, 3, 4, 6]

# miouig = MIoU(num_classes=len(classes_ids))



# miouig.set_weights([weights[0][classes_ids, :][:,classes_ids]])

# rig = miouig.result().numpy()

# print('mIoU: '+str(round(rig*10000)/100)+'%, mIoU (with ign cls): '+str(round(r*10000)/100)+'%')

IoU_sum = 0
IoU_wf=[]
mIoU_wf=0
nii_sum=0
ti_sum=0
IoU = []
CPacc=[]
for class_id in range(len(weights[0])):
    nii = weights[0][class_id,class_id] #correctly inferred
    ti = weights[0][:,class_id].sum() #ground-truth
    sumjnij= weights[0][class_id,:].sum() #all classes inferred ad truth
    IoU.append(nii/(ti+sumjnij-nii))
    IoU_sum += IoU[-1]
    nii_sum+=nii
    ti_sum+=ti
    if ti > 0:
        CPacc.append(nii/ti)
        IoU_wf.append(IoU[-1]*(ti/np.array(weights).sum()))
        mIoU_wf +=IoU_wf[-1]
    else:
        CPacc.append(0)
    if class_id in classes_ids:
        print('class: '+ classes[class_id]['name']+'\t IoU: '+str(round(IoU[-1]*10000)/100)+',\t CPacc: '+str(str(round(CPacc[-1]*10000)/100)))
mIoU = 100*IoU_sum/(class_id+1)


print('mIoU: '+str(round(np.array(IoU)[classes_ids].mean()*10000)/100)
    +'%, mIoU (with ign cls): '+str(round(r*10000)/100)+'%, '
    + 'FWmIoU: '+str(round(mIoU_wf*10000)/100)
    +', mCPacc: '+str(round(np.array(CPacc)[classes_ids].mean()*10000)/100)+'%'
    +', Pacc: '+str(round(nii_sum/ti_sum*10000)/100)+'%')




number_of_testes = 3000
test_divisor = 10
input_test = np.random.rand(int(number_of_testes/test_divisor),1,height_crop, width_crop,3)


cmsnet.predict(np.ones((1,height_crop, width_crop,3)))
cmsnet.predict(np.ones((1,height_crop, width_crop,3)))

inference_time = 0
for i in range(test_divisor):
    for image in input_test:
        start = time.time()
        data_output = cmsnet.predict(image)
        inference_time += time.time()-start

inference_time = inference_time/number_of_testes
fps = 1/inference_time
inference_time = inference_time*1000

print('inference_time: '+str(round(inference_time*100)/100)
      +'ns, fps: '+str(round(fps*100)/100))



input_test = np.random.rand(int(number_of_testes/test_divisor),4,height_crop, width_crop,3)

cmsnet.predict(np.ones((1,height_crop, width_crop,3)))
cmsnet.predict(np.ones((1,height_crop, width_crop,3)))

inference_time = 0
for i in range(test_divisor):
    for image in input_test:
        start = time.time()
        data_output = cmsnet.predict(image)
        inference_time += (time.time()-start)/4

inference_time = inference_time/number_of_testes
fps = 1/inference_time
inference_time = inference_time*1000

print('for batch 4: inference_time: '+str(round(inference_time*100)/100)
      +'ns, fps: '+str(round(fps*100)/100))
