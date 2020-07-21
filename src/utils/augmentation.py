#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 22:37:41 2019

@author: nelson
"""


import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(__file__) + '/..')
# from datasets.citysmall import CityDataset




def plot_images(dataset, n_images, samples_per_image, classes):
    height = [a[0].shape[-3] for a in dataset.take(1)][0]
    width = [a[0].shape[-2] for a in dataset.take(1)][0]

    output = np.zeros((height * n_images, width*2 * samples_per_image, 3))
    mkhsv = np.ones((n_images, height, width, 3))
    row = 0
    newdata =dataset.take(n_images)
    for images, mask in newdata.repeat(samples_per_image).batch(n_images):
        output[:, row*width:(row+1)*width] = np.vstack((images.numpy()+1)/2)
        row += 1

        # mkhsv[...,0] = (tf.cast(mask,dtype=tf.float32)/(255.0/(n_classes+1)))[...,0]
        # output[:, row*width:(row+1)*width] = np.vstack(tf.image.hsv_to_rgb(mkhsv).numpy())
        maskpy = mask.numpy()
        output[:, row*width:(row+1)*width] = np.vstack(
                np.array([classes[id]['color'] for id in maskpy.reshape(n_images*height*width)]).reshape(n_images,height, width,3)/255.0
                )


        row += 1

    plt.figure()
    plt.imshow(output)
    plt.show()
    Image.fromarray(np.uint8(output*255)).show()


def flip(x: tf.Tensor, y:tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x, y = tf.cond(choice < 0.5, lambda: (x, y), lambda: (tf.image.flip_left_right(x),  tf.image.flip_left_right(y)))

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    x, y = tf.cond(choice < 0.5, lambda: (x, y), lambda: (tf.image.flip_up_down(x),  tf.image.flip_up_down(y)))

    return x, y

def color(x: tf.Tensor, y:tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    std = tf.random.uniform(shape=[], minval=0., maxval=.4, dtype=tf.float32)
    # noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=std, dtype=tf.float32)
    # x = x + noise
    return x, y


def rotate(x: tf.Tensor, y:tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    height = x.shape[-3]
    width  = x.shape[-2]

    rand = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    return (tf.image.resize_with_crop_or_pad(tf.image.rot90(x, rand),height, width),
            tf.cast(tf.image.resize_with_crop_or_pad(tf.image.rot90(y, rand),height, width), dtype=tf.uint8))

def zoom(x: tf.Tensor, y:tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    height = x.shape[-3]
    width  = x.shape[-2]
    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.7, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop_img(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(height, width))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]

    def random_crop_mask(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(height, width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Return a random crop
        return tf.cast(crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)], dtype=tf.uint8)

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: (x, y), lambda: (random_crop_img(x), random_crop_mask(y)))


def zoom2(x: tf.Tensor, y:tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    height = x.shape[-3]
    width  = x.shape[-2]

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    x1 = tf.random.uniform(shape=[], minval=-0.2, maxval=.2, dtype=tf.float32)
    y1 = tf.random.uniform(shape=[], minval=-0.2, maxval=.2, dtype=tf.float32)
    x2 = tf.random.uniform(shape=[], minval=.8, maxval=1.2, dtype=tf.float32)
    y2 = tf.random.uniform(shape=[], minval=.8, maxval=1.2, dtype=tf.float32)


    boxe = [[x1, y1, x2, y2]]

    # Only apply cropping 50% of the time
    return (tf.image.crop_and_resize([x], boxes=boxe, box_indices=[0], crop_size=[height, width])[0],
            tf.cast(tf.image.crop_and_resize([y], boxes=boxe, box_indices=[0], crop_size=[height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[0], dtype=tf.uint8))



# batch_size = 1




# ds_train, ds_test = tfds.load(name="city_dataset", split=["train", "test"], as_supervised=True)

# height = 483
# width  = 769
# # n_classes = 12
# def _normalize_img(image, label):
#   image = tf.cast(image, tf.float32)/127.5 - 1
#   image = tf.image.resize(image, (height,width), method=tf.image.ResizeMethod.BILINEAR)
#   label = tf.image.resize(label, (height,width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#   return (image, label)

# ds_train = ds_train.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_test = ds_test.map(_normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# # Add augmentations
# augmentations = [flip, color, rotate, zoom2]

# for f in augmentations:
#     ds_train = ds_train.map(lambda x, y: tf.cond(tf.random.uniform([], 0, 1) > 0.7, lambda: f(x, y), lambda: (x, y)),
#                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
# ds_train = ds_train.map(lambda x, y: (tf.clip_by_value(x, -1, 1), y),  num_parallel_calls=tf.data.experimental.AUTOTUNE)

# plot_images(ds_train, n_images=2, samples_per_image=20)


















# # Build your input pipeline
# ds_train = ds_train.shuffle(124).batch(batch_size).prefetch(7)
# ds_test = ds_test.batch(batch_size).prefetch(101)

# for features in ds_train.take(1):
#   image, label = features["image"], features["label"]
# Image.fromarray(label.numpy()[0,...,0])
# Image.fromarray(image.numpy()[0,...,:])

