#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 00:54:08 2019

@author: nelson
"""

import tensorflow_datasets.public_api as tfds
import os
import tensorflow as tf



classes =  [{'name': 'road'         , 'color': [128, 64,128]},
            {'name': 'sidewalk'     , 'color': [244, 35,232]},
            {'name': 'building'     , 'color': [ 70, 70, 70]},
            {'name': 'wall'         , 'color': [102,102,156]},
            {'name': 'fence'        , 'color': [190,153,153]},
            {'name': 'pole'         , 'color': [153,153,153]},
            {'name': 'traffic light', 'color': [250,170, 30]},
            {'name': 'traffic sign' , 'color': [220,220,  0]},
            {'name': 'vegetation'   , 'color': [107,142, 35]},
            {'name': 'terrain'      , 'color': [152,251,152]},
            {'name': 'sky'          , 'color': [ 70,130,180]},
            {'name': 'person'       , 'color': [220, 20, 60]},
            {'name': 'rider'        , 'color': [255,  0,  0]},
            {'name': 'car'          , 'color': [  0,  0,142]},
            {'name': 'truck'        , 'color': [  0,  0, 70]},
            {'name': 'bus'          , 'color': [  0, 60,100]},
            {'name': 'train'        , 'color': [  0, 80,100]},
            {'name': 'motorcycle'   , 'color': [  0,  0,230]},
            {'name': 'bicycle'      , 'color': [119, 11, 32]},
            {'name': 'ignore'       , 'color': [  0,  0, 0]}]


class Cityscape(tfds.core.GeneratorBasedBuilder):
    """Short description of my dataset."""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=("This is the dataset for xxx. It contains yyy. The "
                         "images are kept at their original dimensions."),
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(1024, 2048, 3)),
                # Here, labels can be of 5 distinct values.
                "label": tfds.features.Image(shape=(1024, 2048, 1)),
            }),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("image", "label"),
            # Homepage of the dataset for documentation
            urls=["https://dataset-homepage.org"],

            # Bibtex citation for the dataset
            citation=r"""@article{my-awesome-dataset-2020,
                                  author = {Smith, John},"}""",
        )

    def _split_generators(self, dl_manager):
        # Download source data
        extracted_path = "/home/nelson/Pictures/datasets/cityscape"
    
        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=10,
                gen_kwargs={
                    "images_dir_path": os.path.join(extracted_path, "leftImg8bit/train"),
                    "labels": os.path.join(extracted_path, "gtFine/train"),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=1,
                gen_kwargs={
                    "images_dir_path": os.path.join(extracted_path, "leftImg8bit/val"),
                    "labels": os.path.join(extracted_path, "gtFine/val"),
                },
            ),
        ]


    def _generate_examples(self, images_dir_path, labels):
        # Read the input data out of the source files
        label_suffix = '_gtFine_labelTrainIds.png'
        image_suffix = '_leftImg8bit.png'

        key=0
        for label_path in tf.io.gfile.glob(labels+'/*/*'+label_suffix):
            image_path = label_path.replace(label_suffix, image_suffix
                                            ).replace(labels, images_dir_path)
            yield key, {
                "image": image_path,
                "label": label_path,
            }

            key+=1