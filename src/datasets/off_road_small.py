#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:26:10 2019

@author: nelson
"""


import tensorflow_datasets.public_api as tfds
import os
import tensorflow as tf
import random



classes =  [
            {'name': 'ignore'       , 'color': [  0,  0,  0]},
            {'name': 'road'         , 'color': [128, 64,128]},
            {'name': 'car'          , 'color': [  0,  0,142]},
            {'name': 'person'       , 'color': [220, 20, 60]},
            {'name': 'truck'        , 'color': [  0,  0, 70]},
            {'name': 'bus'          , 'color': [  0, 60,100]},
            {'name': 'cone'         , 'color': [153,153,153]},
            {'name': 'motorcycle'   , 'color': [  0,  0,230]},
            {'name': 'animal'       , 'color': [190,153,153]},
            {'name': 'bicycle'      , 'color': [119, 11, 32]},
            # {'name': 'dog'          , 'color': [ 70,130,180]},
            # {'name': 'traffic light', 'color': [250,170, 30]},
            # {'name': 'traffic sign' , 'color': [220,220,  0]},
            ]

# TODO
# tfds.core.DatasetInfo(
#     name='off_road_small',
#     version=1.0.0,
#     description='This is the dataset for xxx. It contains yyy. The images are kept at their original dimensions.',
#     homepage='https://dataset-homepage.org',
#     features=FeaturesDict({
#         'image': Image(shape=(1208, 1920, 3), dtype=tf.uint8),
#         'label': Image(shape=(1208, 1920, 1), dtype=tf.uint8),
#     }),
#     total_num_examples=5523,
#     splits={
#         'test': 1048,
#         'train': 4027,
#         'validation': 448,
#     },
#     supervised_keys=('image', 'label'),
#     citation="""@article{my-awesome-dataset-2020,
#                                       author = {Nelson Alves, ...},"}""",
#     redistribution_info=,
# )

class OffRoadSmall(tfds.core.GeneratorBasedBuilder):
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
                "image": tfds.features.Image(shape=(1208, 1920, 3)),
                # Here, labels can be of 5 distinct values.
                "label": tfds.features.Image(shape=(1208, 1920, 1)),
            }),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("image", "label"),
            # Homepage of the dataset for documentation
            homepage="https://dataset-homepage.org",

            # Bibtex citation for the dataset
            citation=r"""@article{my-awesome-dataset-2020,
                                  author = {Nelson Alves, ...},"}""",
        )

    def _split_generators(self, dl_manager):
        # Download source data
        extracted_path = "/home/nelson/projects/da_art_perception/data/dataset"
        label_suffix = '*label_raw.png'
        image_suffix = '.jpg'

        img_list = []
        lbl_list = []
        test_img_list = []
        test_lbl_list = []


        for list_path in tf.io.gfile.glob(extracted_path+'/**/**/**/*small.txt'):
            print(list_path)
            lineList = open(list_path).readlines()
            for name in lineList:
                search_name_path = list_path[:list_path.rfind('/')]+'/'+name.replace('\n', '')
                full_name_path = tf.io.gfile.glob(search_name_path + label_suffix)[0]
                if 'train' in full_name_path:
                    img_list.append(search_name_path + image_suffix)
                    lbl_list.append(full_name_path)
                elif 'test' in full_name_path:
                    test_img_list.append(search_name_path + image_suffix)
                    test_lbl_list.append(full_name_path)
                    
                    
        split = round(len(img_list)*.10)
        random.Random(0).shuffle(img_list)
        random.Random(0).shuffle(lbl_list)
        val_img_list = img_list[:split]
        val_lbl_list = lbl_list[:split]
        train_img_list = img_list[split:]
        train_lbl_list = lbl_list[split:]

        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                # num_shards=10,
                gen_kwargs={
                    "img_list": train_img_list,
                    "lbl_list": train_lbl_list,
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                # num_shards=10,
                gen_kwargs={
                    "img_list": val_img_list,
                    "lbl_list": val_lbl_list,
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                # num_shards=1,
                gen_kwargs={
                    "img_list": test_img_list,
                    "lbl_list": test_lbl_list,
                },
            ),
        ]


    def _generate_examples(self, img_list, lbl_list):
        # Read the input data out of the source files

        key=0
        for image_path, label_path in zip(img_list, lbl_list):
            yield key, {
                "image": image_path,
                "label": label_path,
            }
            key+=1
