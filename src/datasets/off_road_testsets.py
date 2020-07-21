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

class OffRoadTestsets(tfds.core.GeneratorBasedBuilder):
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
        
        paths = ['/off-road/evening/cimatec-industrial/small.txt',
         '/off-road/rain/cimatec-industrial/small.txt',
         '/unpaved/rain/jaua/small.txt',
         '/unpaved/rain/praia-do-forte/small.txt',
         '/unpaved/rain/estrada-dos-tropeiros/small.txt',
         '/unpaved/day/jaua/small.txt',
         '/unpaved/day/praia-do-forte/small.txt',
         '/unpaved/day/estrada-dos-tropeiros/small.txt']
        
        offroad_paths = ['/night_offroad_clean-test_subset.txt',
        '/day_offroad_clean-test_subset.txt',
        '/night_offroad_dusty-test_subset.txt',
        '/day_offroad_dusty-test_subset.txt']
        
        
        img_list = {"evening":[], "rain":[], "day":[], "day_offroad_clean":[],
                    "day_offroad_dusty":[], "night_offroad_clean":[], 
                    "night_offroad_dusty":[]}
        lbl_list = {"evening":[],"rain":[],"day":[], "day_offroad_clean":[],
                    "day_offroad_dusty":[], "night_offroad_clean":[], 
                    "night_offroad_dusty":[]}
        
        
        for path in paths:
            print(path)
            lineList = open(extracted_path + path).readlines()
            for name in lineList:
                search_name_path = path[:path.rfind('/')]+'/'+name.replace('\n', '')
                full_name_path = tf.io.gfile.glob(extracted_path + search_name_path + label_suffix)[0]
                if 'test' in full_name_path:
                    img_list[path.split("/")[2]].append(extracted_path + search_name_path + image_suffix)
                    lbl_list[path.split("/")[2]].append(full_name_path)
        
        for path in offroad_paths:
            print(path)
            lineList = open(extracted_path + path).readlines()
            for name in lineList:
                search_name_path = path[:path.rfind('/')]+'/'+name.replace('.jpg\n', '')
                full_name_path = tf.io.gfile.glob(extracted_path + search_name_path + label_suffix)[0]
                if 'test' in full_name_path:
                    img_list[path.replace('/', '').split("-")[0]].append(extracted_path + search_name_path + image_suffix)
                    lbl_list[path.replace('/', '').split("-")[0]].append(full_name_path)


        for imgs, lbls in zip(img_list, lbl_list):
            random.Random(0).shuffle(img_list[imgs])
            random.Random(0).shuffle(lbl_list[lbls])
        
        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name="evening",
                gen_kwargs={
                    "img_list": img_list["evening"],
                    "lbl_list": lbl_list["evening"],
                },
            ),
            tfds.core.SplitGenerator(
                name="rain",
                gen_kwargs={
                    "img_list": img_list["rain"],
                    "lbl_list": lbl_list["rain"],
                },
            ),
            tfds.core.SplitGenerator(
                name="day",
                gen_kwargs={
                    "img_list": img_list["day"],
                    "lbl_list": lbl_list["day"],
                },
            ),
            tfds.core.SplitGenerator(
                name="day_offroad_clean",
                gen_kwargs={
                    "img_list": img_list["day_offroad_clean"],
                    "lbl_list": lbl_list["day_offroad_clean"],
                },
            ),
            tfds.core.SplitGenerator(
                name="day_offroad_dusty",
                gen_kwargs={
                    "img_list": img_list["day_offroad_dusty"],
                    "lbl_list": lbl_list["day_offroad_dusty"],
                },
            ),
            tfds.core.SplitGenerator(
                name="night_offroad_clean",
                gen_kwargs={
                    "img_list": img_list["night_offroad_clean"],
                    "lbl_list": lbl_list["night_offroad_clean"],
                },
            ),
            tfds.core.SplitGenerator(
                name="night_offroad_dusty",
                gen_kwargs={
                    "img_list": img_list["night_offroad_dusty"],
                    "lbl_list": lbl_list["night_offroad_dusty"],
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
