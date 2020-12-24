#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 17:26:08 2019

@author: nelson
"""

import tensorflow_datasets.public_api as tfds
import os
import tensorflow as tf

classes =  [{'name': 'road'         , 'color': [128, 64,128]},
            {'name': 'sidewalk'     , 'color': [244, 35,232]},
            {'name': 'building'     , 'color': [ 70, 70, 70]},
            {'name': 'wall'         , 'color': [102,102,156]},
            {'name': 'pole'         , 'color': [153,153,153]},
            {'name': 'traffic light', 'color': [250,170, 30]},
            {'name': 'vegetation'   , 'color': [107,142, 35]},
            {'name': 'sky'          , 'color': [ 70,130,180]},
            {'name': 'person'       , 'color': [220, 20, 60]},
            {'name': 'car'          , 'color': [  0,  0,142]},
            {'name': 'truck'        , 'color': [  0,  0, 70]},
            {'name': 'motorcycle'   , 'color': [  0,  0,230]}]

class Citysmall(tfds.core.GeneratorBasedBuilder):
    """Short description of my dataset."""
    num_classes = 12

    VERSION = tfds.core.Version("0.1.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            # This is the description that will appear on the datasets page.
            description=("This is the dataset for xxx. It contains yyy. The "
                         "images are kept at their original dimensions."),
            # tfds.features.FeatureConnectors
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(360, 480, 3)),
                # Here, labels can be of 5 distinct values.
                "label": tfds.features.Image(shape=(360, 480, 1)),
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
        extracted_path = "/home/nelson/projects/learning/data/dataset1"
    
        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=10,
                gen_kwargs={
                    "images_dir_path": os.path.join(extracted_path, "images_prepped_train"),
                    "labels": os.path.join(extracted_path, "annotations_prepped_train"),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=1,
                gen_kwargs={
                    "images_dir_path": os.path.join(extracted_path, "images_prepped_test"),
                    "labels": os.path.join(extracted_path, "annotations_prepped_test"),
                },
            ),
        ]


    def _generate_examples(self, images_dir_path, labels):
        # Read the input data out of the source files
        key=0
        for image_file in tf.io.gfile.listdir(images_dir_path):
            yield key, {
                "image": "%s/%s" % (images_dir_path, image_file),
                "label": "%s/%s" % (labels, image_file),
            }
            key +=1
    # def _generate_examples(self, images_dir_path, labels):
    #     # Read the input data out of the source files
    #     for image_file in tf.io.gfile.listdir(images_dir_path):
    #         ...
    #     with tf.io.gfile.GFile(labels) as f:
    #         ...

    #     # And yield examples as feature dictionaries
    #     for image_id, description, label in data:
    #         yield image_id, {
    #             "image_description": description,
    #             "image": "%s/%s.jpeg" % (images_dir_path, image_id),
    #             "label": label,
    #         }
    



# tfds works in both Eager and Graph modes
#tf.enable_eager_execution()

# # See available datasets
# print(tfds.list_builders())

# # Construct a tf.data.Dataset
# ds_train, ds_test = tfds.load(name="city_dataset", split=["train", "test"])

# # Build your input pipeline
# ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
# for features in ds_train.take(1):
#   image, label = features["image"], features["label"]

    
#import tensorflow as tf
##from tensorflow_datasets import my_dataset
#import tensorflow_datasets.testing as tfds_test
#
#
#class MyDatasetTest(tfds_test.DatasetBuilderTestCase):
##  DATASET_CLASS = my_dataset.MyDataset
#  DATASET_CLASS = CityDataset
#
#  SPLITS = {  # Expected number of examples on each split from fake example.
#      "train": 367,
#      "test": 101,
#  }
#  # If dataset `download_and_extract`s more than one resource:
##  DL_EXTRACT_RESULT = {
##      "images_dir_path": "/home/nelson/projects/learning/data/dataset1/images_prepped_train/",  # Relative to fake_examples/my_dataset dir.
###      "labels": "file2",
##  }
#
#if __name__ == "__main__":
#  tfds_test.test_main()