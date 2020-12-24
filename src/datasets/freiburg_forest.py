#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:26:10 2019

@author: nelson
"""


import tensorflow_datasets.public_api as tfds
import os
import tensorflow as tf



classes =  [
            {'name': 'road'         , 'color': [170, 170, 170]},
            {'name': 'grass'        , 'color': [  0, 255,   0]},
            {'name': 'vegetation'   , 'color': [102, 102,  51]},
            {'name': 'sky'          , 'color': [  0, 120, 255]},
            {'name': 'obstacle'     , 'color': [255,   0,   0]},
            ]


class FreiburgForest(tfds.core.GeneratorBasedBuilder):
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
                "image": tfds.features.Image(shape=(None,None , 3)),
                # Here, labels can be of 5 distinct values.
                "label": tfds.features.Image(shape=(None,None , 1)),
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
        extracted_path = "/home/nelson/Pictures/datasets/freiburg_forest_annotated"
    
        # Specify the splits
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=10,
                gen_kwargs={
                    "images_dir_path": os.path.join(extracted_path, "train/rgb"),
                    "labels": os.path.join(extracted_path, "train/label"),
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=1,
                gen_kwargs={
                    "images_dir_path": os.path.join(extracted_path, "test/rgb"),
                    "labels": os.path.join(extracted_path, "test/label"),
                },
            ),
        ]





    def _generate_examples(self, images_dir_path, labels):
        # Read the input data out of the source files
        label_suffix = '_mask.png'
        image_suffix = '_Clipped.jpg'

        key=0
        for label_path in tf.io.gfile.glob(labels+'/*'+label_suffix):
            image_path = label_path.replace(label_suffix, image_suffix
                                            ).replace(labels, images_dir_path)
            yield key, {
                "image": image_path,
                "label": label_path,
            }

            key+=1