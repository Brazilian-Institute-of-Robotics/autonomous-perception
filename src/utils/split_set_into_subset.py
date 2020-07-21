#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 23:26:46 2020

@author: nelson
"""

import tensorflow as tf
import os 

# dataset_base = "/home/nelson/projects/da_art_perception/data/dataset"
# night_dust_ts = "/night_dust-test_subset.txt"
# label_suffix = '*label_raw.png'
# night_ts = '/**/night/**/' + label_suffix

# night_dust_paths = open(dataset_base + night_dust_ts, "r").readlines()


# test = []
# test_label = []
# train = []
# nigth_test = []
# nigth_test_label = []
# nigth_train = []
# for nigth_path in tf.io.gfile.glob(dataset_base + night_ts):
#     thereis = False
#     for night_dust_path in night_dust_paths:
#         if nigth_path.split("/")[-1].split("-")[0] == night_dust_path.split("-")[0]:
#             if nigth_path.split("/")[-1].split("-")[1] == "test":
#                 # print("Test path found: " + nigth_path)
#                 test.append(nigth_path.split(
#                     "/home/nelson/projects/da_art_perception/data/dataset/")[1]
#                     .split("-test-label_raw.png")[0]+".jpg")
#                 test_label.append(nigth_path.split(
#                     "/home/nelson/projects/da_art_perception/data/dataset/")[1])
#             else:
#                 # print("Train path found: " + nigth_path)
#                 train.append(nigth_path)
#             thereis = True
#             #night_dust_paths.remove(night_dust_path)
#     if not thereis:
#         if nigth_path.split("/")[-1].split("-")[1] == "test":
#             # print("Test path found: " + nigth_path)
#             nigth_test.append(nigth_path.split(
#                     "/home/nelson/projects/da_art_perception/data/dataset/")[1]
#                     .split("-test-label_raw.png")[0]+".jpg")
#             nigth_test_label.append(nigth_path.split(
#                     "/home/nelson/projects/da_art_perception/data/dataset/")[1])
#         else:
#             # print("Train path found: " + nigth_path)
#             nigth_train.append(nigth_path)
#         thereis = True
        
# import shutil as sh

# os.makedirs("/home/nelson/projects/da_art_perception/build/debug/nigth_dust", exist_ok=True)
# for file in test:
#     sh.copy(dataset_base + "/" + file,
#                 "/home/nelson/projects/da_art_perception/build/debug/nigth_dust/" + file.split("/")[-1])



# os.makedirs("/home/nelson/projects/da_art_perception/build/debug/nigth_clean", exist_ok=True)
# for file in nigth_test:
#     sh.copy(dataset_base + "/" + file,
#                 "/home/nelson/projects/da_art_perception/build/debug/nigth_clean/" + file.split("/")[-1])




def generate_and_split_subset(base_path, dataset_base_path, dusty_ts, clean_ts, 
                              dusty_intervals, img_suffix, label_suffix,
                              search_tag, debug_en = False) :
    
    # night_dust_paths = open(dataset_base_path + dusty_ts, "r").readlines()
    night_dusty_paths = []
    dusty_intervals_paths = open(dataset_base_path + dusty_intervals, "r").readlines()
    list_files = tf.io.gfile.glob(dataset_base_path + search_tag + img_suffix)
    list_files.sort()
    for nigth_path in list_files:
        nigth_path = nigth_path.split("/")[-1]
        if nigth_path == dusty_intervals_paths[0].split("\n")[0]:
            dusty_intervals_paths.pop(0)
            night_dusty_paths.append(nigth_path)
        elif dusty_intervals_paths[0] == "...\n":
            night_dusty_paths.append(nigth_path)
            if nigth_path == dusty_intervals_paths[1].split("\n")[0]:
                 dusty_intervals_paths.pop(0)
                 dusty_intervals_paths.pop(0)
             
             
    night_dusty_test = []
    night_dusty_test_label = []
    night_dusty_train = []
    nigth_clean_test = []
    nigth_clean_test_label = []
    nigth_clean_train = []
    for nigth_path in tf.io.gfile.glob(dataset_base_path + search_tag + label_suffix):
        thereis = False
        for night_dust_path in night_dusty_paths:
            if nigth_path.split("/")[-1].split("-")[0] == night_dust_path.split(".")[0]:
                if nigth_path.split("/")[-1].split("-")[1] == "test":
                    # print("Test path found: " + nigth_path)
                    night_dusty_test.append(nigth_path.split(
                        dataset_base_path+"/")[1]
                        .split("-test-label_raw.png")[0]+".jpg")
                    night_dusty_test_label.append(nigth_path.split(
                        dataset_base_path+"/")[1])
                else:
                    # print("Train path found: " + nigth_path)
                    night_dusty_train.append(nigth_path)
                thereis = True
                #night_dusty_paths.remove(night_dust_path)
        if not thereis:
            if nigth_path.split("/")[-1].split("-")[1] == "test":
                # print("Test path found: " + nigth_path)
                nigth_clean_test.append(nigth_path.split(
                        dataset_base_path+"/")[1]
                        .split("-test-label_raw.png")[0]+".jpg")
                nigth_clean_test_label.append(nigth_path.split(
                        dataset_base_path+"/")[1])
            else:
                # print("Train path found: " + nigth_path)
                nigth_clean_train.append(nigth_path)
            thereis = True
            
    import shutil as sh
    
    if debug_en:
        os.makedirs(base_path + "/build/debug/dust", exist_ok=True)
        for file in night_dusty_test:
            sh.copy(dataset_base_path + "/" + file,
                        base_path + "/build/debug/dust/" + file.split("/")[-1])
    with open(dataset_base_path + dusty_ts, "w") as filehandle:
        filehandle.writelines("%s\n" % place for place in night_dusty_test)
    
    if debug_en:
        os.makedirs(base_path + "/build/debug/clean", exist_ok=True)
        for file in nigth_clean_test:
            sh.copy(dataset_base_path + "/" + file,
                        base_path + "/build/debug/clean/" + file.split("/")[-1])
    with open(dataset_base_path + clean_ts, "w") as filehandle:
        filehandle.writelines("%s\n" % place for place in nigth_clean_test)

    return


base="/home/nelson/projects/da_art_perception"
dataset_base_path = base+"/data/dataset"
day_dusty_man = "/day_dusty_all-man.txt"
night_dusty_man = "/night_dusty_all-man.txt"

day_dusty_ts = "/day_offroad_dusty-test_subset.txt"
night_dusty_ts = "/night_offroad_dusty-test_subset.txt"
day_clean_ts = "/day_offroad_clean-test_subset.txt"
night_clean_ts = "/night_offroad_clean-test_subset.txt"

label_suffix = '*label_raw.png'
img_suffix = "*.jpg"
night_ts = '/**/night/**/'
day_ts = '/off-road/day/**/'




             
             
             
generate_and_split_subset(base_path = base,
                          dataset_base_path = dataset_base_path, 
                          dusty_ts = night_dusty_ts,
                          clean_ts = night_clean_ts,
                          dusty_intervals = night_dusty_man,
                          img_suffix = img_suffix,
                          label_suffix = label_suffix,
                          search_tag = night_ts,
                          debug_en = True)

generate_and_split_subset(base_path = base,
                          dataset_base_path = dataset_base_path, 
                          dusty_ts = day_dusty_ts,
                          clean_ts = day_clean_ts,
                          dusty_intervals = day_dusty_man,
                          img_suffix = img_suffix,
                          label_suffix = label_suffix,
                          search_tag = day_ts,
                          debug_en = True)