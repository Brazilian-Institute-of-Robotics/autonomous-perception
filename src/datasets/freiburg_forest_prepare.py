#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:26:10 2019

@author: nelson
"""



import argparse
import os
import os.path as osp
import glob
import PIL.Image as Image
import numpy as np

classes =  [
            #{'name': 'void'         , 'color': [ -1,  -1,  -1], 'id': 6},
            {'name': 'road'         , 'color': [170, 170, 170], 'id': 0},
            {'name': 'grass'        , 'color': [  0, 255,   0], 'id': 1},
            {'name': 'vegetation'   , 'color': [102, 102,  51], 'id': 2},
            {'name': 'tree'         , 'color': [  0,  60,   0], 'id': 2},
            {'name': 'sky'          , 'color': [  0, 120, 255], 'id': 3},
            {'name': 'obstacle'     , 'color': [  0,   0,   0], 'id': 4},
            ]


folders = ['test', 'train']

color_id_folder = 'GT_color'
label_folder = 'label'


def main():
    #rename -n 's/_Clipped/_mask/gi' *.png

    parser = argparse.ArgumentParser()
    parser.add_argument('-freiburg_forest_path', default='/home/nelson/Pictures/datasets/freiburg_forest_annotated')

    args = parser.parse_args()

    print(args.freiburg_forest_path)
    freiburg_forest_path = args.freiburg_forest_path

    count = np.zeros(len(classes)+1)

    for folder in folders:

        ############################Create output dir##############################
        in_dir = osp.join(freiburg_forest_path, folder, color_id_folder)
        out_dir = osp.join(freiburg_forest_path, folder, label_folder)

        if not osp.exists(out_dir):
            os.mkdir(out_dir)


        #Load labels path and generate images path
        colorIdsFilesPath = glob.glob(in_dir + '/*.png', recursive=True)
        print(str(len(colorIdsFilesPath)) + "images with annotation was found in "+folder+'.')

        for file in colorIdsFilesPath:
            img = Image.open(file)
            npimg = np.array(img)
            nplabel = np.ones(npimg.shape[:-1], dtype=np.uint8)*5
            for i in range(len(classes)):
                index = np.logical_and(npimg[...,0]==(classes[i]['color'][0]), npimg[...,1]==(classes[i]['color'][1]), npimg[...,2]==(classes[i]['color'][2]))
                nplabel[index] = classes[i]['id']
                count[i] += index.sum()
            count[-1] += (nplabel==5).sum()
            Image.fromarray(nplabel).save(file.replace(in_dir, out_dir))

    print(count)


if __name__ == '__main__':
    
    main()
