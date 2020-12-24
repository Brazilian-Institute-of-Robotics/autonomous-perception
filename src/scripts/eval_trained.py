# -*- coding: utf-8 -*-
import json
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')
from evaluation.trained import eval_models, print_result, eval_models2

# build_path = os.path.dirname(__file__) + '/../../build/'
build_path = os.path.dirname(__file__) + '/../../trained/'


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'






test_cases =[
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
                # 'initial_epoch' : -1, 
                # 'continue_traning' : False,
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
                # 'initial_epoch' : 88, 
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
                # 'initial_epoch' : -1, 
                # 'continue_traning' : False, 
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
                # 'initial_epoch' : -1, 
                # 'continue_traning' : False, 
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
                # 'initial_epoch' : 20, 
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
                # 'initial_epoch' : 173, 
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
                # 'initial_epoch' : -1, 
                # 'continue_traning' : False, 
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
                # 'initial_epoch' : -1, 
                # 'continue_traning' : False, 
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
                # 'initial_epoch' : 16, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                
                
                
                
                
                
                
                
                
                #######################################################
                ###############################################
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
                
                {'pooling' : 'aspp',
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
                 
                'multigpu_enable' : True,
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
                
                {'pooling' : 'global',
                'backbonetype' : 'mobilenetv2',
                'output_stride' : 16,
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
                
                'batch_size' : 4,
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
                'output_stride' : 16,
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

                'batch_size' : 4,
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
                
                {'pooling' : 'aspp',
                  'backbonetype' : 'mobilenetv2',
                'output_stride' : 16,
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
                {'pooling' : 'global',
                'backbonetype' : 'mobilenetv2',
                'output_stride' : 16,
                'residual_shortcut' : True,
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
                
                'batch_size' : 4,
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
                'output_stride' : 16,
                'residual_shortcut' : True,
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

                'batch_size' : 4,
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
                {'pooling' : 'aspp',
                  'backbonetype' : 'mobilenetv2',
                'output_stride' : 16,
                'residual_shortcut' : True,
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
                
                ########################################################
                # {'pooling' : 'global',
                #   'backbonetype' : 'mobilenetv2',
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
                
                # 'batch_size' : 16,
                # 'epochs' : 300,
                # 'initial_epoch' : -1, 
                # 'continue_traning' : False,
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'spp',
                #   'backbonetype' : 'mobilenetv2',
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

                # 'batch_size' : 16,
                # 'epochs' : 300,
                # 'initial_epoch' : -1, 
                # 'continue_traning' : False, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'aspp',
                #   'backbonetype' : 'mobilenetv2',
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
                
                # 'batch_size' : 8,
                # 'epochs' : 300,
                # 'initial_epoch' : 85,
                # 'continue_traning' : False,
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
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
                 
                # 'multigpu_enable' : True,
                # 'debug_en' : False,
                
                # 'batch_size' : 16,
                # 'epochs' : 300,
                # 'initial_epoch' : -1, 
                # 'continue_traning' : False, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # ######################################################33
                # {'pooling' : 'spp',
                #  'backbonetype' : 'mobilenetv2',
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
                # {'pooling' : 'aspp',
                #   'backbonetype' : 'mobilenetv2',
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
                
                # 'batch_size' : 2,
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
                # {'pooling' : 'global',
                # 'backbonetype' : 'mobilenetv2',
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
                 
                # 'multigpu_enable' : False,
                # 'debug_en' : False,
                
                # 'batch_size' : 4,
                # 'epochs' : 200,
                # 'initial_epoch' : -1, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'spp',
                #   'backbonetype' : 'mobilenetv2',
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
                 
                # 'multigpu_enable' : False,
                # 'debug_en' : False,

                # 'batch_size' : 4,
                # 'epochs' : 200,
                # 'initial_epoch' : -1, 
                # 'continue_traning' : True, 
                # 'fine_tune_last' : False, 
                
                # 'base_learning_rate' : 0.007, 
                # 'learning_power' : 0.98, 
                # 'decay_steps' : 1,
                # 'learning_rate_decay_step' : 4,
                # 'decay' : 5**(-4)
                # },
                # {'pooling' : 'aspp',
                #   'backbonetype' : 'mobilenetv2',
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
                 
                # 'multigpu_enable' : False,
                # 'debug_en' : False,
                
                # 'batch_size' : 2,
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
                #   ]


recalcalculate = False
if recalcalculate==True:
    results = []
    for t, i in zip(test_cases, range(len(test_cases)) ):
        
        #Traing eval_models, eval_models2
        r = eval_models(pooling = t['pooling'], 
                  backbonetype=t['backbonetype'], 
                  output_stride = t['output_stride'], 
                  residual_shortcut = t['residual_shortcut'],
                  height_image = t['height_image'], 
                  width_image = t['width_image'], 
                  channels = t['channels'], 
                  crop_enable = t['crop_enable'], 
                  height_crop = t['height_crop'], 
                  width_crop = t['width_crop'],
                  debug_en = t['debug_en'], 
                  dataset_name = t['dataset_name'], 
                  class_imbalance_correction = t['class_imbalance_correction'],
                  data_augmentation = t['data_augmentation'], 
                  multigpu_enable = t['multigpu_enable'], 
                  batch_size = t['batch_size'], epochs = t['epochs'], 
                  initial_epoch = t['initial_epoch'],
                  continue_traning = t['continue_traning'], 
                  fine_tune_last = t['fine_tune_last'],
                  base_learning_rate = t['base_learning_rate'], 
                  learning_power = t['learning_power'], 
                  decay_steps = t['decay_steps'], 
                  learning_rate_decay_step = t['learning_rate_decay_step'],
                  decay = t['decay'])
        results.append(r)
        tf.keras.backend.clear_session()
        
        
        
    with open(build_path+'reslut.txt', 'w') as file:
        file.write(json.dumps(results))


classes_ids = [0, 1, 2, 3]

with open(build_path+'reslut.txt', 'r') as file:
    loaded_results = json.loads(file.read())


for result in loaded_results:
    result['confusion_matrix']
    result['classes']
    print(result['name'])
    print("Params: "+str(result['count_params']))
    print_result(result['classes'], np.array(result['confusion_matrix']), classes_ids)
    


