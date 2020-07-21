# -*- coding: utf-8 -*-
import json
import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')
from evaluation.trained import eval_models_conditions, print_result, print_result2

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
                # 'class_imbalance_correction' : False, 
                # 'data_augmentation' : True, 
                # 'multigpu_enable' : True,
                # 'batch_size' : 2,
                # 'epochs' : 200,
                # 'fine_tune_last' : False, 
                # },
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
 
    
                
condition_cases = {"day_dusty": 
                   [("evening[:0%]"+
                      "+rain[:0%]"+
                      "+day[:0%]"+
                      "+day_offroad_clean[:258]"+ #100%
                      "+day_offroad_dusty[:0%]"+
                      "+night_offroad_clean[:0%]"+
                      "+night_offroad_dusty[:0%]"),
                   
                   ("evening[:0%]"+
                      "+rain[:0%]"+
                      "+day[:0%]"+
                      "+day_offroad_clean[-228:]"+  #80%
                      "+day_offroad_dusty[:57]"+   #20%
                      "+night_offroad_clean[:0%]"+
                      "+night_offroad_dusty[:0%]"),
                   
                   ("evening[:0%]"+
                      "+rain[:0%]"+
                      "+day[:0%]"+
                      "+day_offroad_clean[-84:]"+#60%
                      "+day_offroad_dusty[:56]"+#40%
                      "+night_offroad_clean[:0%]"+
                      "+night_offroad_dusty[:0%]"),
                   
                   ("evening[:0%]"+
                      "+rain[:0%]"+
                      "+day[:0%]"+
                      "+day_offroad_clean[-38:]"+#40%
                      "+day_offroad_dusty[:57]"+#60%
                      "+night_offroad_clean[:0%]"+
                      "+night_offroad_dusty[:0%]"),
                   
                   ("evening[:0%]"+
                      "+rain[:0%]"+
                      "+day[:0%]"+
                      "+day_offroad_clean[-14:]"+#20%
                      "+day_offroad_dusty[:57]"+#80%
                      "+night_offroad_clean[:0%]"+
                      "+night_offroad_dusty[:0%]"),
                   
                   ("evening[:0%]"+
                      "+rain[:0%]"+
                      "+day[:0%]"+
                      "+day_offroad_clean[:0%]"+
                      "+day_offroad_dusty[:100%]"+
                      "+night_offroad_clean[:0%]"+
                      "+night_offroad_dusty[:0%]"),
                   ],
                   
                    "day_night": 
                    [("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[:258]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                   
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[:258]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:64]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[:258]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:172]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-222:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:333]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-83:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:332]"+
                       "+night_offroad_dusty[:0%]"),
                                        
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[:0%]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:100%]"+
                       "+night_offroad_dusty[:0%]"),
                    ],
                    
                    "day_night_dusty": 
                    [("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[:258]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-256:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:64]"),
                   
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-102:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:68]"),
                   
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-44:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:66]"),
                   
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-17:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:68]"),
                   
                    ("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[:0%]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:100%]"),
                    ],

                    "day_evening": 
                    [("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[:258]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:64]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-256:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:152]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-228:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:153]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-102:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:152]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[-38:]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:100%]"+
                       "+rain[:0%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[:0%]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                    ],
                
                    "day_rain": 
                    [("evening[:0%]"+
                       "+rain[:0%]"+
                       "+day[:367]"+
                       "+day_offroad_clean[:0%]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:0%]"+
                       "+rain[:91]"+
                       "+day[:364]"+
                       "+day_offroad_clean[:0%]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:0%]"+
                       "+rain[:148]"+
                       "+day[:222]"+
                       "+day_offroad_clean[:0%]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:0%]"+
                       "+rain[:147]"+
                       "+day[:98]"+
                       "+day_offroad_clean[:0%]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:0%]"+
                       "+rain[:148]"+
                       "+day[:37]"+
                       "+day_offroad_clean[:0%]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                   
                    ("evening[:0%]"+
                       "+rain[:100%]"+
                       "+day[:0%]"+
                       "+day_offroad_clean[:0%]"+
                       "+day_offroad_dusty[:0%]"+
                       "+night_offroad_clean[:0%]"+
                       "+night_offroad_dusty[:0%]"),
                    ],
                   }


    



# total_num_examples=1386,
# splits={
#     'day': 367,
#     'day_offroad_clean': 258,
#     'day_offroad_dusty': 57,
#     'evening': 153,
#     'night_offroad_clean': 335,
#     'night_offroad_dusty': 68,
#     'rain': 148,

recalcalculate = False
if recalcalculate==True:
    results = {}
    for condition_case in condition_cases:
        case_result = []
        for t, i in zip(test_cases, range(len(test_cases)) ):
            for consdition_split in condition_cases[condition_case]:
            
                #Traing eval_models, eval_models2
                r = eval_models_conditions(pooling = t['pooling'], 
                          backbonetype=t['backbonetype'], 
                          output_stride = t['output_stride'], 
                          residual_shortcut = t['residual_shortcut'],
                          height_image = t['height_image'], 
                          width_image = t['width_image'], 
                          channels = t['channels'], 
                          crop_enable = t['crop_enable'], 
                          height_crop = t['height_crop'], 
                          width_crop = t['width_crop'],
                          class_imbalance_correction = t['class_imbalance_correction'],
                          data_augmentation = t['data_augmentation'], 
                          multigpu_enable = t['multigpu_enable'], 
                          batch_size = t['batch_size'], 
                          epochs = t['epochs'], 
                          fine_tune_last = t['fine_tune_last'],
                          splits = consdition_split)
                case_result.append(r)
                tf.keras.backend.clear_session()
        results[condition_case] = case_result
        
        
        
    with open(build_path+'reslut_conditions.txt', 'w') as file:
        file.write(json.dumps(results))


classes_ids = [0, 1, 2, 3]

with open(build_path+'reslut_conditions.txt', 'r') as file:
    loaded_results = json.loads(file.read())


for split in loaded_results:
    print(split)
    for result in loaded_results[split]:
        result['confusion_matrix']
        result['classes']
        print(result['name'])
        print("Params: "+str(result['count_params']))
        mIoU, FWmIoU, mCPacc, Pacc = print_result(result['classes'], np.array(result['confusion_matrix']), classes_ids)
        result["mIoU"] = mIoU
        result["FWmIoU"] = FWmIoU
        result["mCPacc"] = mCPacc
        result["Pacc"] = Pacc
    
# previous_name=""
# for split in loaded_results:
#     print(split)
#     for result in loaded_results[split]:
#         if previous_name != result['name'].split("_ep200")[0]:
#             print(result['name'].split("_ep200")[0])
#             previous_name = result['name'].split("_ep200")[0]
#         print("mIoU: "+str(result["mIoU"]))
        
previous_name=""
for split in loaded_results:
    previous_name=""
    print("\n% "+split)
    print("0 20 40 60 80 100")
    for result in loaded_results[split]:
        if previous_name != result['name'].split("_ep200")[0]:
            # print(result['name'].split("_ep200")[0])
            print("% "+previous_name)
            previous_name = result['name'].split("_ep200")[0]
        print(str(result["mIoU"])+" ", end="")
    print("% "+previous_name)