# -*- coding: utf-8 -*-

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/..')
from models.cmsnet import CMSNet
from models.others import CnnsFcn, Darkfcn, UpNet
from evaluation.others import load_model
import time
build_path = os.path.dirname(__file__) + '/../../build/'




gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def inference_test(test_case, test_divisor, infb4):
    
    
    number_of_testes= test_case['ntests']
    test_size =int(number_of_testes/test_divisor)
    inference_time_b4 = np.array([])
    
    if ("CMSNet" in test_case['name'] or "cnns_fcn" in test_case['name'] or 
        "darkfcn" in test_case['name'] or "upnet" in test_case['name']):
        
        height_crop=  test_case['height']
        width_crop= test_case['width']
        channels= test_case['ch']
        output_stride= test_case['os']
        pooling= test_case['pool']
        n_classes= test_case['cls']
        residual_shortcut= test_case['residual']
        
        
        if test_case['mltgpu']:
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                if "CMSNet" in test_case['name']:
                    model = CMSNet(dl_input_shape=(None, height_crop, 
                            width_crop, channels), num_classes=n_classes, 
                            output_stride=output_stride, pooling=pooling, 
                            residual_shortcut=residual_shortcut)
                elif "cnns_fcn" in test_case['name']:
                    model = CnnsFcn(input_shape=(height_crop, width_crop, channels), 
                                    n_classes=n_classes)
                elif "darkfcn" in test_case['name']:
                    model = Darkfcn(input_shape=(height_crop, width_crop, channels), 
                                    n_classes=n_classes)
                elif "upnet" in test_case['name']:
                    model = UpNet(input_shape=(height_crop, width_crop, channels), 
                                  n_classes=n_classes)
                else:
                    print("Error...:")
                #model.summary()
                print("Parameters: "+str(model.count_params()))
        
        else:
            if "CMSNet" in test_case['name']:
                model = CMSNet(dl_input_shape=(None, height_crop, 
                        width_crop, channels), num_classes=n_classes, 
                        output_stride=output_stride, pooling=pooling, 
                        residual_shortcut=residual_shortcut)
            elif "cnns_fcn" in test_case['name']:
                model = CnnsFcn(input_shape=(height_crop, width_crop, channels), 
                                n_classes=n_classes)
            elif "darkfcn" in test_case['name']:
                model = Darkfcn(input_shape=(height_crop, width_crop, channels), 
                                n_classes=n_classes)
            elif "upnet" in test_case['name']:
                model = UpNet(input_shape=(height_crop, width_crop, channels), 
                              n_classes=n_classes)
            else:
                print("Error...:")
            #model.summary()
    
        
        image = np.random.rand(number_of_testes, 1,height_crop, width_crop,3)
        
        model.predict(np.ones((1,height_crop, width_crop,3)))
        model.predict(np.ones((1,height_crop, width_crop,3)))
        
        inference_time = np.zeros(number_of_testes)
        for i in range(test_divisor):
            for j in range(test_size):
                start = time.time()
                data_output = model.predict(image[test_size*i+j])
                inference_time[test_size*i+j] = time.time()-start
    
        
        
        if infb4:
            inference_time_b4 = np.zeros(number_of_testes)
            image = np.random.rand(number_of_testes, 4,height_crop, width_crop,3)
            model.predict(np.ones((1,height_crop, width_crop,3)))
            model.predict(np.ones((1,height_crop, width_crop,3)))
            
            for i in range(test_divisor):
                for j in range(test_size):
                    start = time.time()
                    data_output = model.predict(image[test_size*i+j])
                    inference_time_b4[test_size*i+j] = (time.time()-start)/4
        
    else: #If not CMSNet
        print(test_case['name'])
        [model, height_image, width_image] = load_model(test_case['model'])
        
        # tf.config.set_visible_devices([], 'GPU')
        image = np.random.rand(number_of_testes,height_image, width_image,3)
        inference_time = np.zeros(number_of_testes)
        for i in range(test_divisor):
            for j in range(test_size):
                # start = time.time()
                mask, inftime = model.run_np(image[test_size*i+j])
                # inference_time[test_size*i+j] = time.time()-start
                inference_time[test_size*i+j] = inftime
        
        

    

    ###############TBR###############################
    #     mean_inference_time_b4 = (inference_time_b4).mean()*1000
    #     inference_time_b4_std = (inference_time_b4).std()*1000
        
    
    #     print("=======================================================================")
    #     print(str(output_stride)+',  '+ pooling+', r:'+ str(residual_shortcut) +', p:' +str(model.count_params()))
    #     print("-----------------------------------------------------------------------")
    #     print("m iftm b1 | std iftm b1 | m fps b1 | m iftm b4 | std iftm b4 | m fps b4")
    #     print(str(round(mean_inference_time_b1*100)/100)+' ms &  '+str(round(inference_time_b1_std*100)/100)+' ms &     '
    #           +str(round((1000/mean_inference_time_b1)*100)/100)+' &    '+str(round(mean_inference_time_b4*100)/100)+' ms &    '
    #           +str(round(inference_time_b4_std*100)/100)+' ms &  '+str(round((1000/mean_inference_time_b4)*100)/100)
    #           )
    #     print("=======================================================================")
    # else:
    #     print("=======================================================================")
    #     print(str(output_stride)+',  '+ pooling+', r:'+ str(residual_shortcut) +', p:' +str(model.count_params()))
    #     print("-----------------------------------------------------------------------")
    #     print("m iftm b1 | std iftm b1 | m fps b1 ")
    #     print(str(round(mean_inference_time_b1*100)/100)+' ms &  '+str(round(inference_time_b1_std*100)/100)+' ms &     '
    #           +str(round((1000/mean_inference_time_b1)*100)/100)
    #           )
    #     print("=======================================================================")
    #     mean_inference_time_b4 = 0
    #     inference_time_b4_std = 0
    # return mean_inference_time_b1, inference_time_b1_std, mean_inference_time_b4, inference_time_b4_std
    ###############TBR###############################
    return inference_time, inference_time_b4

def generate_latex_boxplot(test_cases, offset=0, print_label=True):
    i=1+offset
    for result in test_cases:
        result_fps = 1/np.array(result['inftime'])
        mean = result_fps.mean()
        std = result_fps.std()
        q1, median, q3 = np.percentile(result_fps, [25, 50, 75])
        iqr = q3 - q1
        mi = q1 - 1.5 * iqr
        ma = q3 + 1.5 * iqr
        outliers = np.compress(
            np.bitwise_or(result_fps >= ma,  result_fps <= mi), result_fps)
    
        print("\\addplot+[")
        print(" boxplot prepared={draw position="+str(i)+",")
        print(" 	lower whisker="+str(mi)+",")
        print(" 	lower quartile="+str(q1)+",")
        print(" 	median="+str(median)+",")
        print(" 	average="+str(mean)+",")
        print(" 	upper quartile="+str(q3)+",")
        print(" 	upper whisker="+str(ma)+",")
        print(" 	%	box extend=\boxwidth,")
        # print(" 	%	std="+str(std)+",")
        print(" },")
        if outliers.any():
            print(" ]coordinates "+str({ (0,value) for value in outliers}).
                  replace("), (", ") (")+";")
        else:
            print(" ]coordinates {};")
        if print_label:
            print("\\addlegendentry{"+result['abbr']+"\;};")
        i +=1
        
        
def generate_latex_mean_std(test_cases):
    for result in test_cases:
        result_fps = 1/np.array(result['inftime'])
        mean = result_fps.mean()
        std = result_fps.std()
    
        print(result['abbr']+" - b1 mean & std %")
        print(str(round(100*mean)/100)+" & "+ str(round(10000*std/mean)/100) + "\%")
        result_fps = 1/np.array(result['inftime_b4'])
        mean = result_fps.mean()
        std = result_fps.std()
        print("b4 mean & std %")
        print(str(round(100*mean)/100)+" & "+ str(round(10000*std/mean)/100) + "\%")
  




#General case
ntests = 100
test_divisor=1
sleep_time = 0
plot_resolution = False
infb4 = True
test_cases = [
    {'abbr' : 'CM0',   'name' : 'CMSNet-M0','model': '','pool' : 'global', 'os' : 8, 
      'residual':False, 'ch' : 3, 'cls' : 10, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 483, 'width'  : 769},
    {'abbr' : 'CM1',   'name' : 'CMSNet-M1','model': '','pool' : 'spp', 'os' : 8, 
      'residual':False, 'ch' : 3, 'cls' : 10, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 483, 'width'  : 769},
    {'abbr' : 'CM2',   'name' : 'CMSNet-M2','model': '','pool' : 'aspp', 'os' : 8, 
      'residual':False, 'ch' : 3, 'cls' : 10, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 483, 'width'  : 769},
    {'abbr' : 'CM3',   'name' : 'CMSNet-M3','model': '','pool' : 'global', 'os' : 16, 
      'residual':False, 'ch' : 3, 'cls' : 10, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 483, 'width'  : 769},
    {'abbr' : 'CM4',   'name' : 'CMSNet-M4','model': '','pool' : 'spp', 'os' : 16, 
      'residual':False, 'ch' : 3, 'cls' : 10, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 483, 'width'  : 769},
    {'abbr' : 'CM5',   'name' : 'CMSNet-M5','model': '','pool' : 'aspp', 'os' : 16, 
      'residual':False, 'ch' : 3, 'cls' : 10, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 483, 'width'  : 769},
    {'abbr' : 'CM6',   'name' : 'CMSNet-M6','model': '','pool' : 'global', 'os' : 16, 
      'residual':True, 'ch' : 3, 'cls' : 10, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 483, 'width'  : 769},
    {'abbr' : 'CM7',   'name' : 'CMSNet-M7','model': '','pool' : 'spp', 'os' : 16, 
      'residual':True, 'ch' : 3, 'cls' : 10, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 483, 'width'  : 769},
    {'abbr' : 'CM8',   'name' : 'CMSNet-M8','model': '','pool' : 'aspp', 'os' : 16, 
      'residual':True, 'ch' : 3, 'cls' : 10, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 483, 'width'  : 769},
    {'abbr' : 'PSP',   'name' : 'PSPNet','ntests' : ntests,       
     'model': 'pspnet101_cityscape'},
    {'abbr' : 'DLabM', 'name' : 'DeepLab-MNV2','ntests' : ntests,
     'model': 'mnv2_cityscapes_train'},
    {'abbr' : 'DLX6',  'name' : 'DeepLab-XC65','ntests' : ntests,
      'model': 'xception65_cityscapes_trainfine'},
    {'abbr' : 'DLX7',  'name' : 'DeepLab-XC71', 'ntests' : ntests,
     'model': 'xception71_dpc_cityscapes_trainfine'},
    ]




#General case
ntests = 100
test_divisor=1
sleep_time = 0
plot_resolution = False
infb4 = True
test_cases = [
    {'abbr' : 'CM0-300',   'name' : 'CMSNet-M0','model': '','pool' : 'global', 'os' : 8, 
        'residual':False, 'ch' : 3, 'cls' : 5, 'ntests' : ntests, 'mltgpu':False, 
        'height' : 300, 'width'  : 300},
    {'abbr' : 'CM0-448',   'name' : 'CMSNet-M0','model': '','pool' : 'global', 'os' : 8, 
        'residual':False, 'ch' : 3, 'cls' : 5, 'ntests' : ntests, 'mltgpu':False, 
        'height' : 448, 'width'  : 448},
    {'abbr' : 'CM3-300',   'name' : 'CMSNet-M3','model': '','pool' : 'global', 'os' : 16, 
      'residual':False, 'ch' : 3, 'cls' : 5, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 300, 'width'  : 300},
    {'abbr' : 'CM3-448',   'name' : 'CMSNet-M3','model': '','pool' : 'global', 'os' : 16, 
      'residual':False, 'ch' : 3, 'cls' : 5, 'ntests' : ntests, 'mltgpu':False, 
      'height' : 448, 'width'  : 448},
    {'abbr' : 'upnet-300',  'name' : 'upnet','ntests' : ntests, 'model': '','mltgpu':False, 
      'height' : 300, 'width'  : 300,'pool' : '', 'os' : '', 
        'residual':'', 'ch' : 3, 'cls' :  5},
    {'abbr' : 'cnns_fcn-227',   'name' : 'cnns_fcn','ntests' : ntests, 'model': '','mltgpu':False, 
     'height' : 227, 'width'  : 227, 'pool' : '', 'os' : '', 
      'residual':'', 'ch' : 3, 'cls' :  5},
    {'abbr' : 'darkfcn-448', 'name' : 'darkfcn','ntests' : ntests,'model': '','mltgpu':False, 
      'height' : 448, 'width'  : 448,'pool' : '', 'os' : '', 
        'residual':'', 'ch' : 3, 'cls' :  5},
    ]



recalcalculate = True
if recalcalculate==True:
    for  test_case, i in zip(test_cases, range(len(test_cases))):
        time.sleep(sleep_time)
        print("==================================")
        print(test_case['name'])
        inftime, inftime_b4 = inference_test(test_case=test_case, 
                                              test_divisor=test_divisor, 
                                              infb4=infb4)
        test_cases[i]['inftime']    = inftime.tolist()
        test_cases[i]['inftime_b4'] = inftime_b4.tolist()
        print("==================================")
        tf.keras.backend.clear_session()
    
    with open(build_path+'test_cases3.txt', 'w') as file:
        file.write(json.dumps(test_cases))




with open(build_path+'test_cases3.txt', 'r') as file:
    test_cases = json.loads(file.read())






# plt.hist(loaded_results[4]['inftime_b4'])

x = np.array([result['inftime']  for result in test_cases])#, showfliers=False
y = [1/value for value in x]


meanlineprops = dict(marker='.', markeredgecolor='purple',
                      markerfacecolor='firebrick')
flierprops = dict(marker='+', markeredgecolor='black',
                      markerfacecolor='firebrick')
plt.rc('text', usetex=True)
plt.boxplot(y, vert=True, showfliers=True, showmeans=True, 
            meanprops=meanlineprops,flierprops=flierprops)
#plt.savefig(build_path+'boxplot.pdf')  

generate_latex_boxplot(test_cases, offset=16, print_label=False)
generate_latex_mean_std(test_cases)

# # Tweak spacing to prevent clipping of ylabel
# plt.subplots_adjust(left=0.15)
# plt.show()

# # First quartile (Q1) 
# Q1 = np.percentile(x, 25, interpolation = 'midpoint') 
# # First quartile (Q1) 
# Q2 = np.percentile(x, 50, interpolation = 'midpoint') 
# # Third quartile (Q3) 
# Q3 = np.percentile(x, 75, interpolation = 'midpoint') 
  
# # Interquaritle range (IQR) 
# IQR = Q3 - Q1 












##################TBR##############
# if plot_resolution :
#     r = np.zeros((len(test_cases), 4))
#     for t, i in zip(test_cases, range(len(test_cases))):
    
#         # print(str(t['os'])+',  '+ t['pool']+', r:'+ str(t['residual']))
#         time.sleep(sleep_time)
#         r[i] = inference_test(height_crop=t['height'], width_crop=t['width'], channels=t['ch'],
#          output_stride=t['os'], pooling=t['pool'], n_classes=t['cls'], residual_shortcut=t['residual'],
#          number_of_testes=t['ntests'], mltgpu=t['residual'], test_divisor=test_divisor, infb4=infb4)

#     print("=======================================================================")
#     print("\\begin{tabularx}{\\textwidth}{r|X|X|X|X|X|X|X|X|X}")
#     print("\\hline")
#     print("Rede &  \\multicolumn{3}{c|}{227\\times227} &  \\multicolumn{3}{c|}{300\\times300} &  \\multicolumn{3}{c}{448\\times448} \\\\")
#     print("\\cline{2-10}")
#     print("        & $mTI$\spi{2} (ms) & ${\\sigma}TI$\spi{3} (\\%) & $FPS$\spi{4} & $mTI$ (ms) & ${\\sigma}TI$ (\\%) & $FPS$  & $mTI$ (ms) & ${\\sigma}TI$ (\\%) & $FPS$\\\\")
#     print("\\hline")
#     cases = 3
#     stride = int(len(test_cases)/cases)
#     for t, i in zip(test_cases, range(stride)):
#         print('CMSNet-S' +str(t['os'])+'-'+t['pool'][0].upper()+ ('-R' if t['residual'] else '  ')+'\t& '
#               +str(round(r[i,0]*100)/100).replace('.', ',')+' & '+str(round((r[i,1]/r[i,0])*10000)/100).replace('.', ',')+' & '
#               +str(round((1000/r[i,0])*100)/100).replace('.', ',')+' & '

#               +str(round(r[i+stride,0]*100)/100).replace('.', ',')+' & '+str(round((r[i+stride,1]/r[i+stride,0])*10000)/100).replace('.', ',')+' & '
#               +str(round((1000/r[i+stride,0])*100)/100).replace('.', ',')+' & '

#               +str(round(r[i+stride*2,0]*100)/100).replace('.', ',')+' & '+str(round((r[i+stride*2,1]/r[i+stride*2,0])*10000)/100).replace('.', ',')+' & '
#               +str(round((1000/r[i+stride*2,0])*100)/100).replace('.', ',')
#               +' \\\\'
#               )
#     print("\\hline")
#     print("\\end{tabularx}")
#     print("\\footnotesize{\\textit{1}--Resolução da imagem, \\textit{2}--Tempo médio para a realização de uma inferência, \\textit{3}--Desvio padrão do tempo de realização de inferência, \\textit{4}--\\textit{Frames} por segundo}")
#     print("=======================================================================")

# else:
#     r = np.zeros((len(test_cases), 4))
#     for t, i in zip(test_cases, range(len(test_cases))):
    
#         # print(str(t['os'])+',  '+ t['pool']+', r:'+ str(t['residual']))
#         time.sleep(sleep_time)
#         r[i] = inference_test(height_crop=t['height'], width_crop=t['width'], channels=t['ch'],
#          output_stride=t['os'], pooling=t['pool'], n_classes=t['cls'], residual_shortcut=t['residual'],
#          number_of_testes=t['ntests'], mltgpu=t['residual'], test_divisor=test_divisor)

#     print("=======================================================================")
#     print("\\begin{tabularx}{\\textwidth}{c|X|c|c|c|c|c|c|c}")
#     print("\\hline")
#     print("OS\spi{1}  & Pirâmide  & Resíduo &  \\multicolumn{3}{c|}{\\textit{Batch} de tamanho 1} &  \\multicolumn{3}{c}{\\textit{Batch} de tamanho 4} \\\\")
#     print("\\cline{4-9}")
#     print("           &   &  & $mTI$\spi{2} & ${\\sigma}TI$\spi{3} & $FPS$\spi{4} & $mTI$ & ${\\sigma}TI$ & $FPS$ \\\\")
#     print("\\hline")
#     for t, i in zip(test_cases, range(len(test_cases))):
#         print(str(t['os'])+'\t& '+('Global' if t['pool']=='global'else t['pool'].upper()+'\t') +'& '+ ('Sim' if t['residual'] else '  ')+'\t& '
#               +str(round(r[i,0]*100)/100).replace('.', ',')+' ms & '+str(round(r[i,1]*100)/100).replace('.', ',')+' ms & '
#               +str(round((1000/r[i,0])*100)/100).replace('.', ',')+' & '+str(round(r[i,2]*100)/100).replace('.', ',')+' ms & '
#               +str(round(r[i,3]*100)/100).replace('.', ',')+' ms & '+str(round((1000/r[i,2])*100)/100).replace('.', ',')+' \\\\'
#               )
#     print("\\hline")
#     print("\\end{tabularx}")
#     print("\\footnotesize{\\textit{1}--\\textit{Output Stride} (Fator de redução da resolução das \\textit{features} na saída do \\textit{backbone}), \\textit{2}--Tempo médio para a realização de uma inferência, \\textit{3}--Desvio padrão do tempo de realização de inferência, \\textit{4}--\\textit{Frames} por segundo}")
#     print("=======================================================================")

