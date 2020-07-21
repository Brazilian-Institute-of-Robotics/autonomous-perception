

import os
import sys
import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

sys.path.append(os.environ['PSPNET_HOME'])

from model import PSPNet101, PSPNet50
from tools import *


#Set up paths for Images and Weights
# TODO: Change these values to where your model files are
ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150, 
                'model': PSPNet50,
                'weights_path': os.environ['PSPNET_HOME']+'/model/pspnet50-ade20k/model.ckpt-0'}
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 19,
                    'model': PSPNet101,
                    'weights_path': os.environ['PSPNET_HOME']+'/model/pspnet101-cityscapes/model.ckpt-0'}

IMAGE_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
image_path = os.environ['PSPNET_HOME']+'/input/test1.png'

# TODO: If you want to inference on indoor data, change this value to `ADE20k_param`
param = cityscapes_param

# pre-proecess image
img_np, filename = load_img(image_path)
img_shape = tf.shape(img_np)
h, w = (tf.maximum(param['crop_size'][0], img_shape[0]), tf.maximum(param['crop_size'][1], img_shape[1]))
#img = preprocess(img_np, h, w)

input_image = tf.placeholder(dtype=tf.float32, shape=(None,None,3),name='input_image')
input_h = tf.placeholder(dtype=tf.int32,name='input_h')
input_w = tf.placeholder(dtype=tf.int32,name='input_w')

img = preprocess(input_image, input_h, input_w)

# Create network.
PSPNet = param['model']
#net = PSPNet({'data': img}, is_training=False, num_classes=param['num_classes'])
net = PSPNet({'data': img}, is_training=False, num_classes=param['num_classes'])


raw_output = net.layers['conv6']

# Predictions.
raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
raw_output_up = tf.argmax(raw_output_up, dimension=3)
pred = decode_labels(raw_output_up, img_shape, param['num_classes'])

# Init tf Session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

sess.run(init)

ckpt_path = param['weights_path']
loader = tf.train.Saver(var_list=tf.global_variables())
loader.restore(sess, ckpt_path)
print("Restored model parameters from {}".format(ckpt_path))
    
# Run and get result image
#preds = sess.run(pred)
img_np = img_np.eval(session=sess)
h = h.eval(session=sess)
w = w.eval(session=sess)
preds = sess.run('ArgMax:0', feed_dict={'input_image:0': img_np,
                                           'input_h:0': h,
                                           'input_w:0': w})


plt.figure(1, [15, 30])
plt.subplot(121)
plt.imshow(img_np)
plt.axis('off')
plt.subplot(122)
plt.imshow(preds[0])
plt.axis('off')
plt.show()

logdir = '../../build/log'
graph_def = sess.graph.as_graph_def()
constant_graph =  graph_util.convert_variables_to_constants(sess, graph_def, ['ArgMax'])
inference_graph = graph_util.remove_training_nodes(constant_graph)
graph_io.write_graph(inference_graph, logdir, "../../data/psp_cityscape/model.pb", as_text=False)




sess.close()
tf.keras.backend.clear_session()

    
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(inference_graph, name='')
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True
sess_inf = tf.Session(graph=graph, config=config)

tensor_mask = sess_inf.run('ArgMax:0', feed_dict={'input_image:0': img_np,
                                           'input_h:0': h,
                                           'input_w:0': w})
plt.imshow(tensor_mask[0])
plt.show()
