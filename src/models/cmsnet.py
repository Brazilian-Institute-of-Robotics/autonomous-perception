# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AvgPool2D, Conv2D, BatchNormalization, ReLU, Lambda, Add
from tensorflow.keras.layers import DepthwiseConv2D, Concatenate, Softmax, Dropout, ZeroPadding2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet101, ResNet50
from tensorflow.keras.regularizers import l2
import re
import numpy as np
import logging


class CMSNet(Model):
    """ """
    def __init__(self, num_classes=19, output_stride=16, backbonetype='mobilenetv2',
                 weights='imagenet',  dl_input_shape=(None, 483,769,3), weight_decay=0.00004,
                 pooling='global',  residual_shortcut=False):
        super(CMSNet, self).__init__(name='cmsnet')
        """
        :param num_classes:  (Default value = 19)
        :param output_stride:  (Default value = 16) 
            if strid count is 4 remove stride from block 13 and inser atrous in 14, 15 and 16
            if strid count is 3 remove stride from block 6/13 and inser atrous rate 2 in 7-13/ and rate 4 14-16
        :param backbonetype:  (Default value = 'mobilenetv2')
        :param weights:  (Default value = 'imagenet')
        :param input_shape:  (Default value = (None, 483,769,3)
        :param weight_decay: use 0.00004 for MobileNet-V2 or Xcpetion model backbonetype. Use 0.0001 for ResNet backbonetype.

        """
        self.logger = logging.getLogger('perception.models.CMSNet')
        self.logger.info('creating an instance of CMSNet with backbone '+backbonetype
                         +', OS'+str(output_stride)+', nclass='+str(num_classes)
                         +', input='+str(dl_input_shape)+', pooling='+pooling
                         +', residual='+str(residual_shortcut))

        self.num_classes = num_classes
        self.output_stride = output_stride
        self.dl_input_shape = dl_input_shape
        self._createBackbone(backbonetype=backbonetype, output_stride=output_stride)
        # All with 256 filters and batch normalization.
        # one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # Rates are doubled when output stride = 8.

        #Create Spatial Pyramid Pooling
        x = self.backbone.output

        pooling_shape = self.backbone.compute_output_shape(self.dl_input_shape)
        pooling_shape_float = tf.cast(pooling_shape[1:3], tf.float32)

        assert pooling in ['aspp', 'spp', 'global'], "Only suported pooling= 'aspp', 'spp' or 'global'."


        if pooling == 'aspp':
            if output_stride==16:
                rates = (6, 12, 18)
            elif output_stride==8:
                rates = (12, 24, 36)
            #gride lavel: pooling
            x0 = Conv2D(filters=256, kernel_size=3, name='aspp_0_expand', padding="same",
                        dilation_rate=rates[0], kernel_regularizer=l2(weight_decay))(x)
            x0 = BatchNormalization(name='aspp_0_expand_BN')(x0)#epsilon=1e-5
            x0 = ReLU(name='aspp_0_expand_relu')(x0)

            x1 = Conv2D(filters=256, kernel_size=3, name='aspp_1_expand', padding="same",
                        dilation_rate=rates[1], kernel_regularizer=l2(weight_decay))(x)
            x1 = BatchNormalization(name='aspp_1_expand_BN')(x1)#epsilon=1e-5
            x1 = ReLU(name='aspp_1_expand_relu')(x1)

            x2 = Conv2D(filters=256, kernel_size=3, name='aspp_2_expand', padding="same",
                        dilation_rate=rates[2], kernel_regularizer=l2(weight_decay))(x)
            x2 = BatchNormalization(name='aspp_2_expand_BN')(x2)#epsilon=1e-5
            x2 = ReLU(name='aspp_2_expand_relu')(x2)

            #gride lavel: all
            xn = Conv2D(filters=256, kernel_size=1, name='aspp_n_expand', kernel_regularizer=l2(weight_decay))(x)
            xn = BatchNormalization(name='aspp_n_expand_BN')(xn)#epsilon=1e-5
            xn = ReLU(name='aspp_n_expand_relu')(xn)

            #Concatenate spatial pyramid pooling
            x0.set_shape(pooling_shape[0:3].concatenate(x0.get_shape()[-1]))
            x1.set_shape(pooling_shape[0:3].concatenate(x1.get_shape()[-1]))
            x2.set_shape(pooling_shape[0:3].concatenate(x2.get_shape()[-1]))
            xn.set_shape(pooling_shape[0:3].concatenate(xn.get_shape()[-1]))
            x = Concatenate(name='aspp_concatenate')([x0, x1, x2, xn])

        elif pooling == 'spp':
            rates = (1, 2, 3, 6)
            #gride lavel: pooling
            x0 = AvgPool2D(pool_size=tf.cast(pooling_shape_float/rates[0], tf.int32),
                           padding="valid", name='spp_0_average_pooling2d')(x)
            x0 = Conv2D(filters=int(pooling_shape[-1]/len(rates)), kernel_size=1,
                        name='spp_0_expand', kernel_regularizer=l2(weight_decay))(x0)
            x0 = BatchNormalization(name='spp_0_expand_BN')(x0)#epsilon=1e-5
            x0 = ReLU(name='spp_0_expand_relu')(x0)
            if tf.__version__.split('.')[0]=='1':
                x0 = Lambda(lambda x0: tf.image.resize_bilinear(x0,pooling_shape[1:3],
                            align_corners=True),
                            name='spp_0_resize_bilinear')(x0)
            else:
                x0 = Lambda(lambda x0: tf.image.resize(x0,pooling_shape[1:3],
                            method=tf.image.ResizeMethod.BILINEAR),
                            name='spp_0_resize_bilinear')(x0)

            x1 = AvgPool2D(pool_size=tf.cast(pooling_shape_float/rates[1], tf.int32),
                           padding="valid", name='spp_1_average_pooling2d')(x)
            x1 = Conv2D(filters=int(pooling_shape[-1]/len(rates)), kernel_size=1,
                        name='spp_1_expand', kernel_regularizer=l2(weight_decay))(x1)
            x1 = BatchNormalization(name='spp_1_expand_BN')(x1)#epsilon=1e-5
            x1 = ReLU(name='spp_1_expand_relu')(x1)
            if tf.__version__.split('.')[0]=='1':
                x1 = Lambda(lambda x1: tf.image.resize_bilinear(x1,pooling_shape[1:3],
                            align_corners=True),
                            name='spp_1_resize_bilinear')(x1)
            else:
                x1 = Lambda(lambda x1: tf.image.resize(x1,pooling_shape[1:3],
                            method=tf.image.ResizeMethod.BILINEAR),
                            name='spp_1_resize_bilinear')(x1)



            x2 = AvgPool2D(pool_size=tf.cast(pooling_shape_float/rates[2], tf.int32),
                           padding="valid", name='spp_2_average_pooling2d')(x)
            x2 = Conv2D(filters=int(pooling_shape[-1]/len(rates)), kernel_size=1,
                        name='spp_2_expand', kernel_regularizer=l2(weight_decay))(x2)
            x2 = BatchNormalization(name='spp_2_expand_BN')(x2)#epsilon=1e-5
            x2 = ReLU(name='spp_2_expand_relu')(x2)
            if tf.__version__.split('.')[0]=='1':
                x2 = Lambda(lambda x2: tf.image.resize_bilinear(x2,pooling_shape[1:3],
                            align_corners=True),
                            name='spp_2_resize_bilinear')(x2)
            else:
                x2 = Lambda(lambda x2: tf.image.resize(x2,pooling_shape[1:3],
                            method=tf.image.ResizeMethod.BILINEAR),
                            name='spp_2_resize_bilinear')(x2)


            x3 = AvgPool2D(pool_size=tf.cast(pooling_shape_float/rates[3], tf.int32),
                           padding="valid", name='spp_3_average_pooling2d')(x)
            x3 = Conv2D(filters=int(pooling_shape[-1]/len(rates)), kernel_size=1,
                        name='spp_3_expand', kernel_regularizer=l2(weight_decay))(x3)
            x3 = BatchNormalization(name='spp_3_expand_BN')(x3)#epsilon=1e-5
            x3 = ReLU(name='spp_3_expand_relu')(x3)
            if tf.__version__.split('.')[0]=='1':
                x3 = Lambda(lambda x3: tf.image.resize_bilinear(x3,pooling_shape[1:3],
                            align_corners=True),
                            name='spp_3_resize_bilinear')(x3)
            else:
                x3 = Lambda(lambda x3: tf.image.resize(x3,pooling_shape[1:3],
                            method=tf.image.ResizeMethod.BILINEAR),
                            name='spp_3_resize_bilinear')(x3)

            #gride lavel: all
            xn = Conv2D(filters=int(pooling_shape[-1]/len(rates)), kernel_size=1,
                        name='spp_n_expand', kernel_regularizer=l2(weight_decay))(x)
            xn = BatchNormalization(name='spp_n_expand_BN')(xn)#epsilon=1e-5
            xn = ReLU(name='spp_n_expand_relu')(xn)
            #Concatenate spatial pyramid pooling
            xn.set_shape(pooling_shape[0:3].concatenate(xn.get_shape()[-1]))
            x = Concatenate(name='spp_concatenate')([x0, x1, x2, xn])

        elif pooling == 'global':
        #gride lavel: pooling
            x0 = AvgPool2D(pool_size=pooling_shape[1:3], padding="valid",
                name='spp_0_average_pooling2d')(x)
            x0 = Conv2D(filters=256, kernel_size=1, name='spp_0_expand', kernel_regularizer=l2(weight_decay))(x0)
            x0 = BatchNormalization(name='spp_0_expand_BN')(x0)#epsilon=1e-5
            x0 = ReLU(name='spp_0_expand_relu')(x0)
    #        x0 = tf.image.resize(x0, 
    #            size=pooling_shape[1:3], 
    #            method=tf.image.ResizeMethod.BILINEAR, name='spp_0_resize_bilinear')
            if tf.__version__.split('.')[0]=='1':
                x0 = Lambda(lambda x0: tf.image.resize_bilinear(x0,pooling_shape[1:3],
                            align_corners=True),
                            name='spp_0_resize_bilinear')(x0)
            else:
                x0 = Lambda(lambda x0: tf.image.resize(x0,pooling_shape[1:3],
                            method=tf.image.ResizeMethod.BILINEAR),
                            name='spp_0_resize_bilinear')(x0)
            
            #gride lavel: all
            xn = Conv2D(filters=256, kernel_size=1, name='spp_1_expand', kernel_regularizer=l2(weight_decay))(x)
            xn = BatchNormalization(name='spp_1_expand_BN')(xn)#epsilon=1e-5
            xn = ReLU(name='spp_1_expand_relu')(xn)
            #Concatenate spatial pyramid pooling
            xn.set_shape(pooling_shape[0:3].concatenate(xn.get_shape()[-1]))
            x = Concatenate(name='spp_concatenate')([x0, xn])




        
        #Concate Projection
        x = Conv2D(filters=256, kernel_size=1, name='spp_concat_project', kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(name='spp_concat_project_BN')(x)#epsilon=1e-5
        x = ReLU(name='spp_concat_project_relu')(x)

        if residual_shortcut:
            assert output_stride==16, "For while residual shotcut is available for atous with os16."

           #self.strideOutput8LayerName #block_6_project_BN (BatchNormal (None, 61, 97, 64)
            os8_shape = self.backbone.get_layer(self.strideOutput8LayerName).output_shape
            os8_output = self.backbone.get_layer(self.strideOutput8LayerName).output

            x = Conv2D(filters=os8_shape[-1], kernel_size=1, name='shotcut_2x_conv',
                       kernel_regularizer=l2(weight_decay))(x)
            x = BatchNormalization(name='shotcut_2x_BN')(x)#epsilon=1e-5
            if tf.__version__.split('.')[0]=='1':
                x = Lambda(lambda x: tf.image.resize_bilinear(x, os8_shape[1:3], align_corners=True),
                            name='shotcut_2x_bilinear')(x)
            else:
                x = Lambda(lambda x: tf.image.resize(x, os8_shape[1:3], method=tf.image.ResizeMethod.BILINEAR),
                            name='shotcut_2x_bilinear')(x)
            x = ReLU(name='shotcut_2x_relu')(x)
            x = Add(name='shotcut_2x_add')([x, os8_output])

        x = Dropout(rate=0.1, name='dropout')(x)

        #Semantic Segmentation
        x = Conv2D(filters=num_classes, kernel_size=1, name='segmentation', kernel_regularizer=l2(weight_decay))(x)
        #x = BatchNormalization(name='segmentation_BN')(x)
#        x = tf.image.resize(x, size=self.dl_input_shape[1:3], 
#                method=tf.image.ResizeMethod.BILINEAR, name='segmentation_bilinear')
        if tf.__version__.split('.')[0]=='1':
            x = Lambda(lambda x: tf.image.resize_bilinear(x,self.dl_input_shape[1:3],
                        align_corners=True), 
                        name='segmentation_bilinear')(x)
        else:
            x = Lambda(lambda x: tf.image.resize(x,self.dl_input_shape[1:3],
                        method=tf.image.ResizeMethod.BILINEAR), 
                        name='segmentation_bilinear')(x)
        x = Softmax(name='logistic_softmax')(x)
        #logist to training
        #argmax
        super(CMSNet, self).__init__(inputs=self.backbone.input, outputs=x,name='cmsnet')


    def mySummary(self, input_shape=()):
        """ """
        if input_shape==():
            input_shape = self.dl_input_shape
        
        notFirstLayer = False
        for layer in super().layers:
            if layer.name.split('_')[-1]=='depthwise':
                print(layer.name + '\t('+ str(Model(inputs=super().input, outputs=layer.output).compute_output_shape(input_shape).as_list())+') '+ ', p='+ layer.padding+', s='+ str(layer.strides) +', d=' + str(layer.dilation_rate))
            else:
                if notFirstLayer:
                    print(layer.name + '\t('+ str(Model(inputs=super().input, outputs=layer.output).compute_output_shape(input_shape).as_list())+') ')
                else:
                    notFirstLayer = True


    def _createBackbone(self, backbonetype='mobilenetv2', output_stride=16):
        if backbonetype=='mobilenetv2':
            self.logger.info("Creating backbone mobilenetv2.")
            self._createMobilenetv2Backbone(output_stride=output_stride)
        elif backbonetype=='vgg16':
            self.logger.info("Creating backbone vgg16.")
            self._createVGG16Backbone(output_stride=output_stride)
        elif backbonetype=='resnet101' or backbonetype=='resnet50':
            self.logger.info("Creating backbone vgg16.")
            self._createResnetBackbone(output_stride=output_stride, depth=backbonetype.split('resnet')[-1])
        else:
            self.logger.error("Only backbonetype 'mobilenetv2', 'resnet101'', 'resnet50' and 'vgg16' accept until now.")
            raise Exception("Only backbonetype 'mobilenetv2' and 'vgg16' accept until now.")


    def _createMobilenetv2Backbone(self, output_stride=16):
        mobilenetv2 = MobileNetV2(weights='imagenet',  input_shape=(self.dl_input_shape[1], self.dl_input_shape[2], 3), include_top=False)

        mobile_config = mobilenetv2.get_config()
        mobile_weights = mobilenetv2.get_weights()

        #tf.keras.backend.clear_session()
        
        assert output_stride in [8, 16], "Only suported output_stride= 8 o 16 for backbone mobilenetv2."
        
        dilatation = 1
        for layer in mobile_config['layers']:
            if layer['name'] == 'input_1':
                layer['config']['batch_input_shape'] = (None, self.dl_input_shape[-3], self.dl_input_shape[-2], self.dl_input_shape[-1])
                self.logger.info(layer['name']+', '+str(layer['config']['batch_input_shape']))

            if output_stride == 8 :
                if layer['name'] == 'block_6_depthwise':
                    layer['config']['strides'] = (1, 1)
                    dilatation = dilatation*2 #Replace stride for dilatation 
                    self.logger.info(layer['name']+', strides='+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                if layer['name'] == 'block_7_depthwise':
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                if layer['name'] == 'block_8_depthwise':
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                if layer['name'] == 'block_9_depthwise':
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                if layer['name'] == 'block_10_depthwise':
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                if layer['name'] == 'block_11_depthwise':
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                if layer['name'] == 'block_12_depthwise':
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
            if output_stride in [8, 16] :
                if layer['name'] == 'block_13_depthwise':
                    layer['config']['strides'] = (1, 1)
                    dilatation = dilatation*2 #Replace stride for dilatation 
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                if layer['name'] == 'block_14_depthwise':
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                if layer['name'] == 'block_15_depthwise':
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                if layer['name'] == 'block_16_depthwise':
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))


#        tf.keras.backend.clear_session()
        new_mobilenetv2 = Model.from_config(mobile_config)
        mobile_weights[0] = np.resize(mobile_weights[0], [3, 3, self.dl_input_shape[-1], 32])
        new_mobilenetv2.set_weights(mobile_weights)
        
        self.backbone = Model(inputs=new_mobilenetv2.inputs, outputs=new_mobilenetv2.get_layer('block_16_project_BN').output)
        self.strideOutput32LayerName = 'block_16_project_BN'
        self.strideOutput16LayerName = 'block_12_add'
        self.strideOutput8LayerName = 'block_5_add'
        self.inputLayerName = mobilenetv2.layers[0].name


    def _createVGG16Backbone(self, output_stride=16):
        vgg16 = VGG16(weights='imagenet',  input_shape=(self.dl_input_shape[1],
                                                        self.dl_input_shape[2],
                                                        3),  include_top=False)

        assert output_stride in [8, 16], "Only suported output_stride= 8 o 16 for backbone vgg16."

        mobile_config = vgg16.get_config()
        mobile_weights = vgg16.get_weights()
        #tf.keras.backend.clear_session()

        dilatation = 1
        stride_enable = False
        for layer in mobile_config['layers']:
            if layer['name'] == 'input_1':
                layer['config']['batch_input_shape'] = (None, self.dl_input_shape[-3],
                     self.dl_input_shape[-2], self.dl_input_shape[-1])
                self.logger.info(layer['name']+', '+str(layer['config']['batch_input_shape']))
            if output_stride == 8 :
                if layer['name'] == 'block4_pool':
                    layer['config']['strides'] = (1, 1)
                    layer['config']['pool_size'] = (1, 1)
                    dilatation = dilatation*2 #Replace stride for dilatation 
                    self.logger.info(layer['name']+', strides='+str(layer['config']['strides'])+
                                     ', '+str(layer['config']['pool_size']))
                    stride_enable = True

            if output_stride in [8, 16] :
                if layer['name'] == 'block5_pool':
                    layer['config']['strides'] = (1, 1)
                    layer['config']['pool_size'] = (1, 1)
                    dilatation = dilatation*2 #Replace stride for dilatation 
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+
                                     ', '+str(layer['config']['pool_size']))
                    stride_enable = True
                if 'conv' in layer['name'] and stride_enable:
                    layer['config']['dilation_rate'] = (dilatation, dilatation)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+
                                     ', '+str(layer['config']['dilation_rate']))
        
        
        new_vgg16 = Model.from_config(mobile_config)
        mobile_weights[0] = np.resize(mobile_weights[0], [mobile_weights[0].shape[0], mobile_weights[0].shape[1],
                                      self.dl_input_shape[-1], mobile_weights[0].shape[-1]]) #update input to suport 4 channels
        new_vgg16.set_weights(mobile_weights)
        
        
        x = new_vgg16.output
        x = Conv2D(filters=512, kernel_size=3, name='block_6_conv1', padding="same",
                   dilation_rate=dilatation, activation='relu')(x)
        x = Conv2D(filters=512, kernel_size=3, name='block_6_conv2', padding="same",
                   dilation_rate=dilatation, activation='relu')(x)
        x = Conv2D(filters=512, kernel_size=3, name='block_6_conv3', padding="same",
                   dilation_rate=dilatation, activation='relu')(x)
        
        
        
        self.backbone = Model(inputs=new_vgg16.inputs, outputs=x)
        self.strideOutput32LayerName = 'block6_conv3'
        self.strideOutput16LayerName = 'block5_conv3'
        self.strideOutput8LayerName = 'block4_conv3'
        self.inputLayerName = vgg16.layers[0].name




    def _createResnetBackbone(self, output_stride=16, depth='101'):
        resnet101 = ResNet101(weights='imagenet',  input_shape=(self.dl_input_shape[1],
                                                                self.dl_input_shape[2],
                                                                3), include_top=False)

        assert output_stride in [8, 16], "Only suported output_stride= 8 o 16 for backbone resnet."


        resnet101_config = resnet101.get_config()
        resnet101_weights = resnet101.get_weights()
        #tf.keras.backend.clear_session()

        
        output_stride = 8
        dilatation = 1
        stride_enable = False
        for layer in resnet101_config['layers']:
            if layer['name'] == 'input_1':
                layer['config']['batch_input_shape'] = (None, self.dl_input_shape[-3],
                     self.dl_input_shape[-2], self.dl_input_shape[-1])
                self.logger.info(layer['name']+', '+str(layer['config']['batch_input_shape']))
            if output_stride == 8 and (layer['name'] == 'conv4_block1_1_conv'  or
                                       layer['name'] == 'conv4_block1_0_conv'):
                layer['config']['strides'] = (1, 1)
                self.logger.info(layer['name']+', strides='+str(layer['config']['strides'])+
                            ', '+str(layer['config']['dilation_rate']))
                stride_enable = True
                if  layer['name'] == 'conv4_block1_1_conv':
                    dilatation = dilatation*2 #Replace stride for dilatation
        
            elif output_stride in [8, 16] :
                if layer['name'] == 'conv5_block1_1_conv' or layer['name'] == 'conv5_block1_0_conv':
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+
                                     ', '+str(layer['config']['dilation_rate']))
                    layer['config']['strides'] = (1, 1)
                    self.logger.info(layer['name']+', '+str(layer['config']['strides'])+
                                     ', '+str(layer['config']['dilation_rate']))
                    stride_enable = True
                    if layer['name'] == 'conv5_block1_1_conv':
                        dilatation = dilatation*2 #Replace stride for dilatation
                elif stride_enable and ('_conv' in layer['name']):
                    if layer['config']['kernel_size']!=(1,1):
                        layer['config']['dilation_rate'] = (dilatation, dilatation)
                        self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
                    else:
                        self.logger.info(layer['name']+', '+str(layer['config']['strides'])+', '+str(layer['config']['dilation_rate']))
        
        
        
        self.backbone = Model.from_config(resnet101_config)
        resnet101_weights[0] = np.resize(resnet101_weights[0], [resnet101_weights[0].shape[0],
                                      resnet101_weights[0].shape[1], self.dl_input_shape[-1],
                                      resnet101_weights[0].shape[-1]]) #update input to suport 4 channels
        self.backbone.set_weights(resnet101_weights)
        
        
        self.strideOutput32LayerName = 'conv5_block3_out'
        self.strideOutput16LayerName = 'conv4_block23_out'
        self.strideOutput8LayerName = 'conv3_block4_out'
        self.inputLayerName = resnet101.layers[0].name
        

    def setFineTuning(self, lavel='fromSPPool'):
        """
        :param lavel:  (Default value = 'fromSPPool')
        """
        
        assert lavel in ['fromAll', 'fromOS8', 'fromOS16', 'fromOS32'], "Lavel not suported."
        
        #Get layer name 
        trainable=False
        for layer in super().layers:
            if lavel=='fromAll' and layer.name==self.inputLayerName:
                trainable=True
            elif lavel=='fromOS8' and layer.name==self.strideOutput8LayerName:
                trainable=True
            elif lavel=='fromOS16' and layer.name==self.strideOutput16LayerName:
                trainable=True    
            elif lavel=='fromOS32' and layer.name==self.strideOutput32LayerName:
                trainable=True       
            # elif lavel=='fromSPPool' and layer.name=='spp_0_average_pooling2d':
            #     trainable=True
            
            layer.trainable = trainable
            print(layer.name + '\ttainable = '+ str(layer.trainable))
    
    
    def _make_list(self, X):
        if isinstance(X, list):
            return X
        return [X]
    
    def _list_no_list(self, X):
        if len(X) == 1:
            return X[0]
        return X
    
    def _replace_layer(self, model, name, function_fn, **kwargs):
        """
        args:
            model :: keras.models.Model instance
            replace_layer_subname :: str -- if str in layer name, replace it
            replacement_fn :: fn to call to replace all instances
                > fn output must produce shape as the replaced layers input
        returns:
            new model with replaced layers
        quick examples:
            want to just remove all layers with 'batch_norm' in the name:
                > new_model = replace_layer(model, 'batch_norm', lambda **kwargs : (lambda u:u))
            want to replace all Conv1D(N, m, padding='same') with an LSTM (lets say all have 'conv1d' in name)
                > new_model = replace_layer(model, 'conv1d', lambda layer, **kwargs: LSTM(units=layer.filters, return_sequences=True)
        """
        model_inputs = []
        model_outputs = []
        tsr_dict = {}
    
        model_output_names = [out.name for out in self._make_list(model.output)]
    
        for i, layer in enumerate(model.layers):
            ### Loop if layer is used multiple times
            for j in range(len(layer._inbound_nodes)):
    
                ### check layer inp/outp
                inpt_names = [inp.name for inp in self._make_list(layer.get_input_at(j))]
                outp_names = [out.name for out in self._make_list(layer.get_output_at(j))]
    
                ### setup model inputs
                if 'input' in layer.name:
                    for inpt_tsr in self._make_list(layer.get_output_at(j)):
                        model_inputs.append(inpt_tsr)
                        tsr_dict[inpt_tsr.name] = inpt_tsr
                    continue
    
                ### setup layer inputs
                inpt = self._list_no_list([tsr_dict[name] for name in inpt_names])
    
                ### remake layer 
                if name in layer.name:
                        print('replacing '+layer.name)
                        layer._inbound_nodes.pop(0)
                        x = function_fn(old_layer=layer, **kwargs)(inpt)
                else:
#                    layer._inbound_nodes.pop(0)
                    x = layer(inpt)
                    

                ### reinstantialize outputs into dict
                for name, out_tsr in zip(outp_names, self._make_list(x)):
    
                    ### check if is an output
                    if name in model_output_names:
                        model_outputs.append(out_tsr)
                    tsr_dict[name] = out_tsr
                    
                
#        while(len(model._layers)!=1):
#            model._layers.pop()
            
        return Model(model_inputs, model_outputs)


    def _replace_layers(self, model, layers_to_replace, **kwargs):
        """
        args:
            model :: keras.models.Model instance
            replace_layer_subname :: str -- if str in layer name, replace it
            replacement_fn :: fn to call to replace all instances
                > fn output must produce shape as the replaced layers input
        returns:
            new model with replaced layers
        quick examples:
            want to just remove all layers with 'batch_norm' in the name:
                > new_model = replace_layer(model, 'batch_norm', lambda **kwargs : (lambda u:u))
            want to replace all Conv1D(N, m, padding='same') with an LSTM (lets say all have 'conv1d' in name)
                > new_model = replace_layer(model, 'conv1d', lambda layer, **kwargs: LSTM(units=layer.filters, return_sequences=True)
        """
        model_inputs = []
        model_outputs = []
        tsr_dict = {}
    
        model_output_names = [out.name for out in self._make_list(model.output)]
    
        for i, layer in enumerate(model.layers):
            ### Loop if layer is used multiple times
            for j in range(len(layer._inbound_nodes)):
    
                ### check layer inp/outp
                inpt_names = [inp.name for inp in self._make_list(layer.get_input_at(j))]
                outp_names = [out.name for out in self._make_list(layer.get_output_at(j))]
    
                ### setup model inputs
                if 'input' in layer.name:
                    for inpt_tsr in self._make_list(layer.get_output_at(j)):
                        model_inputs.append(inpt_tsr)
                        tsr_dict[inpt_tsr.name] = inpt_tsr
                    continue
    
                ### setup layer inputs
                inpt = self._list_no_list([tsr_dict[name] for name in inpt_names])
    
                ### remake layer 
                layer_not_found = True
                for layer_to_replace in layers_to_replace:
                    if layer_to_replace[0] == layer.name:
                        print('replacing '+layer.name)
                        layer._inbound_nodes.pop(0)
                        x = layer_to_replace[1](old_layer=layer, **kwargs)(inpt)
                        layer_not_found = False
                if layer_not_found:
#                    layer._inbound_nodes.pop(0)
                    x = layer(inpt)
                    
    
                ### reinstantialize outputs into dict
                for name, out_tsr in zip(outp_names, self._make_list(x)):
    
                    ### check if is an output
                    if name in model_output_names:
                        model_outputs.append(out_tsr)
                    tsr_dict[name] = out_tsr
                    
                
#        while(len(model._layers)!=1):
#            model._layers.pop()
            
        return Model(model_inputs, model_outputs)

import math
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard


    
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


def step_decay(epoch, initial_lrate=0.007):
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

def polynomial_decay(epoch, initial_lrate = 0.007, learning_rate_decay_step=400, 
                     learning_power=0.9, end_learning_rate=0, cycle=False):
    global_step = epoch
#    if epoch < learning_rate_decay_step*.5:
#        decay_steps = learning_rate_decay_step*.7
#    else:
#        decay_steps = learning_rate_decay_step
    decay_steps = learning_rate_decay_step


    #If cycle is True then a multiple of decay_steps is used, the first one that is bigger than global_steps.
    if cycle:
        decay_steps = decay_steps * math.ceil(global_step / decay_steps)
    else:
        global_step = min(global_step, decay_steps)
        
    decayed_learning_rate = (initial_lrate - end_learning_rate) * (1 - global_step / decay_steps) ** (learning_power) + end_learning_rate

        
    return decayed_learning_rate


def exponential_decay (learning_rate, global_step, decay_steps, decay_rate):

    decayed_learning_rate = learning_rate * decay_rate ** (global_step / decay_steps)
    return decayed_learning_rate