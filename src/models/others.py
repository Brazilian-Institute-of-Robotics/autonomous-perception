# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AvgPool2D, MaxPool2D, Conv2D, BatchNormalization, Cropping2D, ReLU, Lambda, Add
from tensorflow.keras.layers import DepthwiseConv2D, Concatenate, Softmax, Dropout, ZeroPadding2D, Conv2DTranspose
from tensorflow.keras.layers import Layer, Input, ZeroPadding2D, UpSampling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import ResNet101, ResNet50
from tensorflow.keras.regularizers import l2
import re
import numpy as np
import logging



# name            type                               units    sz    st    inshape     outshape    inputs
# --------------  ---------------------------------  -------  ----  ----  ----------  ----------  -------------------
# input           InputLayer                                                          3x300x300
# conv1           Conv2DDNNLayer                     96       7     2     3x300x300   96x227x227  input
# norm1           LocalResponseNormalization2DLayer                       96x227x227  96x227x227  conv1
# pool1           MaxPool2DLayer                              3     3     96x227x227  96x75x75    norm1
# conv2           Conv2DDNNLayer                     256      5     1     96x75x75    256x71x71   pool1
# pool2           MaxPool2DLayer                              2     2     256x71x71   256x35x35   conv2
# conv3           Conv2DDNNLayer                     512      3     1     256x35x35   512x35x35   pool2
# conv4           Conv2DDNNLayer                     512      3     1     512x35x35   512x35x35   conv3
# conv5           Conv2DDNNLayer                     512      3     1     512x35x35   512x35x35   conv4
# pool5           MaxPool2DLayer                              3     3     512x35x35   512x11x11   conv5
# conv_fc6        Conv2DDNNLayer                     4096     6     1     512x11x11   4096x10x10  pool5
# drop6           TiedDropoutLayer                                        4096x10x10  4096x10x10  conv_fc6
# conv_fc7        Conv2DDNNLayer                     4096     1     1     4096x10x10  4096x10x10  drop6
# drop7           TiedDropoutLayer                                        4096x10x10  4096x10x10  conv_fc7
# nin_fc7         NINLayer                           8                    4096x10x10  8x10x10     drop7
# up1             TransposedConv2DLayer              8        4     2     8x10x10     8x20x20     nin_fc7
# crop_conv5      Crop2dLayer                                             512x35x35   512x20x20   conv5
# nin_crop_conv5  NINLayer                           8                    512x20x20   8x20x20     crop_conv5
# fuse1           ElemwiseSumLayer                                                    8x20x20     up1, nin_crop_conv5
# up2             TransposedConv2DLayer              8        4     2     8x20x20     8x40x40     fuse1
# nin_pool1       NINLayer                           8                    96x75x75    8x75x75     pool1
# drop_nin_pool1  TiedDropoutLayer                                        8x75x75     8x75x75     nin_pool1
# crop_nin_pool1  Crop2dLayer                                             8x75x75     8x40x40     drop_nin_pool1
# fuse2           ElemwiseSumLayer                                                    8x40x40     up2, crop_nin_pool1
# up3             TransposedConv2DLayer              8        5     3     8x40x40     8x120x120   fuse2
# nin_conv1       NINLayer                           8                    96x227x227  8x227x227   conv1
# crop_nin_conv1  Crop2dLayer                                             8x227x227   8x120x120   nin_conv1
# fuse3           ElemwiseSumLayer                                                                8x120x120   up3, crop_nin_conv1


class NIN(Layer):
    def __init__(self, kernel, mlps, strides, **kwargs):
        super(NIN, self).__init__(**kwargs)
        self.conv0 = Conv2D(mlps[0], kernel, strides=strides, padding='same', activation='relu')
        self.convs = []
        for size in mlps[1:]:
            self.convs.append(Conv2D(size, 1, strides=[1,1], activation='relu'))

    def call(self, x):
        x = self.conv0(x)
        for conv in self.convs:
            x = conv(x)
        return x



class CnnsFcn(Model):
    def __init__(self, input_shape=(227,227,3), n_classes=8):
        super(CnnsFcn, self).__init__()

        inp            = Input(shape=input_shape)
        conv1          = Conv2D(96, 7, strides = 2, activation='relu')(inp)     #3x300x300   96x227x227  input
        norm1          = BatchNormalization()(conv1)                                            #96x227x227  96x227x227  conv1
        pool1          = MaxPool2D((2, 2), strides=2)(norm1)                                    #96x227x227  96x75x75    norm1
        conv2          = Conv2D(256, 5, strides = 1, activation='relu')(pool1)  #96x75x75    256x71x71   pool1
        pool2          = MaxPool2D((2, 2), strides=2)(conv2)                                    #256x71x71   256x35x35   conv2
        conv3          = Conv2D(512, 3, strides = 1, activation='relu')(pool2)  #256x35x35   512x35x35   pool2
        conv4          = Conv2D(512, 3, strides = 1, padding="same", activation='relu')(conv3)  #512x35x35   512x35x35   conv3
        conv5          = Conv2D(512, 3, strides = 1, padding="same", activation='relu')(conv4)  #512x35x35   512x35x35   conv4
        pool5          = MaxPool2D((3, 3), strides=3)(conv5)                                    #512x35x35   512x11x11   conv5
        conv_fc6       = Conv2D(4096, 6, strides = 1, padding="same", activation='relu')(pool5)                 #512x11x11   4096x10x10  pool5
        drop6          = Dropout(0.5)(conv_fc6)                                                 #4096x10x10  4096x10x10  conv_fc6
        conv_fc7       = Conv2D(4096, 1, strides = 1, padding="same", activation='relu')(drop6) #4096x10x10  4096x10x10  drop6
        drop7          = Dropout(0.5)(conv_fc7)                                                 #4096x10x10  4096x10x10  conv_fc7
        nin_fc7        = NIN(3, [8, 8], [1,1])(drop7)
                                                #4096x10x10  8x10x10     drop7
        up1            = Conv2DTranspose(n_classes, 4, strides=2)(nin_fc7)                              #8x10x10     8x20x20     nin_fc7
        crphb = int((conv5.shape[1] - up1.shape[1])/2)
        crpht = (conv5.shape[1] - up1.shape[1]) - crphb
        crpwb = int((conv5.shape[2] - up1.shape[2])/2)
        crpwt = (conv5.shape[2] - up1.shape[2]) - crpwb
        crop_conv5     = Cropping2D(cropping=((crpht, crphb), (crpwt, crpwb)))(conv5)                           #512x35x35   512x20x20   conv5
        nin_crop_conv5 = NIN(3, [n_classes, n_classes], [1,1])(crop_conv5)                                      #512x20x20   8x20x20     crop_conv5
        fuse1          = Add()([up1, nin_crop_conv5])                                             #8x20x20     up1, nin_crop_conv5
        
        up2            = Conv2DTranspose(n_classes, 4, strides=2)(fuse1)                                #8x20x20     8x40x40     fuse1
        nin_pool1      = NIN(3, [n_classes, n_classes], [1,1])(pool1)                                           #96x75x75    8x75x75     pool1
        drop_nin_pool1 = Dropout(0.5)(nin_pool1)
        crphb = int((drop_nin_pool1.shape[1] - up2.shape[1])/2)
        crpht = (drop_nin_pool1.shape[1] - up2.shape[1]) - crphb
        crpwb = int((drop_nin_pool1.shape[2] - up2.shape[2])/2)
        crpwt = (drop_nin_pool1.shape[2] - up2.shape[2]) - crpwb                                                #8x75x75     8x75x75     nin_pool1
        crop_nin_pool1 = Cropping2D(cropping=((crpht, crphb), (crpwt, crpwb)))(drop_nin_pool1)              #8x75x75     8x40x40     drop_nin_pool1
        fuse2          = Add()([up2, crop_nin_pool1])
                                              #8x40x40     up2, crop_nin_pool1
        up3            = Conv2DTranspose(n_classes, 5, strides=3)(fuse2)                                #8x40x40     8x120x120   fuse2
        nin_conv1      = NIN(3, [n_classes, n_classes], [1,1])(conv1)
        crphb = int((nin_conv1.shape[1] - up3.shape[1])/2)
        crpht = (nin_conv1.shape[1] - up3.shape[1]) - crphb
        crpwb = int((nin_conv1.shape[2] - up3.shape[2])/2)
        crpwt = (nin_conv1.shape[2] - up3.shape[2]) - crpwb                                               #96x227x227  8x227x227   conv1
        crop_nin_conv1 = Cropping2D(cropping=((crpht, crphb), (crpwt, crpwb)))(nin_conv1)                   #8x227x227   8x120x120   nin_conv1
        fuse3          = Add()([up3, crop_nin_conv1])

        super(CnnsFcn, self).__init__(inputs=inp, outputs=fuse3, name='cnns_fcn')


 


class Darkfcn(Model):
    def __init__(self, input_shape=(300,300,3), n_classes=8):
        super(Darkfcn, self).__init__()
        #name         type                   units    sz    st    inshape     outshape    inputs
        #-----------  ---------------------  -------  ----  ----  ----------  ----------  ----------------
        inp         = Input(shape=input_shape)                                            #3x300x300
        conv01      = Conv2D(32  , 3, strides=1, padding='same', activation='relu')(inp    ) #3x300x300   32x330x330  
        pool01      = MaxPool2D((2, 2), strides=2)(conv01 )                                  #32x330x330  32x165x165  
        conv02      = Conv2D(64  , 3, strides=1, padding='same', activation='relu')(pool01 ) #32x165x165  64x165x165  
        pool02      = MaxPool2D((2, 2), strides=2)(conv02 )                                  #64x165x165  64x82x82    
        conv03      = Conv2D(128 , 3, strides=1, padding='same', activation='relu')(pool02 ) #64x82x82    128x82x82   
        conv04      = Conv2D(64  , 1, strides=1, padding='same', activation='relu')(conv03 ) #128x82x82   64x82x82    
        conv05      = Conv2D(128 , 3, strides=1, padding='same', activation='relu')(conv04 ) #64x82x82    128x82x82   
        pool03      = MaxPool2D((2, 2), strides=2)(conv05 )                                  #128x82x82   128x41x41   
        conv06      = Conv2D(256 , 3, strides=1, padding='same', activation='relu')(pool03 ) #128x41x41   256x41x41   
        spdrp6      = Dropout(0.5)(conv06)                                                   #256x41x41   256x41x41   
        conv07      = Conv2D(128 , 1, strides=1, padding='same', activation='relu')(spdrp6 ) #256x41x41   128x41x41   
        conv08      = Conv2D(256 , 3, strides=1, padding='same', activation='relu')(conv07 ) #128x41x41   256x41x41   
        pool04      = MaxPool2D((2, 2), strides=2)(conv08 )                                  #256x41x41   256x20x20   
        conv09      = Conv2D(512 , 3, strides=1, padding='same', activation='relu')(pool04 ) #256x20x20   512x20x20   
        spdrp9      = Dropout(0.5)(conv09)                                                   #512x20x20   512x20x20   
        conv10      = Conv2D(256 , 1, strides=1, padding='same', activation='relu')(spdrp9 ) #512x20x20   256x20x20   
        conv11      = Conv2D(512 , 3, strides=1, padding='same', activation='relu')(conv10 ) #256x20x20   512x20x20   
        spdrp11     = Dropout(0.5)(conv11)                                                   #512x20x20   512x20x20   
        conv12      = Conv2D(256 , 1, strides=1, padding='same', activation='relu')(spdrp11) #512x20x20   256x20x20   
        conv13      = Conv2D(512 , 3, strides=1, padding='same', activation='relu')(conv12 ) #256x20x20   512x20x20   
        pool05      = MaxPool2D((2, 2), strides=2)(conv13 )                                  #512x20x20   512x10x10   
        conv14      = Conv2D(1024, 3, strides=1, padding='same', activation='relu')(pool05 ) #512x10x10   1024x10x10  
        spdrp14     = Dropout(0.5)(conv14)                                                   #1024x10x10  1024x10x10  
        conv15      = Conv2D(512 , 1, strides=1, padding='same', activation='relu')(spdrp14) #1024x10x10  512x10x10   
        spdrp15     = Dropout(0.5)(conv15)                                                   #512x10x10   512x10x10   
        conv16      = Conv2D(1024, 3, strides=1, padding='same', activation='relu')(spdrp15) #512x10x10   1024x10x10  
        spdrp16     = Dropout(0.5)(conv16)                                                   #1024x10x10  1024x10x10  
        conv17      = Conv2D(512 , 1, strides=1, padding='same', activation='relu')(spdrp16) #1024x10x10  512x10x10   
        conv18      = Conv2D(1024, 3, strides=1, padding='same', activation='relu')(conv17 ) #512x10x10   1024x10x10  
        conv18nin   = Conv2D(64  , 1, strides=1, padding='same', activation='relu')(conv18 ) #1024x10x10  64x10x10    
        
        up1         = Conv2DTranspose(64, 4, strides=2, padding='same')(conv18nin)                           #64x10x10    64x20x20
        pool04nin   = Conv2D(64  , 1, strides=1, padding='same', activation='relu')(pool04 ) #256x20x20   64x20x20  
        crphb = int((pool04nin.shape[1] - up1.shape[1])/2)
        crpht =     (pool04nin.shape[1] - up1.shape[1]) - crphb
        crpwb = int((pool04nin.shape[2] - up1.shape[2])/2)
        crpwt =     (pool04nin.shape[2] - up1.shape[2]) - crpwb
        if crphb < 0:
            pool04ninc = ZeroPadding2D(padding=((-crpht, -crphb), (-crpwt, -crpwb)))(pool04nin)
        else:
            pool04ninc  = Cropping2D(cropping=((crpht, crphb), (crpwt, crpwb)))(pool04nin)       #64x20x20    64x20x20
        pool04nincs = BatchNormalization()(pool04ninc)                                       #64x20x20    64x20x20
        fuse1       = Add()([up1, pool04nincs])                                                #64x20x20
        
        up2         = Conv2DTranspose(64, 4, strides=2, padding='same')(fuse1    )                           #64x20x20    64x40x40
        pool03nin   = Conv2D(64  , 1, strides=1, padding='same', activation='relu')(pool03 ) #128x41x41   64x41x41  
        crphb = int((pool03nin.shape[1] - up2.shape[1])/2)
        crpht =     (pool03nin.shape[1] - up2.shape[1]) - crphb
        crpwb = int((pool03nin.shape[2] - up2.shape[2])/2)
        crpwt =     (pool03nin.shape[2] - up2.shape[2]) - crpwb
        if crphb < 0:
            pool03ninc = ZeroPadding2D(padding=((-crpht, -crphb), (-crpwt, -crpwb)))(pool03nin)
        else:
            pool03ninc  = Cropping2D(cropping=((crpht, crphb), (crpwt, crpwb)))(pool03nin)       #64x41x41    64x40x40
        pool03nincs = BatchNormalization()(pool03ninc)                                       #64x40x40    64x40x40  
        fuse2       = Add()([up2, pool03nincs])                                                #64x40x40
        
        up3         = Conv2DTranspose(64, 4, strides=2, padding='same')(fuse2    )                           #64x40x40    64x80x80
        pool02nin   = Conv2D(64  , 1, strides=1, padding='same', activation='relu')(pool02 ) #64x82x82    64x82x82  
        crphb = int((pool02nin.shape[1] - up3.shape[1])/2)
        crpht =     (pool02nin.shape[1] - up3.shape[1]) - crphb
        crpwb = int((pool02nin.shape[2] - up3.shape[2])/2)
        crpwt =     (pool02nin.shape[2] - up3.shape[2]) - crpwb
        if crphb < 0:
            pool02ninc = ZeroPadding2D(padding=((-crpht, -crphb), (-crpwt, -crpwb)))(pool02nin)
        else:
            pool02ninc  = Cropping2D(cropping=((crpht, crphb), (crpwt, crpwb)))(pool02nin)       #64x82x82    64x80x80
        pool02nincs = BatchNormalization()(pool02ninc)                                       #64x80x80    64x80x80  
        fuse3       = Add()([up3, pool02nincs])                                                #64x80x80
        
        up4         = Conv2DTranspose(64, 4, strides=2, padding='same')(fuse3    )                           #64x80x80    64x160x160
        pool01nin   = Conv2D(64  , 1, strides=1, padding='same', activation='relu')(up4    ) #32x165x165  64x165x165
        crphb = int((pool01nin.shape[1] - up4.shape[1])/2)
        crpht =     (pool01nin.shape[1] - up4.shape[1]) - crphb
        crpwb = int((pool01nin.shape[2] - up4.shape[2])/2)
        crpwt =     (pool01nin.shape[2] - up4.shape[2]) - crpwb
        if crphb < 0:
            pool01ninc = ZeroPadding2D(padding=((-crpht, -crphb), (-crpwt, -crpwb)))(pool01nin)
        else:
            pool01ninc  = Cropping2D(cropping=((crpht, crphb), (crpwt, crpwb)))(pool01nin)       #64x165x165  64x160x160
        pool01nincs = BatchNormalization()(pool01ninc)                                       #64x160x160  64x160x160
        fuse4       = Add()([up4, pool01nincs])                                                #64x160x160
        
        up5         = Conv2DTranspose(64, 4, strides=2, padding='same')(fuse4    )                           #64x160x160  64x320x320
        crphb = int((up5.shape[1] - input_shape[0])/2)
        crpht =     (up5.shape[1] - input_shape[0]) - crphb
        crpwb = int((up5.shape[2] - input_shape[1])/2)
        crpwt =     (up5.shape[2] - input_shape[1]) - crpwb
        if crphb < 0:
            up5c = ZeroPadding2D(padding=((-crpht, -crphb), (-crpwt, -crpwb)))(up5)
        else:
            up5c        = Cropping2D(cropping=((crpht, crphb), (crpwt, crpwb)))(up5      )       #64x320x320  64x300x300
        up5ninc     = NIN(3,[n_classes,n_classes], [1,1])(up5c)                                            #64x300x300  8x300x300
        super(Darkfcn, self).__init__(inputs=inp, outputs=up5ninc, name='darkfcn')



class UpNet(Model):
    def __init__(self, input_shape=(300,300,3), n_classes=8):
        super(UpNet, self).__init__()
        # 300x300x3->
        # 318x318x64  2cv1p->
        # 159x159x128 2vc1p->
        # 80x80x256   3cv1p->
        # 40x40x512   3cv1p-> 
        # 20x20x512   3vc1p-> 
        # 8x8x1024    2cv ->|| 
        
        # 12x12xnc    1ur1cvd -> 
        # 26x26xnc    1ur1cv-> 
        # 54x54xnc    1ur1cv-> 
        # 110x110xnc  1ur1cv -> 
        # 182x182xnc  1ur1sfm ->
        # 300x300
        c_scalar = 2;

        inp    = Input(shape=input_shape)
        zp1     = ZeroPadding2D(padding=(9,9))                                 (   inp)
        conv1a = Conv2D(64, 3, strides = 1, padding="same", activation='relu') (   zp1) #318   				
        conv1b = Conv2D(64, 3, strides = 1, padding="same", activation='relu') (conv1a)     				
        pool1  = MaxPool2D((2, 2), strides=2)                                  (conv1b)               
        
        conv2a = Conv2D(128, 3, strides = 1, padding="same", activation='relu')( pool1) #159			
        conv2b = Conv2D(128, 3, strides = 1, padding="same", activation='relu')(conv2a)
        zp2     = ZeroPadding2D(padding=(1,1))                                 (conv2b)  				
        pool2  = MaxPool2D((2, 2), strides=2)                                  (   zp2)                  
        
        
        conv3a = Conv2D(256, 3, strides = 1, padding="same", activation='relu')( pool2)  #80 
        conv3b = Conv2D(256, 3, strides = 1, padding="same", activation='relu')(conv3a)  
        conv3c = Conv2D(256, 3, strides = 1, padding="same", activation='relu')(conv3b)  
        pool3  = MaxPool2D((2, 2), strides=2)                                  (conv3c)    

        conv4a = Conv2D(512, 3, strides = 1, padding="same", activation='relu')( pool3) #40
        conv4b = Conv2D(512, 3, strides = 1, padding="same", activation='relu')(conv4a)  
        conv4c = Conv2D(512, 3, strides = 1, padding="same", activation='relu')(conv4b)  
        pool4  = MaxPool2D((2, 2), strides=2)                                  (conv4c)   

        conv5a = Conv2D(512, 3, strides = 1, padding="same", activation='relu')( pool4) #20
        conv5b = Conv2D(512, 3, strides = 1, padding="same", activation='relu')(conv5a)  
        conv5c = Conv2D(512, 3, strides = 1, padding="same", activation='relu')(conv5b)  
        pool5  = MaxPool2D((2, 2), strides=2)                                  (conv5c) 

        conv6a = Conv2D(1024, 3, strides = 1, padding="valid", activation='relu')(pool5) #8
        conv6b = Conv2D(1024, 3, strides = 1, padding="same")(conv6a)  

        up1    = Conv2DTranspose(c_scalar*n_classes, 5, strides=1, padding='valid', activation='relu')(conv6b)   
        conv_up1 = Conv2D(c_scalar*n_classes, 3, strides = 1, padding="same")(up1) 
        dp = Dropout(0.5)(conv_up1)
        
        up2    = Conv2DTranspose(c_scalar*n_classes, 4, strides=2, padding='valid', activation='relu')(dp)   
        conv_up2 = Conv2D(c_scalar*n_classes, 3, strides = 1, padding="same")(up2) 
        
        up3    = Conv2DTranspose(c_scalar*n_classes, 4, strides=2, padding='valid', activation='relu')(conv_up2)   
        conv_up3 = Conv2D(c_scalar*n_classes, 3, strides = 1, padding="same")(up3) 
        
        up4    = Conv2DTranspose(c_scalar*n_classes, 4, strides=2, padding='valid', activation='relu')(conv_up3)   
        conv_up4 = Conv2D(c_scalar*n_classes, 3, strides = 1, padding="same")(up4) 
        
        up5 = Lambda(lambda x: tf.image.resize(x,(182,182),
                            method=tf.image.ResizeMethod.BILINEAR),
                            name='up5')(conv_up4)
        relu_up5 = ReLU()(up5)
        conv_up5 = Conv2D(n_classes, 3, strides = 1, padding="same")(relu_up5) 
        
        up6 = Lambda(lambda x: tf.image.resize(x,(300,300),
                            method=tf.image.ResizeMethod.BILINEAR),
                            name='up6')(conv_up5)  
        sf = Softmax()(up6)
        UpSampling2D(interpolation='bilinear')
                                  
        super(UpNet, self).__init__(inputs=inp, outputs=sf, name='upnet')
     


# cnns_fcn = Model(inputs=inp, outputs=fuse3, name='cnns_fcn')
cnns_fcn = CnnsFcn(input_shape=(300,300,3), n_classes=5)
darkfcn = Darkfcn(input_shape=(300,300,3), n_classes=5)
upnet = UpNet(input_shape=(300,300,3), n_classes=5)
#cnns_fcn.build((None, 227,227,3))
cnns_fcn.summary()
darkfcn.summary()
upnet.summary()




