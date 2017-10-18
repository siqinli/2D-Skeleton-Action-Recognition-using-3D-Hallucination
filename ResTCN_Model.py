import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, Lambda
from keras.layers.merge import Concatenate, Add
from keras.layers.pooling import AveragePooling1D
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.regularizers import l2,l1_l2,l1
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K

from keras.activations import relu
from functools import partial

def TK_TCN_resnet_v3(
           n_classes, 
           feat_dim,
           max_len,
           gap=1,
           dropout=0.0,
           W_regularizer=l1(1.e-4),
           activation="relu"):

  if K.image_dim_ordering() == 'tf':
    ROW_AXIS = 1
    CHANNEL_AXIS = 2
  else:
    ROW_AXIS = 2
    CHANNEL_AXIS = 1
  
  
  config = [ 
             [[(1,8,32)],[(1,16,32)]],
             [[(1,8,32)],[(1,16,32)]],
             [[(1,8,32)],[(1,16,32)]],
             [[(2,8,64)],[(2,16,64)]],
             [[(1,8,64)],[(1,16,64)]],
             [[(1,8,64)],[(1,16,64)]],
             [[(2,8,128)],[(2,16,128)]],
             [[(1,8,128)],[(1,16,128)]],
             [[(1,8,128)],[(1,16,128)]],
           ]

  initial_conv_len = 8
  initial_conv_num = 64

  input = Input(shape=(max_len,feat_dim))
  model = input

  model = Conv1D(initial_conv_num, 
                initial_conv_len,
                kernel_initializer="he_normal",
                padding="same",
                strides=1,
                kernel_regularizer=W_regularizer)(model)

  for depth in range(0,len(config)):
    blocks = []
    num_filters_this_layer = 0
    bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
    relu = Activation(activation)(bn)
    dr = Dropout(dropout)(relu)
    for multiscale in config[depth]:
      for stride,filter_dim,num in multiscale:
        ## residual block
        conv = Conv1D(num, 
                      filter_dim,
                      kernel_initializer="he_normal",
                      padding="same",
                      strides=stride,
                      kernel_regularizer=W_regularizer)(dr)
        blocks.append(conv)
        num_filters_this_layer += num
    # res = merge(blocks,mode='concat',concat_axis=CHANNEL_AXIS)
    res = Concatenate(axis=CHANNEL_AXIS)(blocks)
      #dr = Dropout(dropout)(conv)


    ## potential downsample
    conv_shape = K.int_shape(res)
    model_shape = K.int_shape(model)
    if conv_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
      model = Conv1D(num_filters_this_layer, 
                    1,
                    kernel_initializer="he_normal",
                    padding="same",
                    strides=stride,
                    kernel_regularizer=W_regularizer)(model)

    ## merge block
    # model = merge([model,res],mode='sum',concat_axis=CHANNEL_AXIS)
    model = Add()([model,res])

  ## final bn+relu
  # bn = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(model)
  bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
  model = Activation(activation)(bn)


  if gap:
    pool_window_shape = K.int_shape(model)
    gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                           strides=1)(model)
    flatten = Flatten()(gap)
  else:
    flatten = Flatten()(model)
  dense = Dense(units=n_classes,
        kernel_initializer="he_normal", name = 'fc1')(flatten)
  dense = Activation("softmax", name = 'softmax')(dense)
  model = Model(inputs=input, outputs=dense)

  return model
