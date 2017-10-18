"""

Multimodel using 2D and 3D hallucination

"""

import ResTCN_Model as Models
import numpy as np
import os

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, Lambda
from keras.layers.merge import Concatenate, Add, Average
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def eucl_dist(vects):
	x, y = vects
	# return np.sum((x - y)**2,axis = 0)
	return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dis_output_shape(vects):
	x, y = vects
	return (x[0],1)

def crossentropy(y_true, y_pred):
	return K.categorical_crossentropy(y_true, y_pred)


def get_output_2D_hal(input_2D, model_2D, model_hal, extract_2D_hal, n_classes): 
	intermediate_model_2D = Model(inputs = model_2D.layers[0].input, 
									outputs = model_2D.layers[extract_2D_hal].output)
	intermediate_model_2D.name = "2D_EXTRACT_MDL"
	in_2D = intermediate_model_2D(input_2D)
	intermediate_model_hal = Model(inputs = model_hal.layers[0].input, 
									outputs = model_hal.layers[extract_2D_hal].output)
	intermediate_model_hal.name = "HAL_EXTRACT_MDL"
	# intermediate_model_hal.layers[len(intermediate_model_hal.layers)-1].name = "HAL_SOFTMAX_OUT"
	in_hal = intermediate_model_hal(input_2D)
	pool_window_shape = K.int_shape(in_2D)
	output_2D_hal = Average()([in_2D, in_hal])
	# output_2D_hal = AveragePooling1D(pool_window_shape[ROW_AXIS],
 #                           strides=1)(output_2D_hal)
	# output_2D_hal = Flatten()(output_2D_hal)
	# output_2D_hal = Dense(units=n_classes,
 #        				kernel_initializer="he_normal",
 #        				activation="softmax", name = '2DHAL_AVE_OUT')(output_2D_hal)
 	output_2D_hal = Activation("softmax", name = '2DHAL')(output_2D_hal)
	return output_2D_hal

def get_output_hal_mid(input_2D, model_hal, defreeze_layer):
	intermediate_model_hal = Model(inputs = model_hal.layers[0].input, 
									outputs = model_hal.layers[defreeze_layer-1].output)
	output_hal = intermediate_model_hal(input_2D)
	return output_hal

def get_output_hal_3D(input_2D, input_3D, model_hal, model_3D, defreeze_layer):
	intermediate_model_hal = Model(inputs = model_hal.layers[0].input, 
									outputs = model_hal.layers[defreeze_layer-1].output)
	intermediate_model_hal.name = "HAL_DEFREEZE_OUTPUT"
	in_hal = intermediate_model_hal(input_2D)
	intermediate_model_3D = Model(inputs = model_3D.layers[0].input, 
									outputs = model_3D.layers[defreeze_layer-1].output)
	intermediate_model_3D.name = "3D_DEFREEZE_OUTPUT"
	in_3D = intermediate_model_3D(input_3D)
	# pool_window_shape = K.int_shape(in_hal)
	# mid_layer_hal = AveragePooling1D(pool_window_shape[ROW_AXIS],
 #                           strides=1)(in_hal)
	# mid_layer_hal = Flatten()(in_hal)
	mid_layer_hal = Activation("sigmoid")(in_hal)
	# mid_layer_3D = AveragePooling1D(pool_window_shape[ROW_AXIS],
	                           # strides=1)(in_3D)
	# mid_layer_3D = Flatten()(in_3D)
	mid_layer_3D = Activation("sigmoid")(in_3D)
	output_hal_3D = Lambda(eucl_dist,output_shape = eucl_dis_output_shape, name = 'HAL3D_OUT')([mid_layer_hal,mid_layer_3D])
	return output_hal_3D

def Multi_mdl(n_classes, feat_dim_2D, feat_dim_3D, max_len, dropout, W_regularizer, activation,
				defreeze_layer, extract_2D_hal):

	model_path = 'pretrained_model/'
	weight2D_name = '2D_0.745.hdf5'		## CHECK!!
	weight3D_name = '3D_0.758.hdf5'		## CHECK!!
	hallucination_weight = '2D_0.745.hdf5'		## CHECK!!

	if K.image_dim_ordering() == 'tf':
	    ROW_AXIS = 1
	    CHANNEL_AXIS = 2
	else:
	    ROW_AXIS = 2
	    CHANNEL_AXIS = 1

	    ## LOAD PRETRAINED MODEL
	model_2D = Models.TK_TCN_resnet_v3( 
	         n_classes, 
	         feat_dim_2D,
	         max_len=300,
	         gap=1,
	         dropout = 0.0,
	         W_regularizer = l1(1.e-4),
	         activation = "relu")
	model_2D.load_weights(model_path + weight2D_name)
	# print(len(model_2D.layers))


	model_3D = Models.TK_TCN_resnet_v3( 
	         n_classes, 
	         feat_dim_3D,
	         max_len=300,
	         gap=1,
	         dropout = 0.0,
	         W_regularizer = l1(1.e-4),
	         activation = "relu")
	model_3D.load_weights(model_path + weight3D_name)
	# FREEZE THE WEIGHTS IN 3D
	for i in range(0,defreeze_layer):
		model_3D.layers[i].trainable = False


	model_hal = Models.TK_TCN_resnet_v3( 
	         n_classes, 
	         feat_dim_2D,
	         max_len=300,
	         gap=1,
	         dropout = 0.0,
	         W_regularizer = l1(1.e-4),
	         activation = "relu")
	model_hal.load_weights(model_path + hallucination_weight)
	# model_hal.summary()

	## PRINT LAYER INFO
	# print "2d and hal extract the output from layer:"
	# print model_2D.layers[extract_2D_hal]
	# print "hal and 3d last fronzen layer is"
	# print model_hal.layers[defreeze_layer-1]

	input_2D = Input(shape=(max_len,feat_dim_2D), name = '2D_INPUT')
	input_3D = Input(shape=(max_len,feat_dim_3D), name = '3D_INPUT')

	output_2D = model_2D(input_2D)
	model_2D.name = "2D"
	output_hal = model_hal(input_2D)
	model_hal.name = "HAL"
	# output_3D = model_3D(input_3D)
	# output_2D = Activation("softmax", name = '2D_SOFTMAX_OUT')(model_2D.get_layer("softmax").input)
	# output_hal = Activation("softmax", name = 'HAL_SOFTMAX_OUT')(model_hal.get_layer("softmax").input)

	output_2D_hal = get_output_2D_hal(input_2D, model_2D, model_hal, extract_2D_hal, n_classes)
	output_hal_3D = get_output_hal_3D(input_2D, input_3D, model_hal, model_3D, defreeze_layer)
	# output_hal_defreeze = get_output_hal_mid(input_2D, model_hal, defreeze_layer)

	multi_model = Model(inputs = [input_2D, input_3D], 
						outputs = [output_2D_hal, output_2D, output_hal, output_hal_3D])
	return multi_model
	# print model_hal.layers[defreeze_layer-1]#.output_shape
