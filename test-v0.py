

import MultiModels
from keras.models import Model
from keras.utils import np_utils
import pdb
import numpy as np 
import os
import threading
import lmdb
import sys
from keras import backend as K
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import RMSprop,SGD,Adam
from keras.layers.core import *

np.random.seed(seed=1234)

data_root = '/home/siqinli/dataset/data_out/cross_subject_2d_norm'	## CHECK!!
lmdb_file_test_x = os.path.join(data_root,'Xtest_lmdb')
lmdb_file_test_y = os.path.join(data_root,'Ytest_lmdb')

weight_dir_name = 'lr0.01w10_0.5'		## CHECK!!
weight_file_name = '099_0.957.hdf5'	## CHECK!!
print "running file: " + weight_file_name

samples_per_validation = 16148 #23185
n_classes = 60
max_len = 300
feat_dim_2D = 100
feat_dim_3D = 150	## CHECK!!
defreeze_layer = 71 #TOTAL 73
extract_2D_hal = 71



lr = 0.001
momentum = 0.9
weight_decay = 0.0005

## MODEL PARAMS 
optimizer = SGD(lr = lr, momentum = momentum, decay = weight_decay, nesterov = True)
dropout = 0.5
reg = l1(1.e-4) 
activation = "relu"

batch_size = 128

## get ground truth label
gt = np.zeros(((samples_per_validation/batch_size)*batch_size, n_classes))

# """Takes an iterator/generator and makes it thread-safe by
#   serializing call to the `next` method of given iterator/generator.
# """
class threadsafe_iter:
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()


def threadsafe_generator(f):
	def g(*a, **kw):
  		return threadsafe_iter(f(*a, **kw))
  	return g


@threadsafe_generator
def nturgbd_test_datagen():
	lmdb_env_x = lmdb.open(lmdb_file_test_x)
	lmdb_txn_x = lmdb_env_x.begin()
	lmdb_cursor_x = lmdb_txn_x.cursor()

	lmdb_env_y = lmdb.open(lmdb_file_test_y)
	lmdb_txn_y = lmdb_env_y.begin()
	lmdb_cursor_y = lmdb_txn_y.cursor()
  
	X = np.zeros((batch_size,max_len,feat_dim_2D))
	Y = np.zeros((batch_size,n_classes))
	print "generating data..."

	count = 0	#testing

	while True:
		batch_count = 0
		indices = range(0,samples_per_validation)
		np.random.shuffle(indices)

		for index in indices:
			value = np.frombuffer(lmdb_cursor_x.get('{:08d}'.format(index)))
			label = np.frombuffer(lmdb_cursor_y.get('{:08d}'.format(index)))
			x = value.reshape(max_len,feat_dim_2D)

			##testing
			if count < len(gt):
				gt[count] = label
				count += 1
			##

			X[batch_count] =  x
			Y[batch_count] = label

			batch_count += 1

			if batch_count == batch_size:
				ret_x = X
				ret_y = Y
				X = np.zeros((batch_size,max_len,feat_dim_2D))
				Y = np.zeros((batch_size,n_classes))
				batch_count = 0

				yield (ret_x, [ret_y,ret_y,ret_y])


def test():
	model = MultiModels.Multi_mdl(n_classes, feat_dim_2D, feat_dim_3D, max_len,
									dropout = dropout, 
									W_regularizer = reg, 
									activation = activation,
									defreeze_layer = defreeze_layer, 
									extract_2D_hal = extract_2D_hal)
	# model.summary()
	## LOAD WEIGHT
	model.load_weights('weights/' + weight_dir_name + '/' + weight_file_name)
	## EXTRATING LAYERS
	# model.get_layer("2D_EXTRACT_OUT").summary()
	# model.get_layer("HAL_EXTRACT_OUT").summary()
	# print model.layers[0].input
	# print model.get_layer("2D_EXTRACT_OUT").layers[len(model.get_layer("2D_EXTRACT_OUT").layers)-1].output
	# print model.get_layer("HAL_EXTRACT_OUT").layers[len(model.get_layer("HAL_EXTRACT_OUT").layers)-1].output
	# print model.get_layer("2DHAL_AVE_OUT").output
	# print model.get_layer("average_1").input[0]
	# print model.get_layer("average_1").input[1]
	# output_2d = Activation("softmax", name = '2D_SOFTMAX_OUT')(model.get_layer("average_1").input[0])
	# output_hal = Activation("softmax", name = 'HAL_SOFTMAX_OUT')(model.get_layer("average_1").input[1])
	# test_mdl = Model(inputs = model.layers[0].input, 
	# 					outputs = [output_2d,
	# 					output_hal, 
	# 					model.get_layer("2DHAL_AVE_OUT").output])
	# test_mdl.summary()
	# test_mdl.compile(loss = ['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
	# 				optimizer = optimizer,
	# 				metrics = {'2D_SOFTMAX_OUT': 'acc','HAL_SOFTMAX_OUT':'acc','2DHAL_AVE_OUT' : 'acc'})

	test_mdl = Model(inputs = model.get_layer("2D_INPUT").input, 
		outputs = [model.output[0], 
					model.output[1], 
					model.output[2]])
	# test_mdl.summary()

	test_mdl.compile(loss = ['categorical_crossentropy', 
						'categorical_crossentropy','categorical_crossentropy'],
						optimizer = optimizer,
						metrics = {'2D_MDL': 'acc','HAL_MDL':'acc','2DHAL_OUT' : 'acc'})

	print "test"
	score = test_mdl.evaluate_generator(nturgbd_test_datagen(),
								steps = samples_per_validation/batch_size,
								workers = 1)

	print test_mdl.metrics_names
	print score

	# print "predict"
	# prediction = test_mdl.predict_generator(nturgbd_test_datagen(), 
	# 										steps = samples_per_validation/batch_size,
	# 										verbose = 1)
 
	# prob = np.multiply(prediction, gt)
	# print "average predicted probability at ground truth classes:"
	# print np.average(np.max(prob,axis = 1))
	# print "accuracy:"
	# print sum(np.argmax(prediction,axis = 1) == np.argmax(gt, axis = 1))/float(samples_per_validation)




if __name__ == "__main__":
	test()

