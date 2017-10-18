import MultiModels
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2,l1_l2,l1
import pdb
import numpy as np
import lmdb
import threading
import os
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import backend as K
from keras.models import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(seed=1234)

##
out_dir_name = 'lr0.01w10_0.5'			#### CHECK!!
data_root_2D = '/home/siqinli/dataset/data_out/cross_subject_2d_norm'		#### CHECK!!
data_root_3D = '/home/siqinli/dataset/data_out/cross_subject_3d_norm'	#### CHECK!!

##
n_classes = 60
max_len = 300
feat_dim_2D = 100
feat_dim_3D = 150	## CHECK!!
defreeze_layer = 71 #TOTAL 73
extract_2D_hal = 71

## OPTIMIZER PARAMS
loss_2D_hal = 'categorical_crossentropy'
loss_hal_3D = 'mean_squared_error'
lr = 0.001
momentum = 0.9
weight_decay = 0.0005

## MODEL PARAMS 
optimizer = SGD(lr = lr, momentum = momentum, decay = weight_decay, nesterov = True)
dropout = 0.5
reg = l1(1.e-4) 
activation = "relu"

## TRAINING PARAMS
batch_size = 128
nb_epoch = 100
verbose = 1
shuffle = 1
weight_2D_hal = 0.2		## CHECK!!
weight_hal_3D = 2	## CHECK!!
weight_2D = 0.2		## CHECK!!
weight_hal = 0.2	## CHECK!!

samples_per_epoch = 39315#33094
samples_per_validation = 16148#23185

lmdb_file_train_x_2d = os.path.join(data_root_2D,'Xtrain_lmdb')
lmdb_file_train_x_3d = os.path.join(data_root_3D,'Xtrain_lmdb')
lmdb_file_train_y = os.path.join(data_root_2D,'Ytrain_lmdb')

lmdb_file_test_x = os.path.join(data_root_2D,'Xtest_lmdb')
lmdb_file_test_y = os.path.join(data_root_2D,'Ytest_lmdb')



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
def nturgbd_train_datagen():
	lmdb_env_x_2d = lmdb.open(lmdb_file_train_x_2d)
	lmdb_txn_x_2d = lmdb_env_x_2d.begin()
	lmdb_cursor_x_2d = lmdb_txn_x_2d.cursor()

	lmdb_env_x_3d = lmdb.open(lmdb_file_train_x_3d)
	lmdb_txn_x_3d = lmdb_env_x_3d.begin()
	lmdb_cursor_x_3d = lmdb_txn_x_3d.cursor()

	lmdb_env_y = lmdb.open(lmdb_file_train_y)
  	lmdb_txn_y = lmdb_env_y.begin()
  	lmdb_cursor_y = lmdb_txn_y.cursor()

  	X_2D = np.zeros((batch_size,max_len,feat_dim_2D))
  	X_3D = np.zeros((batch_size,max_len,feat_dim_3D))
  	Y_2D_hal= np.zeros((batch_size,n_classes))
  	Y_hal_3D = np.zeros((batch_size,1))
  	batch_count = 0

  	while True:
	  	indices = range(0, samples_per_epoch)
	  	np.random.shuffle(indices)

	  	for index in indices:
	  		value_2d = np.frombuffer(lmdb_cursor_x_2d.get('{:0>8d}'.format(index)))
	  		value_3d = np.frombuffer(lmdb_cursor_x_3d.get('{:0>8d}'.format(index)))
	  		label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(index)))

	  		x_2d = value_2d.reshape(max_len,feat_dim_2D)
	  		x_3d = value_3d.reshape(max_len,feat_dim_3D)

	  		## ORIGINAL
	  		X_2D[batch_count] = x_2d
	  		X_3D[batch_count] = x_3d
	  		Y_2D_hal[batch_count] = label
	  		batch_count += 1
	  		if batch_count == batch_size:
	  			ret_x = [X_2D, X_3D]
	  			ret_y = [Y_2D_hal, Y_2D_hal, Y_2D_hal, Y_hal_3D]
	  			X_2D = np.zeros((batch_size,max_len,feat_dim_2D))
	  			X_3D = np.zeros((batch_size,max_len,feat_dim_3D))
	  			Y_2D_hal= np.zeros((batch_size,n_classes))
	  			batch_count = 0
	  			yield (ret_x, ret_y)


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

			# ##testing
			# if count < len(gt):
			# 	gt[count] = label
			# 	count += 1
			# ##

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




def eucl_dist(x, y):
	return np.sum((x - y)**2,axis = 0)

def eucl_dis_output_shape(x, y):
	return (x[0],1)

def crossentropy(y_true, y_pred):
	return K.categorical_crossentropy(y_true, y_pred)

# def joint_loss(y_true, y_pred): 
# 	hal2D_true, hal3D_true= y_true
# 	# hal3D_true = y_true[1]
# 	hal2D_pred, hal3D_pred = y_pred 
# 	# hal3D_pred = y_pred[1]
# 	hal2D_loss = abs(hal3D_true - hal3D_pred)
# 	hal3D_loss = crossentropy(hal2D_true, hal2D_pred)
# 	return weight_2D_hal*hal2D_loss + weight_hal_3D*hal3D_loss


def train():
	model = MultiModels.Multi_mdl(n_classes, feat_dim_2D, feat_dim_3D, max_len, 
									dropout = dropout, 
									W_regularizer = reg, 
									activation = activation,
									defreeze_layer = defreeze_layer, 
									extract_2D_hal = extract_2D_hal)
	# model.load_weights('weights/' + out_dir_name + '/005_0.839.hdf5')

	# print model.summary()

	model.compile(loss = ['categorical_crossentropy', 'categorical_crossentropy',
							'categorical_crossentropy', 'mse'], 
						loss_weights = [weight_2D_hal, weight_2D, weight_hal, weight_hal_3D],
						optimizer = optimizer, 
						metrics={'2DHAL':'acc'})#, 'HAL_MDL':'acc'})

	if not os.path.exists('weights/'+out_dir_name):
		os.makedirs('weights/'+out_dir_name) 
	weight_path = 'weights/'+out_dir_name+'/{epoch:03d}_{2DHAL_acc:0.3f}.hdf5'
	checkpoint = ModelCheckpoint(weight_path, 
	                               monitor='2DHAL_acc', 
	                               verbose=1, 
	                               save_best_only=True, mode='max')
	reduce_lr = ReduceLROnPlateau(monitor='2DHAL_acc', 
	                                factor=0.1,
	                                patience=10, 
	                                verbose=1,
	                                mode='auto',
	                                cooldown=3,
	                                min_lr=0.00001)

	class Validation(Callback):
		# def __init__(self, test_datagen):
		# 	self.test_datagen = test_datagen

		def on_epoch_end(self, epoch, logs={}):
			logs = logs or {}
			test_mdl = Model(inputs = model.get_layer("2D_INPUT").input, 
							outputs = [model.output[0], 
										model.output[1], 
										model.output[2]])
			# test_mdl.get_layer("2DHAL_OUT").name = "TEST_2DHAL_OUT"
			# print test_mdl.summary()
			test_mdl.compile(loss = ['categorical_crossentropy', 
						'categorical_crossentropy','categorical_crossentropy'],
						optimizer = optimizer,
						metrics = {'2D': 'acc','HAL':'acc','2DHAL' : 'acc'})
			print "test"
			score = test_mdl.evaluate_generator(nturgbd_test_datagen(),
								steps = samples_per_validation/batch_size,
								workers = 1)
			for i in range(0, len(score)):
				if (test_mdl.metrics_names[i] == '2DHAL_acc'):
					logs['2DHAL_acc'] = score[i]
				print (test_mdl.metrics_names[i]+ ': {}'.format(score[i]))



  	callbacks_list = [Validation(), checkpoint, reduce_lr]

	model.fit_generator(nturgbd_train_datagen(), 
  						steps_per_epoch = samples_per_epoch/batch_size,
  						epochs = nb_epoch,
  						verbose = 1,
  						callbacks = callbacks_list,
  						workers = 1,
  						initial_epoch = 0
  						)
 # 	weight_dir_name = 'xy_3D_norm'
	# weight_file_name = '004_3.669.hdf5'
 # 	model.load_weights('weights/' + weight_dir_name + '/' + weight_file_name)
	# model.predict_generator(nturgbd_train_datagen(),
	# 							steps = samples_per_epoch,
	# 							workers = 1,
	# 							verbose = 1)


if __name__ == "__main__":
  train()
