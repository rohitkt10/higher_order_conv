import tensorflow as tf
from tensorflow import keras as tfk

from ..layers import PairwiseConv1D, NearestNeighborConv1D
from ..layers import conv1d_block, dense_block
from ..regularizers import PairwiseKernelRegularizer, NearestNeighborKernelRegularizer

__all__ =['get_model']

conv_type_dict = {
    'regular':tfk.layers.Conv1D,
    'pairwise':PairwiseConv1D,
    'nearest_neighbor':NearestNeighborConv1D
}

pool_type_dict = {
	'max': lambda x : tf.reduce_max(x, axis=1), 
	'avg': lambda x : tf.reduce_mean(x, axis=1)
}

def get_kernel_regularizer(conv_type, regularizer_type, regularizer_const, 
							pairwise_regularizer_type, pairwise_regularizer_const, 
							nn_regularizer_type, nn_regularizer_const):
	# set up the regularizer
	assert regularizer_type.lower() in ['l2', 'l1']
	kernel_regularizer = eval('tfk.regularizers.'+regularizer_type)(regularizer_const)
	
	if conv_type == 'pairwise':
		if pairwise_regularizer_type is not None:
			assert pairwise_regularizer_type.lower() in ['l2', 'l1']
			pairwise_reg = eval('tfk.regularizers.'+pairwise_regularizer_type)(pairwise_regularizer_const)
		else:
			pairwise_reg = kernel_regularizer
		kernel_regularizer = PairwiseKernelRegularizer(kernel_regularizer, pairwise_reg)

	if conv_type == 'nearest_neighbor':
		if nn_regularizer_type is not None:
			assert nn_regularizer_type.lower() in ['l2', 'l1']
			nn_reg = eval('tfk.regularizers.'+nn_regularizer_type)(nn_regularizer_const)
		else:
			nn_reg = kernel_regularizer
		kernel_regularizer = NearestNeighborKernelRegularizer(kernel_regularizer, nn_reg)
	return kernel_regularizer

def get_model(L, A, conv_type='regular', 
			num_filters=16, filter_size=24, use_bias=True, padding='valid',
			activation='relu', regularizer_type='l2', regularizer_const=1e-6,
			dropout=0.5, 
			pairwise_regularizer_type=None, pairwise_regularizer_const=1e-6,
			nn_regularizer_type=None, nn_regularizer_const=1e-6,
			name='deepbind'):
	# set up the regularizer
	kernel_regularizer = get_kernel_regularizer(conv_type, regularizer_type, regularizer_const, 
												pairwise_regularizer_type, pairwise_regularizer_const, 
												nn_regularizer_type, nn_regularizer_const)

	# pick the right convolutional layer type 
	assert conv_type.lower() in ['pairwise', 'regular', 'nearest_neighbor']
	Conv1D = conv_type_dict[conv_type]

	#### setup the model  ####
	x = tfk.layers.Input(shape=(L, A))
	y = Conv1D(
		filters=num_filters, 
		kernel_size=filter_size,
		use_bias=True,
		kernel_regularizer=kernel_regularizer,
			)(x)
	y = tfk.layers.Activation('relu')(y)  # rectification
	y = tfk.layers.Lambda(lambda x : tf.reduce_max(x, axis=1))(y)  # max pooling
	y = tfk.layers.Dropout(dropout)(y)  # dropout
	fcnn = tfk.Sequential(
							[
								tfk.layers.Dense(32, use_bias=use_bias,activation=activation),
								tfk.layers.Dense(1, use_bias=use_bias,activation=tfk.activations.sigmoid),
							]
						 )
	y = fcnn(y)


	# compile the model
	model = tfk.Model(inputs=x, outputs=y,name=name+"_"+conv_type)

	return model






