import tensorflow as tf 
from tensorflow import keras as tfk 

from ..layers import PairwiseConv1D
from ..regularizers import PairwiseKernelRegularizer

__all__ =['get_model']

conv_type_dict = {
    'regular':tfk.layers.Conv1D,
    'pairwise':PairwiseConv1D
}


def get_model(L, A, conv_type='regular', activation='relu', regularizer_const=1e-6, pairwise_regularizer_type=None, pairwise_regularizer_const=1e-6, name='residual_bind'):
	## input layer 
	x = tfk.layers.Input((L, A), name='Input')

	# layer  1
	kernel_regularizer = get_kernel_regularizer(regularizer_const, conv_type, pairwise_regularizer_type, pairwise_regularizer_const)
	y = conv_layer(x, conv_type=conv_type, num_filters=24, kernel_size=19, padding='same', kernel_regularizer=kernel_regularizer, activation=activation, dropout=0.1, bn=True)

	# layer 2 
	kernel_regularizer = tfk.regularizers.l2(regularizer_const)
	y = residual_block(y, filter_size=5, activation=activation, kernel_regularizer=kernel_regularizer)
	y = tfk.layers.MaxPool1D(pool_size=10)(y)

	# layer 3 
	kernel_regularizer = tfk.regularizers.l2(regularizer_const)
	y = conv_layer(y, conv_type='regular', num_filters=48, kernel_size=7, padding='same', kernel_regularizer=kernel_regularizer, activation=activation, dropout=0.3, bn=True)
	y = tfk.layers.MaxPool1D(pool_size=5)(y)

	# layer 4
	kernel_regularizer = tfk.regularizers.l2(regularizer_const)
	y = conv_layer(y, conv_type='regular', num_filters=64, kernel_size=3, padding='same', kernel_regularizer=kernel_regularizer, activation=activation, dropout=0.4, bn=True)
	y = tfk.layers.MaxPool1D(pool_size=4)(y)

	# layer 5
	y = tfk.layers.Flatten()(y)
	kernel_regularizer = tfk.regularizers.l2(regularizer_const)
	y = dense_layer(y, num_units=96, activation=activation, dropout=0.5, kernel_regularizer=kernel_regularizer, bn=True)

	# Output layer 
	y = tfk.layers.Dense(1, name='logits')(y)
	y = tfk.layers.Activation('sigmoid')(y)

	# compile model and return 
	model = tfk.Model(inputs=x, outputs=y, name=name+"_"+conv_type)
	return model

def conv_layer(x, conv_type, num_filters, kernel_size, padding='same', activation='relu', dropout=0.2, kernel_regularizer=tfk.regularizers.l2(1e-6), bn=True, kernel_initializer=None):
	"""Implements a conv -> BN -> activation -> Dropout block."""

	# pick the right convolutional layer type 
	assert conv_type.lower() in ['pairwise', 'regular']
	if conv_type == 'pairwise':
		Conv1D = PairwiseConv1D
	else:
		Conv1D = tfk.layers.Conv1D

	y = Conv1D(filters=num_filters,kernel_size=kernel_size, use_bias=False, 
			padding=padding, kernel_initializer=kernel_initializer, 
			kernel_regularizer=kernel_regularizer)(x)
	if bn:                      
		y = tfk.layers.BatchNormalization()(y)
	y = tfk.layers.Activation(activation)(y)
	if dropout:
		y = tfk.layers.Dropout(dropout)(y)
	return y

def dense_layer(x, num_units, activation, dropout=0.5, kernel_regularizer=tfk.regularizers.l2(1e-6), kernel_initializer=None, bn=True):
	"""Implements a dense -> BN -> activation -> Dropout block."""
	y = tfk.layers.Dense(num_units, use_bias=False,  kernel_initializer=kernel_initializer, bias_initializer='zeros', kernel_regularizer=kernel_regularizer)(x)
	if bn:
		y = tfk.layers.BatchNormalization()(y)
	y = tfk.layers.Activation(activation)(y)
	if dropout:
		y = tfk.layers.Dropout(dropout)(y)
	return y

def residual_block(x, filter_size, activation='relu', kernel_regularizer=tfk.regularizers.l2(1e-6)):
	"""
	Implements a residual block of the form y = g ( x + F(x) ), 
	where, F(x) is "conv -> BN -> Actfn -> Conv -> BN" 
	and g(.) is a final activation. 
	"""
	num_filters = x.shape.as_list()[-1]
	y = tfk.layers.Conv1D(filters=num_filters, kernel_size=filter_size, activation=None, use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(x) 
	y = tfk.layers.BatchNormalization()(y)
	y = tfk.layers.Activation(activation)(y)
	y = tfk.layers.Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu',use_bias=False, padding='same', kernel_initializer='he_normal', kernel_regularizer=kernel_regularizer)(y) 
	y = tfk.layers.BatchNormalization()(y)

	# residual connection 
	y = tfk.layers.add([x, y])
	y = tfk.layers.Activation(activation)(y)
	return y

def get_kernel_regularizer(diag_regularizer_const=1e-6, conv_type='regular', pairwise_regularizer_type=None, pairwise_regularizer_const=1e-6):
	kernel_regularizer = tfk.regularizers.l2(diag_regularizer_const)
	if conv_type == 'pairwise':
		if pairwise_regularizer_type is not None:
			assert pairwise_regularizer_type.lower() in ['l2', 'l1']
			pairwise_reg = eval('tfk.regularizers.'+pairwise_regularizer_type)(pairwise_regularizer_const)
		else:
			pairwise_reg = kernel_regularizer
		kernel_regularizer = PairwiseKernelRegularizer(kernel_regularizer, pairwise_reg)
	return kernel_regularizer