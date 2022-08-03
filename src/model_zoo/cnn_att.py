import tensorflow as tf 
from tensorflow import keras as tfk 

from ..layers import PairwiseConv1D, NearestNeighborConv1D
from ..regularizers import PairwiseKernelRegularizer, NearestNeighborKernelRegularizer
from src.layers import MultiHeadAttention
keras = tfk
layers = tfk.layers

__all__ =['get_model']

conv_type_dict = {
    'regular':tfk.layers.Conv1D,
    'pairwise':PairwiseConv1D,
    'nearest_neighbor':NearestNeighborConv1D
}

def get_model(L, A, conv_type='regular', activation='relu', pool_size=4, 
				regularizer_type='l2', regularizer_const=1e-6, 
				pairwise_regularizer_type=None, pairwise_regularizer_const=1e-6, 
				nn_regularizer_type=None, nn_regularizer_const=1e-6, 
				name='cnn_att'):
	
	# pick the right convolutional layer type and the kernel regularizer for the 1st layer.
	assert conv_type.lower() in ['pairwise', 'regular', 'nearest_neighbor']
	Conv1D = conv_type_dict[conv_type]
	kernel_regularizer = get_kernel_regularizer(conv_type, regularizer_type, regularizer_const, 
												pairwise_regularizer_type, pairwise_regularizer_const, 
												nn_regularizer_type, nn_regularizer_const)

	## input layer and 1st convolution layer 
	x = tfk.layers.Input((L, A), name='Input')
	y = Conv1D(filters=32, kernel_size=19, kernel_regularizer=kernel_regularizer, padding='same', name='conv1', use_bias=True)(x)

	# remaining layers
	y = keras.layers.Activation('relu')(y)
	y = keras.layers.MaxPool1D(pool_size=pool_size)(y)
	embedding = keras.layers.Dropout(0.1)(y)
	y, weights = MultiHeadAttention(num_heads=8, key_dim=64, value_dim=64)(embedding, embedding, return_attention_scores=True)
	y = keras.layers.Dropout(0.1)(y)
	y = keras.layers.LayerNormalization(epsilon=1e-6)(y)
	y = keras.layers.Flatten()(y)
	y = keras.layers.Dense(128, activation=None, use_bias=False)(y)
	y = keras.layers.BatchNormalization()(y)
	y = keras.layers.Activation('relu')(y)
	y = keras.layers.Dropout(0.5)(y)
	y = keras.layers.Dense(1, name='logits')(y)
	y = keras.layers.Activation('sigmoid', name='output')(y)

	# assembled model
	model = tfk.Model(inputs=x, outputs=y, name=name+"_"+conv_type)
	return model

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