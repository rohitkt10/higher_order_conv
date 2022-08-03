import tensorflow as tf
from tensorflow import keras as tfk
from .pairwise_conv1d import PairwiseConv1D
from ..utils import get_activation

conv_type_dict = {
    'regular':tfk.layers.Conv1D,
    'pairwise':PairwiseConv1D
}

pool_type_dict = {
	'max':tfk.layers.MaxPooling1D,
	'avg':tfk.layers.AveragePooling1D
}

__all_= ['conv1d_block', 'dense_block']

def conv1d_block(
			filters, 
			kernel_size,
			use_bias,
			padding='same',
			conv_type='regular',
			kernel_initializer='glorot_uniform',
			bias_initializer='zeros',
			kernel_regularizer=None,
			bias_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			strides=1,
			dilation_rate=1,
			groups=1,
			activation='relu',
			batch_norm=True,
			dropout=None,
			pool_size=None,
			pool_type='max',
			name='conv_block'
		):
	"""
	conv1d -> batch norm -> activation -> dropout. 

	- Pick either `max` pooling or `avg` as `pool_type`. 
	"""
	assert conv_type in conv_type_dict.keys(), \
						"`conv_type` must be one of: "+str(list(conv_type_dict.keys()))
	
	layers = []
	Conv1D = conv_type_dict[conv_type]
	conv_layer = Conv1D(filters=filters,
						kernel_size=kernel_size,
						padding=padding,
						use_bias=use_bias,
						kernel_initializer=kernel_initializer,
						bias_initializer=bias_initializer,
						kernel_regularizer=kernel_regularizer,
						bias_regularizer=bias_regularizer,
						activity_regularizer=activity_regularizer,
						kernel_constraint=kernel_constraint,
						bias_constraint=bias_constraint,
						strides=strides,
						dilation_rate=dilation_rate,
						groups=groups)
	layers.append(conv_layer)

	# batch normalization
	if batch_norm:
		layers.append(tfk.layers.BatchNormalization())

	# activation function
	layers.append(tfk.layers.Activation(get_activation(activation)))

	# dropout 
	if dropout:
		layers.append(tfk.layers.Dropout(dropout))

	# max pooling
	if pool_size:
		if pool_type not in pool_type_dict.keys():
			raise ValueError("You may pick from one of \
							these pool types: "+str(list(pool_type_dict.keys())))
		Pool1D = pool_type_dict[pool_type]
		layers.append((Pool1D(pool_size=pool_size)))


	# compose all layers
	out = tfk.Sequential(layers=layers, name=name)

	return out

def dense_block(
			units, 
			use_bias,
			kernel_initializer='glorot_uniform',
			bias_initializer='zeros',
			kernel_regularizer=None,
			bias_regularizer=None,
			activity_regularizer=None,
			kernel_constraint=None,
			bias_constraint=None,
			activation='relu',
			batch_norm=True,
			dropout=None,
			flatten=False,
			name='dense_block'
		):
	"""
	flatten -> dense -> batch norm -> activation -> dropout
	"""
	layers = []

	# flatten the inputs 
	if flatten:
		layers.append(tfk.layers.Flatten())

	# linear layer 
	dense_layer = tfk.layers.Dense(
								units=units,
								use_bias=use_bias,
								kernel_initializer=kernel_initializer,
								bias_initializer=bias_initializer,
								kernel_regularizer=kernel_regularizer,
								bias_regularizer=bias_regularizer,
								activity_regularizer=activity_regularizer,
								kernel_constraint=kernel_constraint,
								bias_constraint=bias_constraint
							)
	layers.append(dense_layer)

	# batch normalization 
	if batch_norm:
		layers.append(tfk.layers.BatchNormalization())

	# activation function 
	layers.append(tfk.layers.Activation(get_activation(activation)))

	# dropout
	if dropout:
		layers.append(tfk.layers.Dropout(dropout))

	# compose all layers 
	out = tfk.Sequential(layers=layers, name=name)

	return out