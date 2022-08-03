import tensorflow as tf
from tensorflow import keras as tfk

from ..utils import get_activation
from ..layers import PairwiseConv1D
from ..layers import conv1d_block, dense_block

__all__ =['get_model']

conv_type_dict = {
    'regular':tfk.layers.Conv1D,
    'pairwise':PairwiseConv1D
}


def get_model(
			conv_type='regular', 
			L=200,
			A=4,
			num_filters_1=300,
			num_filters_2=200,
			num_filters_3=200,
			regularizer_const=1e-6,
			num_outputs=1,
			name='basset',
		):
	# input layer
	x = tfk.layers.Input(shape=(L, A))

	# first convolutional block 
	conv_block1 = conv1d_block(
	                      filters=num_filters_1, 
	                      kernel_size=19,
	                      padding='same', 
	                      conv_type=conv_type,
	                      use_bias=False,
	                      dropout=0.5,
	                      activation='relu',
	                      kernel_regularizer=tfk.regularizers.l2(regularizer_const),
	                      batch_norm=True,
	                      pool_size=3,
	                      name='conv_block_1',
	                         )
	y = conv_block1(x)

	# second convolutional block 
	conv_block2 = conv1d_block(
	                      filters=num_filters_2, 
	                      kernel_size=11,
	                      padding='same', 
	                      conv_type='regular',
	                      use_bias=False,
	                      dropout=0.5,
	                      activation='relu',
	                      kernel_regularizer=tfk.regularizers.l2(regularizer_const),
	                      batch_norm=True,
	                      pool_size=4,
	                      name='conv_block_2',
	                      	 )
	y = conv_block2(y)

	# third convolutional block 
	conv_block3 = conv1d_block(
	                      filters=num_filters_3, 
	                      kernel_size=7,
	                      padding='same', 
	                      conv_type='regular',
	                      use_bias=False,
	                      dropout=0.5,
	                      activation='relu',
	                      kernel_regularizer=tfk.regularizers.l2(regularizer_const),
	                      batch_norm=True,
	                      pool_size=4,
	                      name='conv_block_3',
	                      	 )
	y = conv_block3(y)

	# fully connected layer 1
	dense_block1 = dense_block(
	                        units=1000,
	                        use_bias=True,
	                        dropout=0.5,
	                        batch_norm=False,
	                        kernel_regularizer=tfk.regularizers.l2(regularizer_const),
	                        flatten=True,
	                        activation='relu',
	                        name='dense_block_1'
	                    )
	y = dense_block1(y)

	# fully connected layer 2
	dense_block2 = dense_block(
	                        units=1000,
	                        use_bias=True,
	                        dropout=0.5,
	                        batch_norm=False,
	                        kernel_regularizer=tfk.regularizers.l2(regularizer_const),
	                        activation='relu',
	                        name='dense_block_2'
	                    )
	y = dense_block2(y)

	# output layer 
	output_layer = tfk.layers.Dense(num_outputs, activation=output_activation, name='output_layer')
	y = output_layer(y)

	# compile the model 
	model = tfk.Model(inputs=x, outputs=y, name=name)

	return model





