import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.framework import tensor_shape
import tensorflow_probability as tfp
from pdb import set_trace as keyboard
try:
    fill_triangular = tfp.math.fill_triangular
except:
    fill_triangular = tfp.distributions.fill_triangular

__all__ = ['PairwiseConv1D']
TF2_SUBVERSION = int(tf.__version__.split('.')[1])

class PairwiseConv1D(tfk.layers.Conv1D):
	"""
	This class implement pairwise convolutions
	for 1D signals.  

	Standard convolutions implemented as `keras.layers.Conv1D`
	perform linear transformations of patches from the input 
	signal. 
	Pairwise convolutions perform linear transformations of all
	pairwise terms from entries in all patches of the input signal.

	The implementation is achieved by taking an outer product of 
	each patch and performing a linear transformation of the pairwise
	(i.e. lower diagonal) terms. 

	The rest of this docstring is copied from the keras.layers.Conv1D 
	docstring.


	"""
	__doc__ = __doc__ + super.__doc__

	padding_map_dict = {'same':'SAME', 'valid':'VALID'}

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)


	def _get_patches(self, x):
		"""
		x -> (N, L, A)

		This module is tested for correctness. 
		"""
		assert len(x.shape) == 3

		x = tf.expand_dims(x, axis=2) ## (N, L, 1, A)
		sizes = [1, self.kernel_size[0], 1, 1]
		strides = [1, self.strides[0], 1, 1]
		rates = [1, self.dilation_rate[0], 1, 1]
		padding = self.padding_map_dict[self.padding]
		xp = tf.image.extract_patches(x,sizes=sizes,strides=strides,rates=rates,padding=padding)
		xp = tf.squeeze(xp, axis=2)  ## (N, num patches, flattened patch size)
		return xp

	def _outer_product(self, xpatches):
		"""
		xpatches -> (N, numpatches, patch size*A)

		RETURNS:
		xout -> (N, num patches, patch size*A, patch_size*A)
		"""
		res = tf.einsum("ijk, ijl -> ijkl", xpatches, xpatches)  ## (N, numpatches, P*A, P*A)
		res = tf.linalg.set_diag(res, diagonal=xpatches) ## replace the sq. term with unit power in the diag.
		return res

	@property 
	def full_kernel(self):
		k = tf.transpose(self.kernel, [1,0])  ## (C, numweights,)
		k = fill_triangular(k) ## (C, P*A, P*A)
		k = tf.transpose(k, [1, 2, 0])  ## (P*A, P*A, C)
		return k

	@property 
	def diag_kernel(self):
		"""
		Returns the diagonal of the kernel.
		"""
		k = self.full_kernel  ## (P*A, P*A, C)
		k = tf.transpose(k, [2, 0, 1])  ## (C, P*A, P*A)
		k = tf.linalg.diag_part(k) ## (C, P*A)
		k = tf.transpose(k, [1, 0]) ## (P*A, C)
		return k

	def build(self, input_shape):
		"""
		Expected input_shape is (N, L, A)
		"""
		input_shape = tensor_shape.TensorShape(input_shape)  #(L, A)
		A = input_shape[-1]  # A		
		P = self.kernel_size[0] # P
		flat_patch_size = P*A
		kernel_shape = [ int(flat_patch_size*(flat_patch_size+1)*0.5), self.filters ] ## (numweights, C)

		# add the kernel
		self.kernel = self.add_weight(
								name='kernel',
								shape=kernel_shape,
								initializer=self.kernel_initializer,
								regularizer=self.kernel_regularizer, 
								constraint=self.kernel_constraint, 
								trainable=True, 
								dtype=self.dtype,
										)

		# add the bias
		if self.use_bias:
			self.bias = self.add_weight(
								name = 'bias',
								shape = (self.filters,),
								initializer=self.bias_initializer,
								regularizer=self.bias_regularizer,
								constraint=self.bias_constraint,
								trainable=True,
								dtype=self.dtype
										)
		else:
			self.bias = None
		self.built = True

	def call(self, inputs):
		"""
		inputs -> (N, L, in_channels)

		RETURNS:
		outputs -> (N, L, out_channels)
		"""

		xp = self._get_patches(inputs) ## (N, numpatches, P*A)

		# take the outer product 
		xout = self._outer_product(xp) ## (N, numpatches, P*A, P*A)

		# compute the output 
		kern = self.full_kernel  ## (P*A, P*A, C)
		outputs = tf.einsum("ijkl, klm -> ijm",xout, kern)

		# add the bias
		if self.use_bias:
			outputs = outputs + self.bias

		# apply activation function 
		if self.activation is not None:
			outputs = self.activation(outputs)
		return outputs