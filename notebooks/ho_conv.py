import tensorflow as tf
from tensorflow import keras as tfk
from ho_regularizer import HigherOrderKernelRegularizer

__all__ = ['HigherOrderConv1D',]

class HigherOrderConv1D(tfk.layers.Conv1D):
	"""
	Base-class for higher-order convolutional layer. 
	"""
	padding_map_dict = {'same':'SAME', 'valid':'VALID'}
	def _get_patches(self, x):
		assert len(x.shape) == 3
		x = tf.expand_dims(x, axis=2) ## (N, L, 1, A)
		sizes = [1, self.kernel_size[0], 1, 1]
		strides = [1, self.strides[0], 1, 1]
		rates = [1, self.dilation_rate[0], 1, 1]
		padding = self.padding_map_dict[self.padding]
		xp = tf.image.extract_patches(x,sizes=sizes,strides=strides,rates=rates,padding=padding)
		xp = tf.squeeze(xp, axis=2)  ## (N, num patches, P*A)
		return xp

	def _outer_product(self, xpatches):
		res = tf.einsum("ijk, ijl -> ijkl", xpatches, xpatches)  ## (N, numpatches, P*A, P*A)
		res = tf.linalg.set_diag(res, diagonal=xpatches) ## replace the sq. term with unit power in the diag.
		return res

	@property
	def kernel(self):
		raise NotImplementedError("Subclasses of HigherOrderConv1D should implement this.")

	def build(self, input_shape):
		raise NotImplementedError("Subclasses of HigherOrderConv1D should implement this.")

	def call(self, x):
		# get patches and compute outer product of patches 
		xp = self._get_patches(x)
		xout = self._outer_product(xp)  

		# compute the output 
		kern = self.kernel
		res = tf.einsum("ijkl, klm -> ijm", xout, kern)
		if self.use_bias:
			res = res + self.bias
		if self.activation is not None:
			res = self.activation(res)

		return res