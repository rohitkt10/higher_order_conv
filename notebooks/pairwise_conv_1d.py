import inspect
import tensorflow as tf
from tensorflow import keras as tfk
from ho_conv import HigherOrderConv1D
from ho_regularizer import HigherOrderKernelRegularizer
import tensorflow_probability as tfp
try:
    fill_triangular = tfp.math.fill_triangular
except:
    fill_triangular = tfp.distributions.fill_triangular

__all__ = ['PairwiseConv1D', 'PairwiseFromStdConv1D']

class PairwiseConv1D(HigherOrderConv1D):
	"""
	Pairwise convolution on 1D signals. 
	"""
	def build(self, input_shape):
		A = input_shape[-1]
		P = self.kernel_size[0]
		C = self.filters
		flat_size = A*P
		if isinstance(self.kernel_regularizer, HigherOrderKernelRegularizer):
			diag_regularizer=self.kernel_regularizer.diag_regularizer
			offdiag_regularizer=self.kernel_regularizer.offdiag_regularizer
		else:
			diag_regularizer=self.kernel_regularizer 
			offdiag_regularizer=self.kernel_regularizer

		# set up the diagonal kernel 
		diag_shape = [flat_size, C]
		self.diag_kernel = self.add_weight(
									name="diag_kernel",
									shape=diag_shape,
									regularizer=diag_regularizer,
									initializer=self.kernel_initializer, 
									constraint=self.kernel_constraint,
									trainable=True, 
									dtype=self.dtype
										)

		# set up the offdiagonal kernel 
		num_rows = P - 1 
		block_size = A*A
		num_blocks = int(0.5 * num_rows *(num_rows+1))
		offdiag_shape = [num_blocks*block_size, C]
		self.offdiag_kernel = self.add_weight(
									name="offdiag_kernel", 
									shape=offdiag_shape,
									regularizer=offdiag_regularizer,
									initializer=self.kernel_initializer,
									constraint=self.kernel_constraint,
									trainable=True,
									dtype=self.dtype
										)

		# set up the bias 
		if self.use_bias:
			self.bias = self.add_weight(
										name="bias", 
										shape=(self.filters,),
										initializer=self.bias_initializer, 
										constraint=self.bias_constraint, 
										regularizer=self.bias_regularizer, 
										trainable=True, 
										dtype=self.dtype
										)
		else:
			self.bias = None
		self.built = True

	def _get_diag_kernel(self):
		diag = tf.transpose(self.diag_kernel, [1, 0]) ## (C, P*A)
		diag = tf.linalg.diag(diag) ## (C, P*A, P*A)
		return diag

	def _get_offdiag_kernel(self):
		P = self.kernel_size[0]
		flat_size = self.diag_kernel.shape[0] ## P*A
		A = int(flat_size/P)
		offdiag = tf.transpose(self.offdiag_kernel, [1, 0]) ## (C, num_offdiag)
		num_rows = P-1
		block_size = A*A
		C = self.filters 

		# get the rows (upto 2nd last row)
		rows = []
		start_idx = 0
		for num_blocks in range(num_rows, 0, -1):
			num_zero_blocks = P - num_blocks
			zeros = tf.zeros((C, A, A*num_zero_blocks), dtype=offdiag.dtype)
			num_elements = num_blocks * block_size

			# get the elements of the current row 
			row = offdiag[:, start_idx:start_idx+num_elements] ## (C, num_row_elements)
			row = tf.reshape(row, (C, A, num_blocks*A)) ## (C, A, numblock*A)
			row = tf.concat([zeros, row], 2) ## (C, A, P*A)
			start_idx = start_idx + num_elements
			rows.append(row)

		# get the last row 
		rows.append(tf.zeros((C, A, P*A))) ## (C, A, P*A)

		rows = tf.concat(rows, axis=1) ## (C, P*A, P*A)
		return rows

	@property
	def kernel(self):
		diag = self._get_diag_kernel() ## (C, P*A, P*A)
		offdiag = self._get_offdiag_kernel() ## (C, P*A, P*A)
		kern = diag+offdiag ## (C, P*A, P*A)
		kern = tf.transpose(kern, [1, 2, 0])
		return kern

class PairwiseFromStdConv1D(PairwiseConv1D):
	def __init__(self, 
				stdconv, 
				offdiag_regularizer=None, 
				offdiag_initializer="zeros", 
				offdiag_constraint=None):
		assert isinstance(stdconv, tfk.layers.Conv1D), 'stdconv has to be an instance keras.layers.Conv1D.'
		argspec = inspect.getfullargspec(stdconv.__init__)
		args = argspec.args[1:]
		super_kwargs = {}
		for arg in args:
			super_kwargs[arg] = eval("stdconv.%s"%arg)
		super().__init__(**super_kwargs)

		# set up the diagonal kernel 
		self.A = stdconv.kernel.shape[1]
		diag_kern = tf.convert_to_tensor(stdconv.kernel.numpy(), dtype=self.dtype)
		diag_kern = tf.reshape(diag_kern, (-1, self.filters))
		self.diag_kernel = tf.Variable(name='diag_kernel',
									initial_value=diag_kern,
									dtype=self.dtype,
									trainable=False,
										)

		# set up the bias 
		if self.use_bias:
			self.bias = tf.Variable(initial_value=stdconv.bias, dtype=self.dtype, trainable=False, name='bias')
		else:
			self.bias = None

		# add the remaining initializer args 
		self.offdiag_regularizer=offdiag_regularizer
		self.offdiag_initializer=offdiag_initializer 
		self.offdiag_constraint=offdiag_constraint

	def build(self, input_shape):
		A = input_shape[-1]
		assert A == self.A, 'Invalid shape.'
		P = self.kernel_size[0]
		flat_size = A*P
		C = self.filters 

		# set up the offdiagonal kernel 
		num_rows = P - 1 
		block_size = A*A
		num_blocks = int(0.5 * num_rows *(num_rows+1))
		offdiag_shape = [num_blocks*block_size, C]
		self.offdiag_kernel = self.add_weight(
										name="offdiag_kernel", 
										shape=offdiag_shape,
										regularizer=self.offdiag_regularizer,
										initializer=self.offdiag_initializer,
										constraint=self.offdiag_constraint,
										trainable=True,
										dtype=self.dtype
											)
		self.built = True

# class PairwiseConv1D(HigherOrderConv1D):
# 	"""
# 	Pairwise convolution on 1D signals. 
# 	"""
# 	def build(self, input_shape):
# 		A = input_shape[-1]
# 		P = self.kernel_size[0]
# 		flat_size = A*P
# 		if isinstance(self.kernel_regularizer, HigherOrderKernelRegularizer):
# 			diag_regularizer=self.kernel_regularizer.diag_regularizer
# 			offdiag_regularizer=self.kernel_regularizer.offdiag_regularizer
# 		else:
# 			diag_regularizer=self.kernel_regularizer 
# 			offdiag_regularizer=self.kernel_regularizer

# 		# set up the diagonal kernel 
# 		diag_shape = [flat_size, self.filters]
# 		self.diag_kernel = self.add_weight(
# 									name="diag_kernel",
# 									shape=diag_shape,
# 									regularizer=diag_regularizer,
# 									initializer=self.kernel_initializer, 
# 									constraint=self.kernel_constraint,
# 									trainable=True, 
# 									dtype=self.dtype
# 											)

# 		# set up the offdiagonal kernel
# 		offdiag_shape= [int(flat_size*(flat_size-1)*0.5), self.filters]
# 		self.offdiag_kernel = self.add_weight(
# 									name="offdiag_kernel", 
# 									shape=offdiag_shape,
# 									regularizer=offdiag_regularizer,
# 									initializer=self.kernel_initializer,
# 									constraint=self.kernel_constraint,
# 									trainable=True,
# 									dtype=self.dtype
# 											)

# 		# set up the bias 
# 		if self.use_bias:
# 			self.bias = self.add_weight(
# 									name="bias", 
# 									shape=(self.filters,),
# 									initializer=self.bias_initializer, 
# 									constraint=self.bias_constraint, 
# 									regularizer=self.bias_regularizer, 
# 									trainable=True, 
# 									dtype=self.dtype
# 										)
# 		else:
# 			self.bias = None
# 		self.built = True

# 	def _get_diag_kernel(self):
# 		diag = tf.transpose(self.diag_kernel, [1, 0]) ## (C, P*A)
# 		diag = tf.linalg.diag(diag) ## (C, P*A, P*A)
# 		return diag

# 	def _get_offdiag_kernel(self):
# 		flat_size = self.diag_kernel.shape[0]
# 		offdiag = tf.transpose(self.offdiag_kernel, [1, 0]) ## (C, num_offidiag)
# 		offdiag = fill_triangular(offdiag) ## (C, P*A-1, P*A-1)
# 		left_add = tf.zeros((self.filters, flat_size-1, 1), dtype=offdiag.dtype)  ##(C, P*A-1, 1)
# 		offdiag = tf.concat([offdiag, left_add], axis=2) ## (C, P*A-1, P*A)
# 		top_add = tf.zeros((self.filters, 1, flat_size), dtype=offdiag.dtype) ## (C, 1, P*A)
# 		offdiag = tf.concat([top_add, offdiag], axis=1) ## (C, P*A, P*A)
# 		return offdiag

# 	@property
# 	def kernel(self):
# 		diag = self._get_diag_kernel() ## (C, P*A, P*A)
# 		offdiag = self._get_offdiag_kernel() ## (C, P*A, P*A)
# 		kern = diag+offdiag ## (C, P*A, P*A)
# 		kern = tf.transpose(kern, [1, 2, 0])
# 		return kern

# class PairwiseFromStdConv1D(PairwiseConv1D):
# 	def __init__(self, 
# 				stdconv, 
# 				offdiag_regularizer=None, 
# 				offdiag_initializer="glorot_uniform", 
# 				offdiag_constraint=None):
# 		assert isinstance(stdconv, tfk.layers.Conv1D), 'stdconv has to be an instance keras.layers.Conv1D.'
# 		argspec = inspect.getfullargspec(stdconv.__init__)
# 		args = argspec.args[1:]
# 		super_kwargs = {}
# 		for arg in args:
# 			super_kwargs[arg] = eval("stdconv.%s"%arg)
# 		super().__init__(**super_kwargs)

# 		# set up the diagonal kernel 
# 		self.A = stdconv.kernel.shape[1]
# 		diag_kern = tf.convert_to_tensor(stdconv.kernel.numpy(), dtype=self.dtype)
# 		diag_kern = tf.reshape(diag_kern, (-1, self.filters))
# 		self.diag_kernel = tf.Variable(name='diag_kernel',
# 									initial_value=diag_kern,
# 									dtype=self.dtype,
# 									trainable=False,
# 									)
		
# 		# set up the bias 
# 		if self.use_bias:
# 			self.bias = tf.Variable(initial_value=stdconv.bias, dtype=self.dtype, trainable=False, name='bias')
# 		else:
# 			self.bias = None
		
# 		# add the remaining initializer args 
# 		self.offdiag_regularizer=offdiag_regularizer
# 		self.offdiag_initializer=offdiag_initializer 
# 		self.offdiag_constraint=offdiag_constraint

# 	def build(self, input_shape):
# 		A = input_shape[-1]
# 		assert A == self.A, 'Invalid shape.'
# 		P = self.kernel_size[0]
# 		flat_size = A*P

# 		# set up the off diagonal kernel 
# 		offdiag_shape = [ int(flat_size*(flat_size-1)*0.5), self.filters ]
# 		self.offdiag_kernel = self.add_weight(
# 											name='offdiag_kernel',
# 											shape=offdiag_shape,
# 											regularizer=self.offdiag_regularizer,
# 											initializer=self.offdiag_initializer,
# 											constraint=self.offdiag_constraint, 
# 											trainable=True, 
# 											dtype=self.dtype,
# 											)
# 		self.built = True