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

__all__ = ['NearestNeighborConv1D']
TF2_SUBVERSION = int(tf.__version__.split('.')[1])


class NearestNeighborConv1D(tfk.layers.Conv1D):
    """
    This class implement nearest neighbor convolutions
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
    
    def build(self, input_shape):
        """
        Expected input_shape is (N, L, A)
        """
        input_shape = tensor_shape.TensorShape(input_shape)  #(L, A)
        A = input_shape[-1]  # A		
        P = self.kernel_size[0] # P
        flat_patch_size = P*A
        #numweights = P*int(0.5*A*(A+1)) + (P-1)*A*A  
        #kernel_shape = [ numweights, self.filters ] ## (numweights, C)

        # define upper diag block kernels ; we need P of these
        self.diag_kernels = []
        numweights = int(0.5*A*(A+1)) ## num wieghts per diag block
        for i in range(P):
            kernel = self.add_weight(
                                name='diag_kernel_%d'%(i+1),
                                shape=(numweights, self.filters),
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer, 
                                constraint=self.kernel_constraint, 
                                trainable=True, 
                                dtype=self.dtype,
                                    )
            self.diag_kernels.append(kernel)

        self.nn_kernels = []
        for i in range(P-1):
            kernel = self.add_weight(
                                name='nn_kernel_%d'%(i+1),
                                shape=(A, A, self.filters),
                                initializer=self.kernel_initializer,
                                regularizer=self.kernel_regularizer, 
                                constraint=self.kernel_constraint, 
                                trainable=True, 
                                dtype=self.dtype,
                                    )
            self.nn_kernels.append(kernel)
        
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
        kern = self.kernel  ## (P*A, P*A, C)
        outputs = tf.einsum("ijkl, klm -> ijm",xout, kern)

        # add the bias
        if self.use_bias:
            outputs = outputs + self.bias

        # apply activation function 
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
    
    def _assemble_kernel(self,w_diags, ws):
        """
        Assemble the full 
        w_diags -> List of diagonal kernels each block of shape (C, A,A) 
        ws -> List of NN block kernels each block of shape (C, A, A)
        """
        P, C, A = len(w_diags), w_diags[0].shape[0], w_diags[0].shape[1]
        rows = []
        for rownum in range(P-1):
            row = []
            for j in range(rownum):
                row.append(tf.zeros((C, A, A,)))
            row = row + [w_diags[rownum], ws[rownum]]
            for k in range(rownum+2, P):
                row.append(tf.zeros((C, A, A,)))
            row = tf.concat(row, axis=2)
            rows.append(row)
        row=tf.concat( [tf.zeros((C, A,A*(P-1),)), w_diags[-1]], axis=2 )
        rows.append(row)
        kern = tf.concat( rows, axis=1 )
        return kern
    
    @property 
    def kernel(self):
        diag_kernels = [tf.transpose(k, [1, 0]) for k in self.diag_kernels] ## each block (C, numweights)
        diag_kernels = [fill_triangular(k, upper=True) for k in diag_kernels] ## each block (C, A, A)
        nn_kernels = [tf.transpose(k, [2, 0, 1]) for k in self.nn_kernels] ## each block (C, A, A)
        k = self._assemble_kernel(diag_kernels, nn_kernels)
        k = tf.transpose(k, [1, 2, 0])
        return k
