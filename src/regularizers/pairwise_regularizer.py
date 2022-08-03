import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras.regularizers import l1, l2
import tensorflow_probability as tfp
import numpy as np
try:
    fill_triangular = tfp.math.fill_triangular
except:
    fill_triangular = tfp.distributions.fill_triangular


__all__ = ['PairwiseKernelRegularizer']

TF2_SUBVERSION = int(tf.__version__.split('.')[1])

class PairwiseKernelRegularizer(tfk.regularizers.Regularizer):
    """
    A regularizer than applies separate regularization functions on
    the diagonal and off-diagonal terms in the pairwise kernel.
    """
    def __init__(self, diag_regularizer, offdiag_regularizer, *args, **kwargs):
        """
        diag_regularizer <keras.regularizer.Regularizer> - The 
        offdiag_regularizer <keras.regularizer.Regularizer>
        """
        super().__init__(*args, **kwargs)
        self.diag_regularizer = diag_regularizer
        self.offdiag_regularizer = offdiag_regularizer
    
    def __call__(self, x):
        """
        x -> Pairwise kernel (expected shape = (numterms, A, C))
        """
        ndims = len(x.shape) 
        perm = list(np.arange(1, ndims)) + [0]
        x = tf.transpose(x, perm)  ## move 1st dimension to the end
        x = fill_triangular(x)
        
        diag_part = tf.linalg.diag_part(x)  ##(A, C, P)
        offdiag_part = x - tf.linalg.diag(diag_part) ## (A, C, P, P)
        
        res1 =  self.diag_regularizer(diag_part) 
        res2 = self.offdiag_regularizer(offdiag_part)
        return res1+res2

    def get_config(self):
        config = {}
        config['diag_regularizer'] = self.diag_regularizer
        config['offdiag_regularizer'] = self.offdiag_regularizer
        return config

