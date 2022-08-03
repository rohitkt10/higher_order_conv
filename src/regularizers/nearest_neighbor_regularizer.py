import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras.regularizers import l1, l2
import tensorflow_probability as tfp
import numpy as np
from pdb import set_trace as keyboard
try:
    fill_triangular = tfp.math.fill_triangular
except:
    fill_triangular = tfp.distributions.fill_triangular


__all__ = ['NearestNeighborKernelRegularizer']

class NearestNeighborKernelRegularizer(tfk.regularizers.Regularizer):
    """
    A regularizer than applies separate regularization functions on
    the diagonal and off-diagonal terms in the nearest neighbor kernel.
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
        if 'nn_kernel' in x.name:
            return self.offdiag_regularizer(x)

        if 'diag_kernel' in x.name:
            return self._compute_diag_regularization(x)
        
    def _compute_diag_regularization(self, x):
        """
        Input : x : (numweights, filters)

        We have to first convert the input into a 
        triangular matrix and then apply different regularizations 
        on the diagonal and off diagonal components.
        """
        x = fill_triangular(tf.transpose(x, [1, 0]), upper=True) ## (C, A, A)
        diag_part = tf.linalg.diag_part(x) ## (C, A)
        off_diag_part = x - tf.linalg.diag(diag_part)
        res = self.diag_regularizer(diag_part) + self.offdiag_regularizer(off_diag_part)
        return res

    def get_config(self):
        config = {}
        config['diag_regularizer'] = self.diag_regularizer
        config['offdiag_regularizer'] = self.offdiag_regularizer
        return config