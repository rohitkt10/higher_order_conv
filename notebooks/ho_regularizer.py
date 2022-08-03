import tensorflow as tf
from tensorflow import keras as tfk

__all__ = ['HigherOrderKernelRegularizer']

class HigherOrderKernelRegularizer(tfk.regularizers.Regularizer):
    """
    Regularizer for higher order convolutions which applied separate
    regularization to the diagonal and off-diagonal terms. 
    """
    def __init__(self, diag_regularizer, offdiag_regularizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diag_regularizer = diag_regularizer
        self.offdiag_regularizer = offdiag_regularizer
        
    def __call__(self, x):
        diag = tf.linalg.diag_part(x)
        offdiag = x - tf.linalg.diag(diag)
        if self.diag_regularizer:
            diag_reg = self.diag_regularizer(diag)
        else:
            diag_reg = 0.
        if self.offdiag_regularizer:
            offdiag_reg  = self.offdiag_regularizer(offdiag)
        else:
            offdiag_reg = 0. 
        return diag_reg + offdiag_reg
    
    def get_config(self):
        config = {}
        config['diag_regularizer'] = self.diag_regularizer
        config['offdiag_regularizer'] = self.offdiag_regularizer
        return config