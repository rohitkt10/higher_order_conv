from tensorflow.keras import activations
from functools import wraps
import tensorflow as tf

def get_activation(name):
	"""
	name <str> - Name of the activation function.

	The prescribed activation must be implemented in 
	tensorflow.keras.activations. 
	"""

	if not isinstance(name, str):
		raise ValueError("Must pass a string name for the required activation function.")
	else:
		if not name in activations.__dict__.keys():
			raise NotImplementedError("The prescribed activation function is not \
				implemented in keras.")
		else:
			return activations.__dict__[name]

def numpy_metric(metric_fn):
    """
    A decorator which wraps a metric function with the signature
    fn(ytrue, ypred), whose operations are performed using numpy
    """
    @wraps(metric_fn)
    def wrapper(*args):
        ytrue, ypred = args
        
        if tf.is_tensor(ytrue):
            ytrue = ytrue.numpy()
        if tf.is_tensor(ypred):
            ypred = ypred.numpy()
        return metric_fn(ytrue, ypred)
    return wrapper