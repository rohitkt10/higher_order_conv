import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util.tf_export import keras_export

__all__ = ['MultiHeadAttention', 'MultiHeadDense']

class MultiHeadDense(tfk.layers.Dense):
    def get_config(self):
        config = super(MultiHeadDense, self).get_config()
        config['units'] = self.units 
        config['num_heads'] = self.num_heads
        return config

    def __init__(self, units, num_heads, **kwargs):
        super().__init__(units, **kwargs)
        self.units = units 
        self.num_heads = num_heads
    
    def build(self, input_shape):
        """
        Expected shape -> (L, A)
        """
        input_shape = tensor_shape.TensorShape(input_shape)
        d_model = int(input_shape[-1])  ## size of input embedding 

        # add the kernel
        kernel_shape = (d_model,) + (self.num_heads, self.units)
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
            bias_shape = (self.num_heads, self.units)
            self.bias = self.add_weight(
                                    name = 'bias',
                                    shape = bias_shape,
                                    initializer=self.bias_initializer,
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint,
                                    trainable=True,
                                    dtype=self.dtype
                                    )
    
    def call(self, x):
        """
        x -> (batch, seq length, input dim)

        OUTPUT:
        res -> (batch, seq length, num heads, key dim)
        """
        res = tf.einsum("ijk, klm -> ijlm", x, self.kernel)
        if self.use_bias:
            res = res + self.bias
        return res

class MultiHeadAttention(tfk.layers.Layer):
    def get_config(self):
        config = {
        'num_heads':self.num_heads, 
        'key_dim':self.key_dim, 
        'value_dim':self.value_dim, 
        'use_output_bias':self.use_output_bias
        }
        return config 

    @property
    def name(self):
        return self._name
    
    def __init__(self, num_heads, key_dim, value_dim=None, use_output_bias=True, name='mha', **kwargs):
        super().__init__()
        self._name = name
        self.num_heads = num_heads
        self.key_dim = key_dim 
        self.value_dim = value_dim
        self.use_output_bias = use_output_bias

        # set up the key, query, value projection layers
        self.Query = MultiHeadDense(units=key_dim, num_heads=num_heads, name='query', **kwargs)
        self.Key = MultiHeadDense(units=key_dim, num_heads=num_heads, name='key', **kwargs)
        self.Value = MultiHeadDense(units=value_dim, num_heads=num_heads, name='value', **kwargs)
        self.attn_output = MultiHeadDense(units=value_dim, num_heads=num_heads, name='output', **kwargs)
        self.attn_output.use_bias = False ## add this separately
    
    def build(self, input_shape):
        # add the attention output biases
        input_shape = tensor_shape.TensorShape(input_shape)
        d_model = int(input_shape[-1])  ## size of input embedding

        # build all the layers 
        self.Query.build(input_shape)
        self.Key.build(input_shape)
        self.Value.build(input_shape)
        self.attn_output.build(input_shape)

        if self.use_output_bias:
            self.output_bias = self.add_weight(
                                    name = 'output_bias',
                                    shape = (d_model,),
                                    initializer=self.attn_output.bias_initializer,
                                    regularizer=self.attn_output.bias_regularizer,
                                    constraint=self.attn_output.bias_constraint,
                                    trainable=True,
                                    dtype=self.dtype
                                    )

    def call(self, query, value, key=None, return_attention_scores=False, ):
        """
        query -> (batch, L, d_model)
        value -> (batch, L, d_model)
        key (Optional) -> (batch, L, d_model) 
        """
        d_model = query.shape[-1]
        if key is None:
            key = value
        
        # calculate the queries, keys and values and reshape
        allQ = self.Query(query)  ## (batch, L, h, keydim)
        allK = self.Key(key)      ## (batch, L, h, keydim)
        allV = self.Value(value)  ## (batch, L, h, valuedim)
        allQ = tf.transpose(allQ, perm=[0, 2, 1, 3])  ## (batch, h, L, keydim)
        allK = tf.transpose(allK, perm=[0, 2, 3, 1])  ## (batch, h, keydim, L)
        allV = tf.transpose(allV, perm=[0, 2, 1, 3])  ## (batch, h, L, valuedim)

        # calculate the attention scores
        dot_product = tf.matmul(allQ, allK)  ## (batch, h, L, L)
        scaled_dot_product = dot_product / tf.math.sqrt(2.)
        attn_scores = tf.nn.softmax(scaled_dot_product, axis=-1)  ## (batch, h, L, L)
        attn_res = tf.matmul(attn_scores, allV)  ## (batch, h, L, valuedim)
        res = tf.einsum("ijkl, klm -> ijm", tf.transpose(attn_res, [0,2,1,3]), 
                        tf.transpose(self.attn_output.kernel, [1, 2, 0]))  ## (batch, L, d_model)

        if self.use_output_bias:
            res = res + self.output_bias
        
        if return_attention_scores:
            return res, attn_scores
        return res