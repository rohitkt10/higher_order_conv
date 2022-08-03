import tensorflow as tf
from tensorflow import keras as tfk

def get_model(L, A, first_conv_layer):
    """
    Set up the first convolutional layer outside this function. 
    """
    input_shape = [L, A]
    x = tfk.layers.Input(shape=input_shape, name='input')
    
    # 1st conv layer
    assert isinstance(first_conv_layer, tfk.layers.Conv1D)
    y = first_conv_layer(x)
    y = tfk.layers.BatchNormalization()(y)
    y = tfk.layers.Activation('relu')(y)
    y = tfk.layers.MaxPool1D(pool_size=25)(y)
    y = tfk.layers.Dropout(0.2)(y)
    
    # remaining layers 
    y = tfk.layers.Conv1D(filters=128, kernel_size=7, padding='same', kernel_regularizer=tfk.regularizers.l2(1e-6))(y)
    y = tfk.layers.BatchNormalization()(y)
    y = tfk.layers.Activation('relu')(y)
    y = tfk.layers.MaxPool1D(pool_size=2)(y) 
    y = tfk.layers.Dropout(0.2)(y)
    
    y = tfk.layers.Flatten()(y)
    y = tfk.layers.Dense(512, kernel_regularizer=tfk.regularizers.l2(1e-6))(y)      
    y = tfk.layers.BatchNormalization()(y)
    y = tfk.layers.Activation('relu')(y)
    y = tfk.layers.Dropout(0.5)(y)
    
    y = tfk.layers.Dense(1, name='logits')(y) ## logits
    y = tfk.layers.Activation('sigmoid', name='output')(y) ## final output 
    
    model = tfk.Model(inputs=x, outputs=y, name='cnn-25')
    
    return model