import keras
from matplotlib.pyplot import xkcd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization

# Here the two networks are implemented. 
# The classifier network is self explanatory: Two convolutional layers with maxpooling and two dense layers.

def MNIST_Model(seed):

    initializer = tf.keras.initializers.glorot_uniform(seed=seed)

    model = tf.keras.models.Sequential([
        Conv2D(32, (5, 5), padding="same", activation="relu", strides=(1,1),input_shape = (28,28,1), kernel_initializer=initializer),
        MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
        Conv2D(16, (5, 5), padding="same", strides=(1,1), activation="relu", kernel_initializer=initializer),
        MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding="same"),
        tf.keras.layers.Flatten(),
        Dense(8, activation= "relu",kernel_initializer=initializer),
        Dense(10, activation= "softmax",kernel_initializer=initializer)
    ])
    
    return model

class Hypernetwork(keras.Model):

    def __init__(self, seed):
        super().__init__(seed)

        # The weight initialization is the same as in Deutschs paper and produces weights in the correct scale. 
        kernel_init = tf.keras.initializers.VarianceScaling(scale=0.01, seed=seed)
        # The biases are set to zero 
        bias_init = tf.keras.initializers.Constant(0)

        # This first block is the Extractor with two dense layers and batch normalization
        self.dense_1 = Dense(300, activation="elu", use_bias=False,kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_2 = Dense(855, activation="elu", use_bias=False,kernel_initializer=kernel_init, bias_initializer=bias_init)

        # The following blocks are the weight generators, each has two layers and also batchnormalization
        self.w1_dense_1 = Dense(40, activation="elu", use_bias=False,kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w1_dense_2 = Dense(26, activation="elu", use_bias=False,kernel_initializer=kernel_init, bias_initializer=bias_init) 
        
        self.w2_dense_1 = Dense(100, activation="elu", use_bias=False,kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w2_dense_2 = Dense(801, activation="elu", use_bias=False,kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w3_dense_1 = Dense(100, activation="elu", use_bias=False,kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w3_dense_2 = Dense(785, activation="elu",use_bias=False, kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w4_dense_1 = Dense(60, activation="elu", use_bias=False,kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w4_dense_2 = Dense(90, activation="elu", use_bias=False,kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.batch_norm = BatchNormalization(momentum=0.98)
        self.batch_norm_1 = BatchNormalization(momentum=0.98)
        self.batch_norm_2 = BatchNormalization(momentum=0.98)
        self.batch_norm_3 = BatchNormalization(momentum=0.98)
        self.batch_norm_4 = BatchNormalization(momentum=0.98)


    def call(self, inputs, batch_size):
        
        # These are the sizes of the layers
        layer_1 = 32
        layer_2 = 16
        layer_3 = 8
        layer_4 = 1

        # The size of the encoding for each filter
        encoding = 15

        # The indexes used to generate the weights with the weight generators of the main network
        index_1 = layer_1*encoding
        index_2 = index_1 + layer_2*encoding
        index_3 = index_2 + layer_3*encoding
        index_4 = index_3 + layer_4*encoding

        # Here the call function for the Extractor
        x = self.dense_1(inputs)
        x = self.batch_norm(x)
        x = self.dense_2(x)

        # The following blocks structure the output of the extractor and feed it to the weight generators for the layers 1, 2, 3 and 4.
        input_w1 = x[:,:index_1]
        input_w1 = tf.reshape(input_w1,(batch_size,layer_1,-1))
        w1 = self.w1_dense_1(input_w1)
        w1 = self.batch_norm_1(w1)
        w1 = self.w1_dense_2(w1)

        input_w2 = x[:,index_1:index_2]
        input_w2 = tf.reshape(input_w2,(batch_size,layer_2,-1))
        w2 = self.w2_dense_1(input_w2)
        w2 = self.batch_norm_2(w2)
        w2 = self.w2_dense_2(w2)

        input_w3 = x[:,index_2:index_3]
        input_w3 = tf.reshape(input_w3,(batch_size,layer_3,-1))
        w3 = self.w3_dense_1(input_w3)
        w3 = self.batch_norm_3(w3)
        w3 = self.w3_dense_2(w3)

        input_w4 = x[:,index_3:index_4]
        input_w4 = tf.reshape(input_w4,(batch_size,-1))
        w4 = self.w4_dense_1(input_w4)
        w4 = self.batch_norm_4(w4)
        w4 = self.w4_dense_2(w4)

        return w1, w2, w3, w4
