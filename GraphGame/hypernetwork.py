import keras
import tensorflow as tf 
from keras.layers import Dense
import tensorflow.keras.initializers as uniform


class Hypernetwork_A2C(keras.Model):

    def __init__(self, kernel_init):
        super().__init__()

        bias_init = tf.keras.initializers.Constant(0)

        self.dense_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_2 = Dense(570, activation="elu",kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w1_dense_1 = Dense(100, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w1_dense_2 = Dense(136, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        
        self.w1a_dense_1 = Dense(100, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w1a_dense_2 = Dense(136, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        
        self.w2_dense_1 = Dense(100, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w2_dense_2 = Dense(416, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w3_dense_1 = Dense(100, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w3_dense_2 = Dense(416, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        
        self.w4_dense_1 = Dense(100, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w4_dense_2 = Dense(260, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w5_dense_1 = Dense(100, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w5_dense_2 = Dense(65, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
      

    def call(self, inputs, batch_size):
        
        # This is the layer encoding
        # TODO: maybe create a config file for all this
        layer_1c = 8 
        layer_1a = 8 
        layer_2 = 10
        layer_3 = 10
        layer_4 = 1
        layer_5 = 1
        encoding = 15
        index_1c = layer_1c*encoding
        index_1a = index_1c + layer_1a*encoding
        index_2 = index_1a + layer_2*encoding
        index_3 = index_2 + layer_3*encoding
        index_4 = index_3 + layer_4*encoding
        index_5 = index_4 + layer_5*encoding
        
        # First part of the network, the extractor
        x = self.dense_1(inputs)
        x = self.dense_2(x)

        # Weight generator 1 
        input_w1c = x[:,:index_1c]
        input_w1c = tf.reshape(input_w1c,(batch_size,layer_1c,-1))
        w1c = self.w1_dense_1(input_w1c)
        w1c = self.w1_dense_2(w1c)
        
        # Weight generator 2
        input_w1a = x[:,index_1c:index_1a]
        input_w1a = tf.reshape(input_w1a,(batch_size,layer_1a,-1))
        w1a = self.w1a_dense_1(input_w1a)
        w1a = self.w1a_dense_2(w1a)
        
        # Weight generator 3
        input_w2 = x[:,index_1a:index_2]
        input_w2 = tf.reshape(input_w2,(batch_size,layer_2,-1))
        w2 = self.w2_dense_1(input_w2)
        w2 = self.w2_dense_2(w2)

        # Weight generator 4
        input_w3 = x[:,index_2:index_3]
        input_w3 = tf.reshape(input_w3,(batch_size,layer_3,-1))
        w3 = self.w3_dense_1(input_w3)
        w3 = self.w3_dense_2(w3)
       
        # Weight generator 5
        input_w4 = x[:,index_3:index_4]
        input_w4 = tf.reshape(input_w4,(batch_size,layer_4,-1))
        w4 = self.w4_dense_1(input_w4)
        w4 = self.w4_dense_2(w4)
        
        # Weight generator 6
        input_w5 = x[:,index_4:index_5]
        input_w5 = tf.reshape(input_w5,(batch_size,layer_5,-1))
        w5 = self.w5_dense_1(input_w5)
        w5 = self.w5_dense_2(w5)       

        # TODO This is not the best way, maybe change it
        return [w1c, w1a, w2, w3, w4, w5]


class Hypernetwork_DDQN(keras.Model):

    def __init__(self, kernel_init):
        super().__init__()

        bias_init = tf.keras.initializers.Constant(0)

        self.dense_1 = Dense(300, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.dense_2 = Dense(570, activation="elu",kernel_initializer=kernel_init, bias_initializer=bias_init)

        self.w1a_dense_1 = Dense(100, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w1a_dense_2 = Dense(136, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        
        self.w2_dense_1 = Dense(100, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w2_dense_2 = Dense(416, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
  
        self.w3_dense_1 = Dense(100, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)
        self.w3_dense_2 = Dense(260, activation="elu", kernel_initializer=kernel_init, bias_initializer=bias_init)


    def call(self, inputs, batch_size):
        
        # This is the layer encoding
        # TODO: maybe create a config file for all this
        layer_1a = 8 
        layer_2 = 10
        layer_3 = 1
        encoding = 15

        index_1a = layer_1a*encoding
        index_2 = index_1a + layer_2*encoding
        index_3 = index_2 + layer_3*encoding
        
        # First part of the network, the extractor
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        
        # Weight generator 2
        input_w1a = x[:,:index_1a]
        input_w1a = tf.reshape(input_w1a,(batch_size,layer_1a,-1))
        w1a = self.w1a_dense_1(input_w1a)
        w1a = self.w1a_dense_2(w1a)
        
        # Weight generator 3
        input_w2 = x[:,index_1a:index_2]
        input_w2 = tf.reshape(input_w2,(batch_size,layer_2,-1))
        w2 = self.w2_dense_1(input_w2)
        w2 = self.w2_dense_2(w2)

        # Weight generator 4
        input_w3 = x[:,index_2:index_3]
        input_w3 = tf.reshape(input_w3,(batch_size,layer_3,-1))
        w3 = self.w3_dense_1(input_w3)
        w3 = self.w3_dense_2(w3)    

        # TODO This is not the best way, maybe change it
        return [w1a, w2, w3]






