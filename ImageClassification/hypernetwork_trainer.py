import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
import keras
import os
import csv
import networks
from sklearn.manifold import TSNE

from scipy import interpolate
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score



# This file contains the trainer which calls the hypernetwork and contains the training functions 
# It's called by the main file and calls the network file to load the networks

class Trainer():

  def __init__(self, seed):

      # Here the parameters for the training are specified. The trainer just takes a seed as input. 
      # The other parameters are specified in this file or the networks file
      self.seed = seed

      self.main_network = networks.MNIST_Model(seed) # loading the networks
      self.hypernetwork = networks.Hypernetwork(seed)
      self.training_steps = 10_001 # 10_001 because a for loop is used
      self.log_every = 10 # Intervall for plotting
      self.save_every = 1000 # Intervall for saving
      self.learning_rate = 3e-4
      self.decay_steps = 1
      self.learning_rate_rate = 0.99998
      self.batch_size = 32 # batch size of hypernetwork
      self.main_batch_size = 32 # batch size for each main network
      self.N_val = 500 # size of validation batch
      self.checkpoints = 90 # number of evalutions over the path between the vectors z1 and z2 
      self.image_height = 28 
      self.image_width = 28

      # create a loss function and assign an optimizer with a learning rate schedule
      self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
      self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = self.learning_rate, decay_steps=self.decay_steps, decay_rate=self.learning_rate_rate)
      self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule) 

      # Load the dataset from the keras MNIST data set
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
      x_train = x_train.astype('float32') 
      x_test = x_test.astype('float32')   
      y_train = tf.keras.utils.to_categorical(y_train, 10)
      x_train = x_train.reshape(x_train.shape[0], 28, 28,1)

      # Define train, test and validation set
      self.x_val = x_train[-10000:,:,:,:]
      self.y_val = y_train[-10000:]
      self.x_train = x_train[:-10000]
      self.y_train = y_train[:-10000]
      self.x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
      self.y_test = tf.keras.utils.to_categorical(y_test, 10)

      # Create the logging paths
      self.log_dir = 'Image_classification/experiments_2/'      

  def logger(self):

    # This function creates the logging directory and the logger 
    if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    self.csv_file = open(self.csv_path, 'w')
    fieldnames = ['step', 'train accuracy', 'validation accuracy', 'loss', 'learning_rate', 'diversity loss', 'accuracy loss']
    self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
    self.writer.writeheader()

  def ExpandDims(self, tensor, axis):

      for i in np.sort(axis):
          tensor = tf.expand_dims(tensor,i)
      tensor = tf.identity(tensor)
      return tensor

  def train_hypernetwork(self, lamBda):

    self.csv_path = os.path.join(self.log_dir, 'performance{}.csv'.format(lamBda))
    self.logger()
  
    # Here the training loop for the hypernetwork starts
    for step in range(1, self.training_steps):
      
      train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
      val_acc_metric = tf.keras.metrics.CategoricalAccuracy()

      # From here on (tensorflow specific) the gradients are tracked.
      with tf.GradientTape() as tape:
        
        # The random vector z is drawn from a uniform distribution
        z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,300])
        
        # The hypernetwork is called with the vector z and the weights of the main networks are created
        weights_1, weights_2, weights_3, weights_4 = self.hypernetwork(z, self.batch_size)
        # These are the values specified for the gauge fixing 

        # layer1 : convolutional
        self.layer1_filter_size = 5
        self.layer1_size = 32
        self.layer1_pool_size = 2
        self.number_of_channels = 1

        # layer2 : convolutional
        self.layer2_filter_size = 5
        self.layer2_size = 16
        self.layer2_pool_size = 2

        # layer3 : fully connected
        self.layer3_size = 8
        self.zero_fixer = 1e-8

        # This is the reshaping before the actual gauge fixing
        w1_not_gauged = weights_1[:,:,0:-1]
        w1_not_gauged  = tf.transpose(tf.reshape(w1_not_gauged ,(self.batch_size,32,5,5,1)), [0,2,3,4,1])
        b1_not_gauged  = weights_1[:,:,-1]
        w2_not_gauged  = weights_2[:,:,0:-1]
        w2_not_gauged  = tf.transpose(tf.reshape(w2_not_gauged ,(self.batch_size,16,5,5,32)), [0,2,3,4,1])
        b2_not_gauged  = weights_2[:,:,-1]
        w3_not_gauged  = weights_3[:,:,0:-1]
        w3_not_gauged  = tf.transpose(tf.reshape(w3_not_gauged ,(self.batch_size,8,7,7,16)), [0,2,3,4,1])
        b3_not_gauged  = weights_3[:,:,-1]
        w4_not_gauged  = weights_4[:,0:80]
        w4_not_gauged  = tf.reshape(w4_not_gauged ,(self.batch_size,8,10))
        b4_not_gauged  = weights_4[:,80:]

        # The generated batch of weights are 'normalized'
        required_scale = self.layer1_filter_size*self.layer1_filter_size*self.number_of_channels+1
        scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w1_not_gauged), [1, 2, 3]) + tf.square(b1_not_gauged))/required_scale+self.zero_fixer)
        w1 = w1_not_gauged / (self.ExpandDims(scale_factor, [1, 2, 3]) + self.zero_fixer)
        b1 = b1_not_gauged / (scale_factor + self.zero_fixer)
        w2 = w2_not_gauged * self.ExpandDims(scale_factor, [1, 2, 4])
        w1 = tf.identity(w1,'w1')
        b1 = tf.identity(b1, 'b1')

        required_scale = self.layer2_filter_size*self.layer2_filter_size*self.layer1_size+1
        scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w2), [1, 2, 3]) + tf.square(b2_not_gauged))/required_scale+self.zero_fixer)
        w2 = w2 / (self.ExpandDims(scale_factor, [1, 2, 3]) + self.zero_fixer)
        b2 = b2_not_gauged / (scale_factor + self.zero_fixer)
        w3 = w3_not_gauged * self.ExpandDims(scale_factor, [1, 2, 4])
        w2 = tf.identity(w2, 'w2')
        b2 = tf.identity(b2, 'b2')

        required_scale = (self.image_height / (self.layer1_pool_size * self.layer2_pool_size)) * (self.image_width / (self.layer1_pool_size * self.layer2_pool_size)) * self.layer2_size + 1
        scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w3), [1, 2, 3]) + tf.square(b3_not_gauged))/required_scale+self.zero_fixer)
        w3 = w3 / (self.ExpandDims(scale_factor, [1, 2, 3]) + self.zero_fixer)
        b3 = b3_not_gauged / (scale_factor + self.zero_fixer)
        w4 = w4_not_gauged * self.ExpandDims(scale_factor, [2])
        w3 = tf.identity(w3, 'w3')
        b3 = tf.identity(b3, 'b3')

        required_softmax_bias = 0.0
        softmax_bias_diff = tf.reduce_sum(b4_not_gauged, 1) - required_softmax_bias
        b4 = b4_not_gauged - self.ExpandDims(softmax_bias_diff,[1])
        w4 = tf.identity(w4, 'w4')
        b4 = tf.identity(b4, 'b4')
        
        # Here the batches are drawn form the training dataset
        idx = np.random.randint(low=0, high=self.x_train.shape[0], size=self.batch_size*self.main_batch_size)
        x, y = self.x_train[idx], self.y_train[idx]
        x = tf.reshape(x, (self.batch_size,self.main_batch_size,28,28,1))
        y = tf.reshape(y, (self.batch_size,self.main_batch_size,10))

        # This is equivalent to the MNIST network implemented in the networks.py file. The advantage of this implementation which is that the whole batch is passed through all the networks at once
        # This is copied from https://github.com/sliorde/generating-neural-networks-with-neural-networks
        fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, self.layer1_pool_size, self.layer1_pool_size, 1],strides=[1, self.layer1_pool_size, self.layer1_pool_size, 1],padding='SAME')
        c_layer1_output = tf.map_fn(fn, elems=[x, w1, b1], dtype=tf.float32, name='layer1_output')

        fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, self.layer2_pool_size, self.layer2_pool_size, 1],strides=[1, self.layer2_pool_size, self.layer2_pool_size, 1],padding='SAME')
        c_layer2_output = tf.map_fn(fn, elems=[c_layer1_output, w2, b2], dtype=tf.float32,name='layer2_output')
        c_layer3_output = tf.nn.relu(tf.reduce_sum(tf.expand_dims(c_layer2_output, -1) * tf.expand_dims(w3,1), axis=(2, 3, 4)) + tf.expand_dims(b3,1),name='layer3_output')
        c_layer4_output = tf.identity(tf.reduce_sum(tf.expand_dims(c_layer3_output, -1) * tf.expand_dims(w4,1), axis=2) + tf.expand_dims(b4,1),name='layer4_output')

        # The probabilities and the predictions of the main networks
        probs = tf.nn.softmax(c_layer4_output)
        preds = tf.argmax(probs, axis=2)

        # The accuracy loss is calculated
        loss_acc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(y,shape=(-1,10)), logits=tf.reshape(c_layer4_output,shape=(-1,10))),name='accuracy_loss')
        
        # Here the diversity loss is calculated also according to https://github.com/sliorde/generating-neural-networks-with-neural-networks
        zero_fixer = 1e-8
        input_noise_size = 300
        noise_batch_size = self.batch_size #tf.identity(32,name='noise_batch_size') 
        flattened_network = tf.concat(axis=1, values=[tf.reshape(w1, [noise_batch_size, -1]),tf.reshape(b1, [noise_batch_size, -1]),tf.reshape(w2, [noise_batch_size, -1]),tf.reshape(b2, [noise_batch_size, -1]),tf.reshape(w3, [noise_batch_size, -1]),tf.reshape(b3, [noise_batch_size, -1]),tf.reshape(w4, [noise_batch_size, -1]),tf.reshape(b4, [noise_batch_size, -1])],name='flattened_network')
        # entropy estimated using  Kozachenko-Leonenko estimator, with l1 distances
        mutual_distances = tf.math.reduce_sum(tf.math.abs(tf.expand_dims(flattened_network, 0) - tf.expand_dims(flattened_network, 1)), 2,name='mutual_squared_distances') # all distances between weight vector samples
        nearest_distances = tf.identity(-1*tf.math.top_k(-1 * mutual_distances, k=2)[0][:, 1] ,name='nearest_distances') # distance to nearest neighboor for each weight vector sample
        entropy_estimate = tf.identity(input_noise_size * tf.math.reduce_mean(tf.math.log(nearest_distances + zero_fixer)) + tf.math.digamma(tf.cast(noise_batch_size, tf.float32)), name='entropy_estimate')
        loss_div = tf.identity( - 1 * entropy_estimate)
        
        # The overall loss
        loss = loss_acc*lamBda + loss_div

      # The gradient update
      grads = tape.gradient(loss, self.hypernetwork.trainable_weights)
      self.optimizer.apply_gradients(zip(grads, self.hypernetwork.trainable_weights))
      #print(self.hypernetwork.summary())

      # The logging
      if step % self.log_every == 0:

        # Training set accuracy
        accuracy_train = train_acc_metric(y, probs)

        # Samples for the estimation of the validation set accuracy are drawn
        idx = np.random.randint(low=0, high=self.x_val.shape[0], size=self.batch_size*self.main_batch_size)
        x_val, y_val = self.x_val[idx], self.y_val[idx]
        x_val = tf.reshape(x_val, (self.batch_size,self.main_batch_size,28,28,1))
        y_val = tf.reshape(y_val, (self.batch_size,self.main_batch_size,10))

        fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, self.layer1_pool_size, self.layer1_pool_size, 1],strides=[1, self.layer1_pool_size, self.layer1_pool_size, 1],padding='SAME')
        c_layer1_output = tf.map_fn(fn, elems=[x_val, w1, b1], dtype=tf.float32,name='layer1_output')

        fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, self.layer2_pool_size, self.layer2_pool_size, 1],strides=[1, self.layer2_pool_size, self.layer2_pool_size, 1],padding='SAME')
        c_layer2_output = tf.map_fn(fn, elems=[c_layer1_output, w2, b2], dtype=tf.float32,name='layer2_output')
        c_layer3_output = tf.nn.relu(tf.reduce_sum(tf.expand_dims(c_layer2_output, -1) * tf.expand_dims(w3,1), axis=(2, 3, 4)) + tf.expand_dims(b3,1),name='layer3_output')
        c_layer4_output = tf.identity(tf.reduce_sum(tf.expand_dims(c_layer3_output, -1) * tf.expand_dims(w4,1), axis=2) + tf.expand_dims(b4,1),name='layer4_output')

        # The probabilities and the predictions on the validation set of the main networks
        probs_val = tf.nn.softmax(c_layer4_output)
        preds_val = tf.argmax(probs, axis=2)

        # Validation set accuracy
        accuracy_val = val_acc_metric(y_val, probs_val)
        
        # Printing the values
        print(' Step: {}, validation set accuracy: {:2.4f}, train set accuracy: {:2.4f}, loss: {:2.4f}, loss_acc: {:2.4f}, loss_div: {:2.4f}, learning rate: {:2.4f}'.format(step, accuracy_val, accuracy_train, loss, loss_acc, loss_div, self.optimizer._decayed_lr(tf.float32).numpy()))

        # Logging the values
        self.writer.writerow({'step': step, 'train accuracy': accuracy_train.numpy(), 'validation accuracy': accuracy_val.numpy(), 'loss': loss.numpy(), \
        'learning_rate': self.optimizer._decayed_lr(tf.float32).numpy(), 'diversity loss': loss_div.numpy(), 'accuracy loss': loss_acc.numpy()})
        
        loss_accum = 0.0
      
      if step % self.save_every == 0:
        self.hypernetwork.save_weights('Image_classification/experiments/hyper_model_{}_v{}.h5'.format(lamBda, step))
        
    self.csv_file.close()

  def train_mnist_networks(self, seed):

    # Here, len(seed) networks are trained
    for a in seed:

      self.csv_path = os.path.join(self.log_dir, 'performance{}.csv'.format(a))
      self.logger()
      
      # Model is called and loaded with seed and below the optimizer, metric and loss are definded
      model = networks.MNIST_Model(a)
      optimizer=tf.keras.optimizers.Adam(0.001)
      train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
      val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
      loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
      
      training_steps = 5000 # training steps for the networks

      # The actual training loop
      for i in range(training_steps):

        with tf.GradientTape() as tape:
          
          # Randomly choose a batch from the training set
          idx = np.random.randint(low=0, high=self.x_train.shape[0], size=self.batch_size)
          x, y = self.x_train[idx], self.y_train[idx]

          # Calculate prediction, loss and accuracy of the batch
          pred = model(x)
          loss = loss_fn(y,pred)
          accuracy_train = train_acc_metric(y, pred)

        # Perform the update step
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # Log the data ever 50 batches
        if i % 10 == 0:

          idx = np.random.randint(low=0, high=self.x_val.shape[0], size=self.batch_size)
          x_val, y_val = self.x_val[idx], self.y_val[idx]
          pred_val = model(x_val)
          accuracy_val = train_acc_metric(y_val, pred_val)

          self.writer.writerow({'step': i, 'train accuracy': accuracy_train.numpy(), 'validation accuracy': accuracy_val.numpy(), 'loss': loss.numpy(), \
          'learning_rate': self.optimizer._decayed_lr(tf.float32).numpy(), 'diversity loss': 0, 'accuracy loss': 0})
        
          print(' Step: {}, validation set accuracy: {:2.4f}, train set accuracy: {:2.4f}, loss: {:2.4f}, learning rate: {}'.format(i, accuracy_val.numpy(), accuracy_train.numpy(), loss, self.optimizer._decayed_lr(tf.float32).numpy()))

        # Save the model every 2500 batches  
        if i % 1000 == 0:
          model.save(('Image_classification/experiemtns/classifier{}_{}.h5'.format(a,i)))
      

  def cos_between(self, v1, v2):

    # Returns the angle in radians between vectors 'v1' and 'v2'
    # Helper function for the cosine_matrices function
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.dot(v1_u, v2_u)
  
  def set_parameters(self, weights):

    flattend_weights = []
    for i in range(32):
      flatten1 = tf.reshape(weights[0][i], -1)
      flatten2 = tf.reshape(weights[1][i], -1)
      flatten3 = tf.reshape(weights[2][i], -1)
      flatten4 = tf.reshape(weights[3][i], -1)
      flattend_weight = tf.concat([flatten1, flatten2, flatten3, flatten4],axis=0)
      flattend_weights.append(flattend_weight)

    return flattend_weights


  def cosine_matrices(self):

    self.batch_size = 32
    z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,300])
    #zb = np.random.uniform(low = -1, high = 1, size = [self.batch_size,300])
    self.hypernetwork.compile()
    dummy = self.hypernetwork(z,self.batch_size)
    filepath = 'Image_classification/experiments/hyper_model_10000_v10000.h5'
    self.hypernetwork.load_weights(filepath)
    
    weights_preda = self.hypernetwork(z, self.batch_size)
    flattend_weights = self.set_parameters(weights_preda)
    cos_matrix = np.zeros((32,32))
    for i in range(32):
      for j in range(i,32):
        cos_matrix[i][j] = self.cos_between(flattend_weights[i], flattend_weights[j])
        cos_matrix[j][i] = cos_matrix[i][j]

    plt.imshow(cos_matrix, cmap="bwr", origin="lower")
    plt.colorbar()
    plt.grid("off")

    #title = "Cosine Between Independent Solutions"
    #plt.title(title)

    plt.xlabel("Network weights")
    plt.ylabel("Network weights")
    save_path = 'Image_classification/experiments/cos_hyper_end_new.png'
    plt.savefig(save_path)
    plt.show()
    
    '''
    flattend_weights = []
    cos_matrix = np.zeros((10,10))

    ######
    
    for i in [1,10,20,30,40,50,60,70,80,90]:
      network1 = keras.models.load_model('Image_classification/Image_classification/experiments/classifier{}_4000.h5'.format(i))
      v1 = network1.get_weights()
      x = []
      for idx1 in range(len(v1)):
        a = np.reshape(v1[idx1],[-1])
        x.append(a)
      x = np.concatenate(x, axis=0)
      flattend_weights.append(x)

    for i in range(10):
      for j in range(i,10):
        cos_matrix[i][j] = self.cos_between(flattend_weights[i], flattend_weights[j])
        cos_matrix[j][i] = cos_matrix[i][j]

    plt.imshow(cos_matrix, cmap="bwr", origin="lower")
    plt.colorbar()
    plt.grid("off")

    plt.xlabel("Network weights")
    plt.ylabel("Network weights")

    save_path = 'Image_classification/experiments/cos_normal.png'
    plt.savefig(save_path)

    plt.show()

    network1 = keras.models.load_model('Image_classification/models/checkpoint_10/mnist_v1500.h5')
    network2 = keras.models.load_model('Image_classification/models/checkpoint_2/mnist_v1500.h5')
    network3 = keras.models.load_model('Image_classification/models/checkpoint_3/mnist_v4500.h5')
    networks = [network1, network2, network3]
    
    flattend_weights = []
    cos_matrix = np.zeros((3, 3))

    for network in networks:
      v2 = network.get_weights()
      y = []
      for idx2 in range(len(v2)):
        b = np.reshape(v2[idx2],[-1])
        y.append(b)
      y = np.concatenate(y, axis=0)
      flattend_weights.append(y)

    for i in range(3):
      for j in range(i,3):
        cos_matrix[i][j] = self.cos_between(flattend_weights[i], flattend_weights[j])
        cos_matrix[j][i] = cos_matrix[i][j]

    plt.imshow(cos_matrix, cmap="bwr", origin="lower")
    plt.colorbar()
    plt.grid("off")

    title = "Cosine Between Independent Solutions"
    plt.title(title)

    plt.xlabel("Independent Solution")
    plt.ylabel("Independent Solution")
    plt.show()
    '''

  def weight_path(self):
    
    z = np.random.uniform(low = -1, high = 1, size = [self.batch_size,300])

    self.hypernetwork(z,32)
    self.hypernetwork.load_weights('Image_classification/experiments/hyper_model_10000_v1000.h5')
    
    resolution = np.arange(0,1,0.01)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    y_loss1, y_loss2 = [], []
    y_acc1, y_acc2 = [], []
    x_axis = []
    acc_metric1 = tf.keras.metrics.CategoricalAccuracy()
    acc_metric2 = tf.keras.metrics.CategoricalAccuracy()


    
    weights_1, weights_2, weights_3, weights_4 = self.hypernetwork(z, self.batch_size)
    
    layer1_pool_size = 2
    layer2_pool_size = 2
    self.layer1_filter_size = 5
    self.layer1_size = 32
    self.layer1_pool_size = 2
    self.number_of_channels = 1

    # layer2 : convolutional
    self.layer2_filter_size = 5
    self.layer2_size = 16
    self.layer2_pool_size = 2

    # layer3 : fully connected
    self.layer3_size = 8
    self.zero_fixer = 1e-8

    w1_not_gauged = weights_1[:,:,0:-1]
    w1_not_gauged  = tf.transpose(tf.reshape(w1_not_gauged ,(self.batch_size,32,5,5,1)), [0,2,3,4,1])
    b1_not_gauged  = weights_1[:,:,-1]
    w2_not_gauged  = weights_2[:,:,0:-1]
    w2_not_gauged  = tf.transpose(tf.reshape(w2_not_gauged ,(self.batch_size,16,5,5,32)), [0,2,3,4,1])
    b2_not_gauged  = weights_2[:,:,-1]
    w3_not_gauged  = weights_3[:,:,0:-1]
    w3_not_gauged  = tf.transpose(tf.reshape(w3_not_gauged ,(self.batch_size,8,7,7,16)), [0,2,3,4,1])
    b3_not_gauged  = weights_3[:,:,-1]
    w4_not_gauged  = weights_4[:,0:80]
    w4_not_gauged  = tf.reshape(w4_not_gauged ,(self.batch_size,8,10))
    b4_not_gauged  = weights_4[:,80:]


    required_scale = self.layer1_filter_size*self.layer1_filter_size*self.number_of_channels+1
    scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w1_not_gauged), [1, 2, 3]) + tf.square(b1_not_gauged))/required_scale+self.zero_fixer)
    w1 = w1_not_gauged / (self.ExpandDims(scale_factor, [1, 2, 3]) + self.zero_fixer)
    b1 = b1_not_gauged / (scale_factor + self.zero_fixer)
    w2 = w2_not_gauged * self.ExpandDims(scale_factor, [1, 2, 4])
    w1 = tf.identity(w1,'w1')
    b1 = tf.identity(b1, 'b1')

    required_scale = self.layer2_filter_size*self.layer2_filter_size*self.layer1_size+1
    scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w2), [1, 2, 3]) + tf.square(b2_not_gauged))/required_scale+self.zero_fixer)
    w2 = w2 / (self.ExpandDims(scale_factor, [1, 2, 3]) + self.zero_fixer)
    b2 = b2_not_gauged / (scale_factor + self.zero_fixer)
    w3 = w3_not_gauged * self.ExpandDims(scale_factor, [1, 2, 4])
    w2 = tf.identity(w2, 'w2')
    b2 = tf.identity(b2, 'b2')

    required_scale = (self.image_height / (self.layer1_pool_size * self.layer2_pool_size)) * (self.image_width / (self.layer1_pool_size * self.layer2_pool_size)) * self.layer2_size + 1
    scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w3), [1, 2, 3]) + tf.square(b3_not_gauged))/required_scale+self.zero_fixer)
    w3 = w3 / (self.ExpandDims(scale_factor, [1, 2, 3]) + self.zero_fixer)
    b3 = b3_not_gauged / (scale_factor + self.zero_fixer)
    w4 = w4_not_gauged * self.ExpandDims(scale_factor, [2])
    w3 = tf.identity(w3, 'w3')
    b3 = tf.identity(b3, 'b3')

    required_softmax_bias = 0.0
    softmax_bias_diff = tf.reduce_sum(b4_not_gauged, 1) - required_softmax_bias
    b4 = b4_not_gauged - self.ExpandDims(softmax_bias_diff,[1])
    w4 = tf.identity(w4, 'w4')
    b4 = tf.identity(b4, 'b4')

    self.batch_size = 300
    self.target_batch_size = 100

    idx = np.random.randint(low=0, high=self.x_train.shape[0], size=self.target_batch_size)#*self.target_batch_size)
    x1, y1 = self.x_train[idx], self.y_train[idx]
    x, y = [],[]
    for i in range(self.batch_size):
      x.append(x1)
      y.append(y1)

    x = tf.reshape(x, (self.batch_size,self.target_batch_size,28,28,1))
    y = tf.reshape(y, (self.batch_size,self.target_batch_size,10))

    noise_batch_size = 300

    interp = lambda q: (q[[-1]]-q[[0]])*np.reshape(np.linspace(0,1,noise_batch_size),[-1]+[1]*(q.ndim-1))+q[[0]]
    
    z = interp(z)
    w1 = interp(w1)
    b1 = interp(b1)
    w2 = interp(w2)
    b2 = interp(b2)
    w3 = interp(w3)
    b3 = interp(b3)
    w4 = interp(w4)
    b4 = interp(b4)

    fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, layer1_pool_size, layer1_pool_size, 1],strides=[1, layer1_pool_size, layer1_pool_size, 1],padding='SAME')
    c_layer1_output = tf.map_fn(fn, elems=[x, w1, b1], dtype=tf.float32,name='layer1_output')

    fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, layer2_pool_size, layer2_pool_size, 1],strides=[1, layer2_pool_size, layer2_pool_size, 1],padding='SAME')
    c_layer2_output = tf.map_fn(fn, elems=[c_layer1_output, w2, b2], dtype=tf.float32,name='layer2_output')

    c_layer3_output = tf.nn.relu(tf.reduce_sum(tf.expand_dims(c_layer2_output, -1) * tf.expand_dims(w3,1), axis=(2, 3, 4)) + tf.expand_dims(b3,1),name='layer3_output')

    c_layer4_output = tf.identity(tf.reduce_sum(tf.expand_dims(c_layer3_output, -1) * tf.expand_dims(w4,1), axis=2) + tf.expand_dims(b4,1),name='layer4_output')

    probs = tf.nn.softmax(c_layer4_output,name='probabilities')
    preds = tf.argmax(probs, axis=2, name='prediction')
    correct_predictions = tf.equal(preds, tf.argmax(y, axis=2), name='correct_prediction')
    accuracy1 = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=1, name='accuracy').numpy()

    ############################
    '''
    so = preds[0]
    swe = preds[1]
    classes_predicted = np.concatenate((preds[0],preds[1]))
    fractional_differences = np.zeros((2,2))
    y = 2
    fractional_differences = np.mean(
        classes_predicted.reshape([1, 2, 32]) != classes_predicted.reshape(
                [2, 1, 32]),
        axis=-1)

    plt.imshow(fractional_differences, interpolation="nearest", cmap="bwr", origin="lower")
    plt.colorbar()
    plt.grid("off")

    title = "Disagreement Fraction Btw Independent Solutions"
    plt.title(title)

    plt.xlabel("Independent Solution")
    plt.ylabel("Independent Solution")
    plt.show()
    '''


    ########################
    weights_1, weights_2, weights_3, weights_4 = self.hypernetwork(z, self.batch_size)
    
    layer1_pool_size = 2
    layer2_pool_size = 2
    self.layer1_filter_size = 5
    self.layer1_size = 32
    self.layer1_pool_size = 2
    self.number_of_channels = 1

    # layer2 : convolutional
    self.layer2_filter_size = 5
    self.layer2_size = 16
    self.layer2_pool_size = 2

    # layer3 : fully connected
    self.layer3_size = 8
    self.zero_fixer = 1e-8

    w1_not_gauged = weights_1[:,:,0:-1]
    w1_not_gauged  = tf.transpose(tf.reshape(w1_not_gauged ,(self.batch_size,32,5,5,1)), [0,2,3,4,1])
    b1_not_gauged  = weights_1[:,:,-1]
    w2_not_gauged  = weights_2[:,:,0:-1]
    w2_not_gauged  = tf.transpose(tf.reshape(w2_not_gauged ,(self.batch_size,16,5,5,32)), [0,2,3,4,1])
    b2_not_gauged  = weights_2[:,:,-1]
    w3_not_gauged  = weights_3[:,:,0:-1]
    w3_not_gauged  = tf.transpose(tf.reshape(w3_not_gauged ,(self.batch_size,8,7,7,16)), [0,2,3,4,1])
    b3_not_gauged  = weights_3[:,:,-1]
    w4_not_gauged  = weights_4[:,0:80]
    w4_not_gauged  = tf.reshape(w4_not_gauged ,(self.batch_size,8,10))
    b4_not_gauged  = weights_4[:,80:]


    required_scale = self.layer1_filter_size*self.layer1_filter_size*self.number_of_channels+1
    scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w1_not_gauged), [1, 2, 3]) + tf.square(b1_not_gauged))/required_scale+self.zero_fixer)
    w1 = w1_not_gauged / (self.ExpandDims(scale_factor, [1, 2, 3]) + self.zero_fixer)
    b1 = b1_not_gauged / (scale_factor + self.zero_fixer)
    w2 = w2_not_gauged * self.ExpandDims(scale_factor, [1, 2, 4])
    w1 = tf.identity(w1,'w1')
    b1 = tf.identity(b1, 'b1')

    required_scale = self.layer2_filter_size*self.layer2_filter_size*self.layer1_size+1
    scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w2), [1, 2, 3]) + tf.square(b2_not_gauged))/required_scale+self.zero_fixer)
    w2 = w2 / (self.ExpandDims(scale_factor, [1, 2, 3]) + self.zero_fixer)
    b2 = b2_not_gauged / (scale_factor + self.zero_fixer)
    w3 = w3_not_gauged * self.ExpandDims(scale_factor, [1, 2, 4])
    w2 = tf.identity(w2, 'w2')
    b2 = tf.identity(b2, 'b2')

    required_scale = (self.image_height / (self.layer1_pool_size * self.layer2_pool_size)) * (self.image_width / (self.layer1_pool_size * self.layer2_pool_size)) * self.layer2_size + 1
    scale_factor = tf.sqrt((tf.reduce_sum(tf.square(w3), [1, 2, 3]) + tf.square(b3_not_gauged))/required_scale+self.zero_fixer)
    w3 = w3 / (self.ExpandDims(scale_factor, [1, 2, 3]) + self.zero_fixer)
    b3 = b3_not_gauged / (scale_factor + self.zero_fixer)
    w4 = w4_not_gauged * self.ExpandDims(scale_factor, [2])
    w3 = tf.identity(w3, 'w3')
    b3 = tf.identity(b3, 'b3')

    required_softmax_bias = 0.0
    softmax_bias_diff = tf.reduce_sum(b4_not_gauged, 1) - required_softmax_bias
    b4 = b4_not_gauged - self.ExpandDims(softmax_bias_diff,[1])
    w4 = tf.identity(w4, 'w4')
    b4 = tf.identity(b4, 'b4')
    fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, layer1_pool_size, layer1_pool_size, 1],strides=[1, layer1_pool_size, layer1_pool_size, 1],padding='SAME')
    c_layer1_output = tf.map_fn(fn, elems=[x, w1, b1], dtype=tf.float32,name='layer1_output')

    fn = lambda u: tf.nn.max_pool(tf.nn.relu(tf.nn.conv2d(u[0], u[1], padding='SAME', strides=[1, 1, 1, 1]) + u[2]),ksize=[1, layer2_pool_size, layer2_pool_size, 1],strides=[1, layer2_pool_size, layer2_pool_size, 1],padding='SAME')
    c_layer2_output = tf.map_fn(fn, elems=[c_layer1_output, w2, b2], dtype=tf.float32,name='layer2_output')

    c_layer3_output = tf.nn.relu(tf.reduce_sum(tf.expand_dims(c_layer2_output, -1) * tf.expand_dims(w3,1), axis=(2, 3, 4)) + tf.expand_dims(b3,1),name='layer3_output')

    c_layer4_output = tf.identity(tf.reduce_sum(tf.expand_dims(c_layer3_output, -1) * tf.expand_dims(w4,1), axis=2) + tf.expand_dims(b4,1),name='layer4_output')

    probs = tf.nn.softmax(c_layer4_output,name='probabilities')
    preds = tf.argmax(probs, axis=2, name='prediction')
    correct_predictions = tf.equal(preds, tf.argmax(y, axis=2), name='correct_prediction')
    accuracy2 = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), axis=1, name='accuracy').numpy()

    accuracy2 = np.reshape(accuracy2, -1)
    accuracy1 = np.reshape(accuracy1, -1)
    
    x_axis.append(np.arange(0,300))
    x_axis = np.linspace(0,1,self.batch_size) 
    x_axis = np.reshape(x_axis, -1)
   
    
    fig, ax = plt.subplots()
    #line1, = ax.plot(x_axis, y_loss1, label="Loss of linear combination in weight space", color="g")
    #line2, = ax.plot(x_axis, y_loss2, label="Loss of linear input combination", color="m")
    #line1.set_dashes([2, 2, 10, 2])
    #line2.set_dashes([2, 2, 10, 2])
    #x_new = 
    #a_BSpline_1 = interpolate.interp1d(x_axis,accuracy1, kind = "cubic")
    #a_BSpline_2 = interpolate.interp1d(x_axis,accuracy2, kind = "cubic")
    #accuracy1 = a_BSpline_1(x_new)
    #accuracy2 = a_BSpline_2(x_new)
    ax.plot(x_axis, accuracy1, label="linear combination in weight space", color="g")
    ax.plot(x_axis, accuracy2, label="linear input combination", color="m")
    #ax.legend(loc="lower center")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=5)

    plt.xlabel("percentage")
    plt.ylabel("accuracy")
    save_path = 'Image_classification/experiments/path_1000_new.png'
    plt.savefig(save_path)
    plt.show()

    
    #acc2.append(net.GetAccuracyWithForcedWeights(sess,x,y,w1,b1, w2,b2, w3,b3, w4,b4))

     

    
  def plot_accu(self):

      csv_path = ""
      save_path = ""
      algorithm = "" 
      with open(csv_path) as csvfile:
          print(csv_path)
          reader = csv.DictReader(csvfile)
          xs = []
          ys1 = []
          ys2 = []
          for row in reader:
              xs.append(int(row['generation']))
              ys1.append(float(row['reward']))
              ys2.append(float(row['loss']))

          fig, ax = plt.subplots()
          ax2 = ax.twinx()

          ax.plot(xs, ys1, label=algorithm)
          ax2.plot(xs, ys2, label=algorithm)
          ax.set(xlabel='generation', ylabel='reward')
          ax2.set(xlabel='generation', ylabel='loss')
          ax.legend()
          ax2.legend()
          ax.grid()
          ax2.grid()

          save_dir = os.path.dirname(save_path)
          if not os.path.exists(save_dir):
              os.makedirs(save_dir)

          fig.savefig(save_path)


  def plot_accuracy(self):

      num = [1000, 10_000, 100_000]
      ylabel = 'validation accuracy'
      path = 'Image_classification/experiments_2'
      xlabel = 'step'
      #titel = ''
      xs, ys = [], []

      for i in num:
          csv_path = path +'/performance' + str(i) + '.csv' 
          y = []
          with open(csv_path) as csvfile:
              print(csv_path)
              reader = csv.DictReader(csvfile)
              
              for row in reader:
                  if i == 1000:
                      xs.append(int(row['step']))
                  y.append(float(row[ylabel]))
          ys.append(y)

      #ys1, ys2, ys3, ys4, ys5, ys6 = ys

      ys1, ys2, ys3 = ys

      plt.plot(xs, ys1, c='b')
      plt.plot(xs, ys2, c='g')
      plt.plot(xs, ys3, c='r')
      #plt.plot(xs, ys4, c='c')
      #plt.plot(xs, ys5, c='m')
      #plt.plot(xs, ys6, c='y')
      plt.xlabel(str(xlabel))
      plt.ylabel(str(ylabel))
      plt.axes().yaxis.grid()
      #plt.yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
      #plt.title(str(titel))
      plt.legend(['1.000', '10.000', '100.000'])#, '3e-5', '1e-5'])#, '6e-6', '3e-6'])

      save_dir = os.path.dirname(path)
      if not os.path.exists(save_dir):
          os.makedirs(save_dir)
      save_path = path + '/image_acccuracy_new.png'
      plt.savefig(save_path)
      plt.show()
