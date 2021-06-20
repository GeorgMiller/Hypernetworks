import hypernetwork_trainer
import tensorflow as tf
import numpy as np
import random
# I used tensorflow 2.4.0 to run this so just check if its the same
print('tensorflow version: {}'.format(tf.__version__))
tf.keras.backend.clear_session()


# This is the main file of the experiments. It calls the hypernetwork trainer file where all the actual 'work' is done. 
#Before the actual code, tensorflow is set and the seeds are specified.

seed = 8734
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
trainer = hypernetwork_trainer.Trainer(seed)

# The train_hypernetwork function is called two times with different values for lamBda
# The weights and logging of the experiments are saved for later usage
'''
lamBda1 = 1_000
lamBda2 = 10_000
lamBda3 = 100_000
trainer.train_hypernetwork(lamBda1)
trainer = hypernetwork_trainer.Trainer(seed)
trainer.train_hypernetwork(lamBda2)
trainer = hypernetwork_trainer.Trainer(seed)
trainer.train_hypernetwork(lamBda3)
'''

# Then the train_mnist_networks function is called with a list of 10 seeds as input 
# The weights and logging of the experiments are saved for later usage

seed_c = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#trainer.train_mnist_networks(seed_c)

# After the training, the weight paths are created for the lamBda1 network
#trainer.weight_path()

# The cosine similarity of the flattend network weights of individually trained networks
# as well as the generated one are then compared.
trainer.cosine_matrices()    

# Generate a histogram plot of the generated networks
#trainer.pca()

# Plot the training curves
#trainer.plot_accuracy()