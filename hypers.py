#
# hyperparameters
#

from keras.layers.advanced_activations import PReLU
from keras.layers import Activation
from keras import optimizers

# default values
hyperparameters = {}
hyperparameters['resnet_activation'] = PReLU
hyperparameters['optimization_algorithm'] = optimizers.Adam
hyperparameters['batch_size'] = 64
hyperparameters['learning_rate'] = 0.01
hyperparameters['learning_rate_decay'] = 0.95
hyperparameters['epochs'] = 50

# range of values
hyperparameter_values = {}
hyperparameter_values['resnet_activation'] = [PReLU]
hyperparameter_values['optimization_algorithm'] = [#optimizers.RMSprop,
                                                   optimizers.Adagrad,
                                                   #optimizers.Adadelta,
                                                   optimizers.Adam,
                                                   optimizers.Adamax#,
                                                   #optimizers.Nadam - ojo, no tiene decay
                                                   ]
hyperparameter_values['batch_size'] = [32, 64]
hyperparameter_values['learning_rate'] = [0.01, 0.001] # logarithmic scale
hyperparameter_values['learning_rate_decay'] = [0.99, 0.9, 0.85, 0.8]
hyperparameter_values['epochs'] = [1, 10, 20]

