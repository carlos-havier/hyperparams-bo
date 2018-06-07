
# An exercise in Bayesian Optimization of Hyperparameters of DCNNs

This is an examle of BO (Bayesian Optimization) hyperparameter search to maximace the accuracy of DCNNs. Given the little time to run it and limited GPU computing resources, it is a toy example, but can be used with a broader search with different time constraints and access to GPUs. More details in the notebook __[hyper_mnist.ipynb](https://colab.research.google.com/github/carloshavier/hyperparams-bo/blob/master/hyper_mnist.ipynb)__.

* BO search with different hyperparameters.
* Using Keras and Tensorflow.
* All network share hyperparameters (not optimum but faster search).
* Model architecture from __[here](https://github.com/kkweon/mnist-competition)__. Ensemble of three networks, ResNet50 and two smaller variants of VGG.
* Data augmentation (rotation, projection, resizing).

## Notebook

More info in the notebook __[hyper_mnist.ipynb](https://colab.research.google.com/github/carloshavier/hyperparams-bo/blob/master/hyper_mnist.ipynb)__.

## Run

#### Training and evaluation
```bash
python hysearchbo.py
```

#### Change parameters
```bash
nano hypers.py # hyperparameters and their possible values
```

## Search Result

![BO search of best hyperparameters](https://raw.githubusercontent.com/carloshavier/hyperparams-bo/master/experiment-results/experiment.png)

The maximum accuracy was obtained using:

- Non-linear activation function: PReLU.
- Epochs: 10.
- Batch-size: 32.
- Optimization algorithm: Adam.
- Learning rate: 0.01
- Decay of the learning rate: 0.99.

The ensemble accuracy was 93.55%. The raw data is __[here](https://raw.githubusercontent.com/carloshavier/hyperparams-bo/master/experiment-results/ex-7-6-18-01-small.txt)__.

Augmenting the number of epochs, it was possible to attain an accuracy of 98.30% at $50$ epochs.
