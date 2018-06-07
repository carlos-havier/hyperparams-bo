
# A small exercise in Bayessian Optimization of Hyperparameters

This is a toy examle of BO hyperparameter search, given the little time to run it and limited GPU computing resources.

* BO search with different hyperparameters
* Using Keras and Tensorflow
* Ensemble of three networks
* All network share hyperparameters (not optimum)
* __[Model architecture from ](https://github.com/kkweon/mnist-competition)__
* Data augmentation (rotation, shear, resizing)

## Notebook

More info in the notebook hyper-bo.ipnb.

## Run

#### Training and evaluation
```bash
python hysearchbo.py
```

#### Change parameters
```nano hypers.py 10 # hyperparameters and their possible values
```


