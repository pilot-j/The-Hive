NeuralNet(PyTorch)
------
This directory contains implementations of neural networks using PyTorch and Python. 
What you will learn - Intermediate level coding with python, internals of PyTorch and tensor operations(vectorisation) and obviously using Pytorch to train neuralnetworks on datasets.

1) <a href='https://github.com/pilot-j/The-Hive/blob/main/Neuralnets/SynthWord.ipynb'>SynthWord</a> - Statistical Language model. Bi gram implementation. This is a weak model that can be used to generate random words which sound *name* like. This is a precursor to MLP implemention to random word generator that is based on research paper `Bengio et al. 2003`. This notebook contains both purely statistical modelling and a neural network equivalent of the same.

2) <a href='https://github.com/pilot-j/The-Hive/blob/main/Neuralnets/MLP_languagemodel.ipynb'>MLP_languagemodel</a> - Multi layer perceptron model of SynthWord. Introduces the concept of embeddings, non linearity, cross entropy and softmax. This type the previous dataset was divided into[ train, dev and test datasets]. Loss vs learning rate curves and concept of epochs was also explored. This is an excellent mini proect to implement for better understandinbg of *neuralnetwork structure, loss functions, backprop and hyperparameter tuning*.
3) nanoGPT - implemenetation of a small size generatively pretrained transformer.
