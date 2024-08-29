'''
Created on 21 août 2024

@author: SSM9
'''
import numpy as np

import Parameters

'''

Why Cost Function is Important
The main goal of any neural network is to make accurate predictions. A cost function helps to quantify how far the neural network’s predictions are from the actual values. It is a measure of the error between the predicted output and the actual output. The cost function plays a crucial role in training a neural network. During the training process, the neural network adjusts its weights and biases to minimize the cost function. The goal is to find the minimum value of the cost function, which corresponds to the best set of weights and biases that make accurate predictions.


Types of Cost Functions
There are different types of cost functions, and the choice of cost function depends on the type of problem being solved. Here are some commonly used cost functions:

'''

'''
Mean Squared Error (MSE)
The mean squared error is one of the most popular cost functions for regression problems. It measures the average squared difference between the predicted and actual values. The formula for MSE is:

MSE = (1/n) * Σ(y - ŷ)^2

Where:

n is the number of samples in the dataset
y is the actual value
ŷ is the predicted value
'''
def mean_squared_error(y_true, y_pred):
    y_true = np.array(y_true,dtype=np.float64)
    y_pred = np.array(y_pred,dtype=np.float64)
    return np.mean(np.square(y_true - y_pred))

'''
Binary Cross-Entropy 
The binary cross-entropy cost function is used for binary classification problems. It measures the difference between the predicted and actual values in terms of probabilities. The formula for binary cross-entropy is:

Binary Cross-Entropy = - (1/n) * Σ(y * log(ŷ) + (1 - y) * log(1 - ŷ))

Where:

n is the number of samples in the dataset
y is the actual value (0 or 1)
ŷ is the predicted probability (between 0 and 1)

'''
def binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true,dtype=np.float64)
    y_pred = np.array(y_pred,dtype=np.float64)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))



