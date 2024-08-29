'''
Created on 19 août 2024

@author: SSM9
'''

import numpy as np

import Parameters
from cmath import nan

'''
Linear Activation Function
The linear activation function, also known as "no activation," or "identity function" (multiplied x1.0), is where the activation is proportional to the input.

The function doesn't do anything to the weighted sum of the input, it simply spits out the value it was given.

However, a linear activation function has two major problems :

It’s not possible to use backpropagation as the derivative of the function is a constant and has no relation to the input x. 
All layers of the neural network will collapse into one if a linear activation function is used. No matter the number of layers in the neural network, the last layer will still be a linear function of the first layer. So, essentially, a linear activation function turns the neural network into just one layer.
'''
def linearActivationFun(x):
    return x

def der_linearActivationFun(x):
    return 1

'''
Binary Step Function
Binary step function depends on a threshold value that decides whether a neuron should be activated or not. 

The input fed to the activation function is compared to a certain threshold; if the input is greater than it, then the neuron is activated, else it is deactivated, meaning that its output is not passed on to the next hidden layer.

Here are some of the limitations of binary step function:

It cannot provide multi-value outputs—for example, it cannot be used for multi-class classification problems. 
The gradient of the step function is zero, which causes a hindrance in the backpropagation process.
'''
def binaryStepFun(x):
    if x < 0 :
        return 0
    else :
        return 1
    
    
def der_binaryStepFun(x):
    
    if x != 0 :
        return 0
    else :
        return nan




'''
Non-Linear Activation Functions
The linear activation function shown above is simply a linear regression model. 

Because of its limited power, this does not allow the model to create complex mappings between the network’s inputs and outputs. 

Non-linear activation functions solve the following limitations of linear activation functions:

They allow backpropagation because now the derivative function would be related to the input, and it’s possible to go back and understand which weights in the input neurons can provide a better prediction.
They allow the stacking of multiple layers of neurons as the output would now be a non-linear combination of input passed through multiple layers. Any output can be represented as a functional computation in a neural network.
Now, let’s have a look at ten different non-linear neural networks activation functions and their characteristics. 
'''


'''
10 Non-Linear Neural Networks Activation Functions
'''
   
'''
Sigmoid / Logistic Activation Function 
This function takes any real value as input and outputs values in the range of 0 to 1. 

The larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to 0.0, as shown below.

Here’s why sigmoid/logistic activation function is one of the most widely used functions:

It is commonly used for models where we have to predict the probability as an output. Since probability of anything exists only between the range of 0 and 1, sigmoid is the right choice because of its range.
The function is differentiable and provides a smooth gradient, i.e., preventing jumps in output values. This is represented by an S-shape of the sigmoid activation function. 
The limitations of sigmoid function are discussed below:

The derivative of the function is f'(x) = sigmoid(x)*(1-sigmoid(x)). 

image of the derivative _____/----|_______

As we can see from the above Figure, the gradient values are only significant for range -3 to 3, and the graph gets much flatter in other regions. 

It implies that for values greater than 3 or less than -3, the function will have very small gradients. As the gradient value approaches zero, the network ceases to learn and suffers from the Vanishing gradient problem.

The output of the logistic function is not symmetric around zero. So the output of all the neurons will be of the same sign. This makes the training of the neural network more difficult and unstable.

'''
   
def sigmoidLogisticFun(x):
    return 1/(1+np.exp(-x))

def der_sigmoidLogisticFun(x):
    return sigmoidLogisticFun(x)*(1-sigmoidLogisticFun(x))

'''
Tanh Function (Hyperbolic Tangent)
Tanh function is very similar to the sigmoid/logistic activation function, and even has the same S-shape with the difference in output range of -1 to 1. In Tanh, the larger the input (more positive), the closer the output value will be to 1.0, whereas the smaller the input (more negative), the closer the output will be to -1.0.

Advantages of using this activation function are:

The output of the tanh activation function is Zero centered; hence we can easily map the output values as strongly negative, neutral, or strongly positive.
Usually used in hidden layers of a neural network as its values lie between -1 to; therefore, the mean for the hidden layer comes out to be 0 or very close to it. It helps in centering the data and makes learning for the next layer much easier.
Have a look at the gradient of the tanh activation function to understand its limitations.

As you can see— it also faces the problem of vanishing gradients similar to the sigmoid activation function. Plus the gradient of the tanh function is much steeper as compared to the sigmoid function.

 Note:  Although both sigmoid and tanh face vanishing gradient issue, tanh is zero centered, and the gradients are not restricted to move in a certain direction. Therefore, in practice, tanh nonlinearity is always preferred to sigmoid nonlinearity.

'''

def tanhFun(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def der_tanhFun(x):
    return 1-tanhFun(x)**2

'''
ReLU Function
ReLU stands for Rectified Linear Unit. 

Although it gives an impression of a linear function, ReLU has a derivative function and allows for backpropagation while simultaneously making it computationally efficient. 

The main catch here is that the ReLU function does not activate all the neurons at the same time. 

The neurons will only be deactivated if the output of the linear transformation is less than 0.

The advantages of using ReLU as an activation function are as follows:

Since only a certain number of neurons are activated, the ReLU function is far more computationally efficient when compared to the sigmoid and tanh functions.
ReLU accelerates the convergence of gradient descent towards the global minimum of the loss function due to its linear, non-saturating property.
The limitations faced by this function are:

The Dying ReLU problem, which I explained below.

__________|----------


The negative side of the graph makes the gradient value zero. Due to this reason, during the backpropagation process, the weights and biases for some neurons are not updated. This can create dead neurons which never get activated. 

All the negative input values become zero immediately, which decreases the model’s ability to fit or train from the data properly. 
Note: For building the most reliable ML models, split your data into train, validation, and test sets.

'''

def reLUFun(x):
    if x < 0 :
        return 0
    else :
        return x
        

def der_reLUFun(x):
    if x < 0 :
        return 0
    else :
        return 1

'''
Leaky ReLU Function
Leaky ReLU is an improved version of ReLU function to solve the Dying ReLU problem as it has a small positive slope in the negative area.

The advantages of Leaky ReLU are same as that of ReLU, in addition to the fact that it does enable backpropagation, even for negative input values. 

By making this minor modification for negative input values, the gradient of the left side of the graph comes out to be a non-zero value. Therefore, we would no longer encounter dead neurons in that region. 

Here is the derivative of the Leaky ReLU function. 
__________|----------

The limitations that this function faces include:

The predictions may not be consistent for negative input values. 
The gradient for negative values is a small value that makes the learning of model parameters time-consuming.

'''
def leakyReLUFun(x):
    
    if x < 0 :
        return 0.01*x
    else :
        return x   
   

def der_leakyReLUFun(x):
    
    if x < 0 :
        return 0.01
    else :
        return 1   

'''
Parametric ReLU Function
Parametric ReLU is another variant of ReLU that aims to solve the problem of gradient’s becoming zero for the left half of the axis. 

This function provides the slope of the negative part of the function as an argument a. By performing backpropagation, the most appropriate value of a is learnt.

The parameterized ReLU function is used when the leaky ReLU function still fails at solving the problem of dead neurons, and the relevant information is not successfully passed to the next layer. 

This function’s limitation is that it may perform differently for different problems depending upon the value of slope parameter a.
'''
def parametricReLUFun(x):
    
    threshold = Parameters.a*x
    
    if x < threshold :
        return threshold
    else :
        return x 
    
    
def der_parametricReLUFun(x):
    
    threshold = Parameters.a*x
    
    if x < threshold :
        return Parameters.a
    else :
        return 1    
   
'''
Exponential Linear Units (ELUs) Function
Exponential Linear Unit, or ELU for short, is also a variant of ReLU that modifies the slope of the negative part of the function. 

ELU uses a log curve to define the negativ values unlike the leaky ReLU and Parametric ReLU functions with a straight line.

ELU is a strong alternative for f ReLU because of the following advantages:

ELU becomes smooth slowly until its output equal to -α whereas RELU sharply smoothes.
Avoids dead ReLU problem by introducing log curve for negative values of input. It helps the network nudge weights and biases in the right direction.
The limitations of the ELU function are as follow:

It increases the computational time because of the exponential operation included
No learning of the ‘a’ value takes place
Exploding gradient problem
'''
def eLUFun(x):
    
    if x < 0 :
        return Parameters.alfaParam*(np.exp(x)-1)
    else :
        return x   
   
def der_eLUFun(x):
    
    if x < 0 :
        return Parameters.alfaParam+eLUFun(x)
    else :
        return 1   
   
'''
Softmax Function
Before exploring the ins and outs of the Softmax activation function, we should focus on its building block—the sigmoid/logistic activation function that works on calculating probability values. 

The output of the sigmoid function was in the range of 0 to 1, which can be thought of as probability. 

But—

This function faces certain problems.

Let’s suppose we have five output values of 0.8, 0.9, 0.7, 0.8, and 0.6, respectively. How can we move forward with it?

The answer is: We can’t.

The above values don’t make sense as the sum of all the classes/output probabilities should be equal to 1. 

You see, the Softmax function is described as a combination of multiple sigmoids. 

It calculates the relative probabilities. Similar to the sigmoid/logistic activation function, the SoftMax function returns the probability of each class. 

It is most commonly used as an activation function for the last layer of the neural network in the case of multi-class classification. 

Let’s go over a simple example together.

Assume that you have three classes, meaning that there would be three neurons in the output layer. Now, suppose that your output from the neurons is [1.8, 0.9, 0.68].

Applying the softmax function over these values to give a probabilistic view will result in the following outcome: [0.58, 0.23, 0.19]. 

The function returns 1 for the largest probability index while it returns 0 for the other two array indexes. Here, giving full weight to index 0 and no weight to index 1 and index 2. So the output would be the class corresponding to the 1st neuron(index 0) out of three.

You can see now how softmax activation function make things easy for multi-class classification problems.

'''
#Softmax is defined in the Neuron code as it changes the Neuron behavior
def softmax(x):    
    return x



'''
    Swish
It is a self-gated activation function developed by researchers at Google. 

Swish consistently matches or outperforms ReLU activation function on deep networks applied to various challenging domains such as image classification, machine translation etc. 

This function is bounded below but unbounded above i.e. Y approaches to a constant value as X approaches negative infinity but Y approaches to infinity as X approaches infinity.

Here are a few advantages of the Swish activation function over ReLU:

Swish is a smooth function that means that it does not abruptly change direction like ReLU does near x = 0. Rather, it smoothly bends from 0 towards values < 0 and then upwards again.
Small negative values were zeroed out in ReLU activation function. However, those negative values may still be relevant for capturing patterns underlying the data. Large negative values are zeroed out for reasons of sparsity making it a win-win situation.
The swish function being non-monotonous enhances the expression of input data and weight to be learnt.

'''

def swish(x):    
    return x*sigmoidLogisticFun(x)

def der_swish(x):    
    return sigmoidLogisticFun(x)+x*der_sigmoidLogisticFun(x)

'''
Gaussian Error Linear Unit (GELU)
The Gaussian Error Linear Unit (GELU) activation function is compatible with BERT, ROBERTa, ALBERT, and other top NLP models. This activation function is motivated by combining properties from dropout, zoneout, and ReLUs. 

ReLU and dropout together yield a neuron’s output. ReLU does it deterministically by multiplying the input by zero or one (depending upon the input value being positive or negative) and dropout stochastically multiplying by zero. 

RNN regularizer called zoneout stochastically multiplies inputs by one. 

We merge this functionality by multiplying the input by either zero or one which is stochastically determined and is dependent upon the input. We multiply the neuron input x by 

m ∼ Bernoulli(Φ(x)), where Φ(x) = P(X ≤x), X ∼ N (0, 1) is the cumulative distribution function of the standard normal distribution. 

This distribution is chosen since neuron inputs tend to follow a normal distribution, especially with Batch Normalization.

GELU nonlinearity is better than ReLU and ELU activations and finds performance improvements across all tasks in domains of computer vision, natural language processing, and speech recognition.

'''

def gELU(x):  
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))

'''
Scaled Exponential Linear Unit (SELU)
SELU was defined in self-normalizing networks and takes care of internal normalization which means each layer preserves the mean and variance from the previous layers. SELU enables this normalization by adjusting the mean and variance. 

SELU has both positive and negative values to shift the mean, which was impossible for ReLU activation function as it cannot output negative values. 

Gradients can be used to adjust the variance. The activation function needs a region with a gradient larger than one to increase it.

SELU has values of alpha α and lambda λ predefined. 

Here’s the main advantage of SELU over ReLU:

Internal normalization is faster than external normalization, which means the network converges faster.
SELU is a relatively newer activation function and needs more papers on architectures such as CNNs and RNNs, where it is comparatively explored.

'''

def sELU(x):
    if x < 0 :
        return Parameters.lambdaParam*Parameters.alfaParam*(np.exp(x)-1)
    else :  
        return Parameters.lambdaParam*x


'''
    neuron inhibition function 
'''
   
def neuronInhibitionFun(x):
    return 0

def der_neuronInhibitionFun(x):
    return 0

'''
    Mapping table function example
'''
def fun1(input_value):
    
    transfer_function = np.array([
                                    [0.0,0.0],
                                    [0.1,0.2],
                                    [0.3,0.6],
                                    [0.4,0.8],
                                    [0.5,1.0],
                                    [0.6,1.0],
                                    [0.7,1.0],
                                    [0.8,1.0],
                                    [0.9,1.0],
                                    [1.0,1.0]
                                    ])
    
    if transfer_function[0,0]>input_value or input_value>transfer_function[len(transfer_function)-1,0]:
        print("input_value:"+str(input_value)+" is out of range : ["+str(transfer_function[0,0])
                +","+str(transfer_function[len(transfer_function)-1,0])+"]")
        return 0                                                   

    for x in range(0,len(transfer_function),1):
        if input_value==transfer_function[x,0]:
            return transfer_function[x,1]
        elif transfer_function[x,0]<input_value and input_value<transfer_function[x+1,0]:
            dx=transfer_function[x+1,0]-transfer_function[x,0]
            dy=transfer_function[x+1,1]-transfer_function[x,1]
            bias=transfer_function[x,1]
            return (input_value-transfer_function[x,0])*(dy/dx)+bias
