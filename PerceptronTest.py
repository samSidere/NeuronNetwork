'''
Created on 16 août 2024

@author: SSM9
'''
from Neuron import Neuron
import numpy as np
import Activation_functions

if __name__ == '__main__':
    
    perceptron = Neuron([1,1,1],Activation_functions.sigmoidLogisticFun,Activation_functions.der_sigmoidLogisticFun,0)

    perceptron.input_values =[0,1,1]
    
    for i in range (0,1000,1) :
        perceptron.processInputs()
    
        print("output is : "+str(perceptron.output_value))
    
        #error = 1/2*(0-perceptron.output_value)**2
        
        error = (1 - perceptron.output_value)
    
        perceptron.updateParametersFromOutputError(error, 0.2)
    
        
   
    pass