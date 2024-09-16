'''
Created on 16 août 2024

@author: SSM9
'''
from ArtificialNeuronNetwork.Neuron import Neuron
import numpy as np
from ArtificialNeuronNetwork import Activation_functions

if __name__ == '__main__':
    
    input_data = [0,1,1]
    
    perceptron = Neuron([1,1,1],Activation_functions.sigmoidLogisticFun,Activation_functions.der_sigmoidLogisticFun,0)

    perceptron.input_values = input_data
    
    for i in range (0,1000,1) :
        perceptron.processInputs()
    
        print("output is : "+str(perceptron.output_value))
    
        #error = 1/2*(0-perceptron.output_value)**2
        
        error = (0 - perceptron.output_value)
    
        perceptron.updateParametersFromOutputError(error, 0.2)
    
    out =  perceptron.output_value   
   
    pass