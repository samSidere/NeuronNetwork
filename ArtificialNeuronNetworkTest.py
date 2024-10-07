
'''
Created on 21 août 2024

@author: SSM9
'''
from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork

from ArtificialNeuronNetwork import Cost_functions
from ArtificialNeuronNetwork import Activation_functions
from ArtificialNeuronNetwork.Neuron import Optimizer

import numpy as np

if __name__ == '__main__':
    
    #TODO : Load training data
    #'''
    dummy_input = np.array([[2,1],
                   [5,4],
                   [11,3],
                   [15,5],
                   [15,10],
                   [3,10],
                   [8,10],
                   [11,8],
                   [9,6],
                   [15,2],
                   [8,1],
                   [12,5],
                   [11,5]
                   ])
    
    dummy_input = dummy_input*(1/20)
    
    #dummy_result =  [[1],[1],[0],[0],[0],[1],[0],[1],[1],[0],[1],[0],[1]] #forme chelou
    dummy_result =  [[1],[1],[1],[0],[0],[1],[0],[0],[1],[0],[1],[0],[0]] #droite          
   
    
    MachineLearningModel = NeuronNetwork(2, 1, 2, 12, 0.001,
                                         Cost_functions.binary_cross_entropy,
                                         Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                         Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                         Activation_functions.sigmoidLogisticFun, Activation_functions.der_sigmoidLogisticFun,
                                         False,Optimizer.MOMENTUM, 0.01)

    
    
    #input_data = [1/1000,8/1000]
        
    #print("input data is "+str(input_data))
    #MachineLearningModel.executeModel(input_data)
    
    #print("output_data layer output is")
    #MachineLearningModel.output_layer.printLayerOutput()
    
    for i in range (0, 5000, 1) :
        print("let's do model training")
        performance = MachineLearningModel.supervisedModelTrainingEpochExecution(dummy_input, dummy_result)
        
        if performance < 9.9e-2:
            break
    
    #'''
    input_data = [16/20,2/20]
    print("input data is "+str(input_data))
    result = MachineLearningModel.executeModel(input_data)
    print("output data is "+str(result[0].output_value))
    if(result[0].output_value<1e-2):
        print("test 1 is a success")
    print()
    
    input_data = [6/20,7/20]
    print("input data is "+str(input_data))
    result = MachineLearningModel.executeModel(input_data)
    print("output data is "+str(result[0].output_value))
    if(1-result[0].output_value<1e-2):
        print("test 2 is a success")
    print()
    
    input_data = [11/20,6/20]
    print("input data is "+str(input_data))
    result = MachineLearningModel.executeModel(input_data)
    print("output data is "+str(result[0].output_value))
    if(np.round(result[0].output_value)<1e-2):
        print("test 3 is a success")
    print()
    #'''
    '''
    input_data = [1,1]
    print("input data is "+str(input_data))
    result = MachineLearningModel.executeModel(input_data)
    print("output data is "+str(result[0].output_value))
    #'''

    pass