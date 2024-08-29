'''
Created on 21 août 2024

@author: SSM9
'''
from NeuronNetwork import NeuronNetwork

import Cost_functions
import Activation_functions

if __name__ == '__main__':
    
    #TODO : Load training data
    #'''
    dummy_input = [[2/20,1/20],
                   [5/20,4/20],
                   [11/20,3/20],
                   [15/20,5/20],
                   [15/20,10/20],
                   [3/20,10/20],
                   [8/20,10/20],
                   [11/20,8/20],
                   [9/20,6/20],
                   [15/20,2/20],
                   [8/20,1/20],
                   [12/20,5/20],
                   [11/20,5/20]
                   ]
    
    #dummy_result =  [[1,1,0,0,0,1,0,1,1,0,1,0,1]] #forme chelou
    dummy_result =  [[1,1,1,0,0,1,0,0,1,0,1,0,0]] #droite          
    #'''
    '''
    dummy_input = [[1,0]]
    dummy_result =  [[0]]
    #'''
    '''
    dummy_input = [[1,0],[0,1],[1,1],[0,0]]
    dummy_result =  [[1,1,0,0]]
    #'''
    #dummy_result =  [[1,1,1,0],[1,1,1,0]]
    
    
    MachineLearningModel = NeuronNetwork(2, 1, 2, 12, 0.02,
                                         Cost_functions.binary_cross_entropy, 
                                         Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                         Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                         Activation_functions.sigmoidLogisticFun, Activation_functions.der_sigmoidLogisticFun)

    
    
    #input_data = [1/1000,8/1000]
        
    #print("input data is "+str(input_data))
    #MachineLearningModel.executeModel(input_data)
    
    #print("output_data layer output is")
    #MachineLearningModel.output_layer.printLayerOutput()
    
    for i in range (0, 1000, 1) :
        print("let's test model training")
        MachineLearningModel.supervisedModelTrainingEpochExecution(dummy_input, dummy_result)
    
    #'''
    input_data = [16/20,2/20]
    print("input data is "+str(input_data))
    result = MachineLearningModel.executeModel(input_data)
    print("output data is "+str(result[0].output_value))
    
    input_data = [6/20,7/20]
    print("input data is "+str(input_data))
    result = MachineLearningModel.executeModel(input_data)
    print("output data is "+str(result[0].output_value))
    
    input_data = [11/20,6/20]
    print("input data is "+str(input_data))
    result = MachineLearningModel.executeModel(input_data)
    print("output data is "+str(result[0].output_value))
    #'''
    '''
    input_data = [1,1]
    print("input data is "+str(input_data))
    result = MachineLearningModel.executeModel(input_data)
    print("output data is "+str(result[0].output_value))
    #'''

    pass