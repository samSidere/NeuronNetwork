'''
Created on 1 oct. 2024

@author: SSM9
'''

from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork

from ArtificialNeuronNetwork import Cost_functions
from ArtificialNeuronNetwork import Activation_functions

from ArtificialNeuronNetwork.Neuron import Optimizer

import numpy as np

if __name__ == '__main__':
    
    #TODO : Load training data
    '''
    dummy_input = np.array([[1,1,1,1],
                   [1,1,1,0],
                   [1,1,0,1],
                   [1,1,0,0],
                   [1,0,1,1],
                   [1,0,1,0],
                   [1,0,0,1],
                   [1,0,0,0],
                   [0,1,1,1],
                   [0,1,1,0],
                   [0,1,0,1],
                   [0,1,0,0],
                   [0,0,1,1],
                   [0,0,1,0],
                   [0,0,0,1],
                   [0,0,0,0]
                   ])
    
    dummy_input = dummy_input*(1/10)
        
    dummy_result = np.array([[1,0,0,0,0,0],
                   [1,0,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,0,1,0,0,0],
                   [0,0,0,1,0,0],
                   [0,0,0,0,1,0],
                   [0,0,0,0,0,1],
                   [0,1,0,0,0,0],
                   [1,0,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,1,0,0,0,0],
                   [0,0,1,0,0,0],
                   [0,0,0,1,0,0],
                   [1,0,0,0,0,0],
                   [1,0,0,0,0,0]
                   ])
                   
    '''
    dummy_input = np.array([[1,1,1],
                            [1,1,0],
                            [1,0,1],
                            [1,0,0],
                            [0,1,1],
                            [0,1,0],
                            [0,0,1]
                   ])
    
    dummy_result = np.array([[1,0],
                             [1,0],
                             [1,0],
                             [1,0],
                             [0,1],
                             [0,1],
                             [0,1]
                   ])
    '''
    dummy_result = np.array([[1,0,0],
                             [1,0,0],
                             [1,0,0],
                             [0,1,0],
                             [0,1,0],
                             [0,0,1],
                             [0,0,1]
                   ])
    ''    
    dummy_result = np.array([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,0],
                             [0,0,0,1],
                             [0,0,0,1]
                   ])
    
    #'''
    
    print('Do you want to load an previously created Network?')
    filename = input("Insert your network parameters file path")    
    
    if filename =="":
        MachineLearningModel = NeuronNetwork(3, 2, 8, 8, 1e-2,
                                         Cost_functions.categorical_cross_entropy, 
                                         Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                         Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                         Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun,
                                         True, Optimizer.ADAM, 0.9,0.999)
    else :
        MachineLearningModel = NeuronNetwork()
        MachineLearningModel.loadNetworkParameterFromfile(filename)
    
    
    
    
    for i in range (0, 500, 1) :
        print("For epoch : "+str(i))
        performance = MachineLearningModel.supervisedModelTrainingEpochExecution(dummy_input, dummy_result)
        #performance = MachineLearningModel.TDB_supervisedModelTrainingByBatchEpochExecution(dummy_input, dummy_result)
        
        if performance < 1e-3:
            break
    
    if filename =="":
        MachineLearningModel.saveNetworkParameterIntofile("E:\\users\\sami\\trash\\CNNtest.json")
    else :
        MachineLearningModel.saveNetworkParameterIntofile(filename)
    
    #'''
    
    input_data = dummy_input[0]
    print("input data is "+str(input_data))
    MachineLearningModel.executeModel(input_data)
    result = MachineLearningModel.getNetworkOutput()
    print("output data is "+str(result))
    print("expected result is "+str(dummy_result[0]))
    if np.argmax(result) == np.argmax(dummy_result[0]):
        print("test 1 is a success")
    print()
    
    input_data = dummy_input[1]
    print("input data is "+str(input_data))
    MachineLearningModel.executeModel(input_data)
    result = MachineLearningModel.getNetworkOutput()
    print("output data is "+str(result))
    print("expected result is "+str(dummy_result[1]))
    if np.argmax(result) == np.argmax(dummy_result[1]):
        print("test 2 is a success")
    print()
    
    input_data = dummy_input[2]
    print("input data is "+str(input_data))
    MachineLearningModel.executeModel(input_data)
    result = MachineLearningModel.getNetworkOutput()
    print("output data is "+str(result))
    print("expected result is "+str(dummy_result[2]))
    if np.argmax(result) == np.argmax(dummy_result[2]):
        print("test 3 is a success")
    print()
    
    input_data = dummy_input[3]
    print("input data is "+str(input_data))
    MachineLearningModel.executeModel(input_data)
    result = MachineLearningModel.getNetworkOutput()
    print("output data is "+str(result))
    print("expected result is "+str(dummy_result[3]))
    if np.argmax(result) == np.argmax(dummy_result[3]):
        print("test 4 is a success")
    print()
    
    input_data = dummy_input[4]
    print("input data is "+str(input_data))
    MachineLearningModel.executeModel(input_data)
    result = MachineLearningModel.getNetworkOutput()
    print("output data is "+str(result))
    print("expected result is "+str(dummy_result[4]))
    if np.argmax(result) == np.argmax(dummy_result[4]):
        print("test 5 is a success")
    print()
    
    input_data = dummy_input[5]
    print("input data is "+str(input_data))
    MachineLearningModel.executeModel(input_data)
    result = MachineLearningModel.getNetworkOutput()
    print("output data is "+str(result))
    print("expected result is "+str(dummy_result[5]))
    if np.argmax(result) == np.argmax(dummy_result[5]):
        print("test 6 is a success")
    print()
    
    input_data = dummy_input[6]
    print("input data is "+str(input_data))
    MachineLearningModel.executeModel(input_data)
    result = MachineLearningModel.getNetworkOutput()
    print("output data is "+str(result))
    print("expected result is "+str(dummy_result[6]))
    if np.argmax(result) == np.argmax(dummy_result[6]):
        print("test 7 is a success")
    print()
    #'''
   

    pass