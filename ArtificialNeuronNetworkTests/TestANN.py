'''
Created on 19 nov. 2024

@author: SSM9
'''
import unittest

from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork

from ArtificialNeuronNetwork import Cost_functions
from ArtificialNeuronNetwork import Activation_functions
from ArtificialNeuronNetwork.Neuron import Optimizer

import numpy as np

class Test(unittest.TestCase):


    def testArtificialNeuronNetwork(self):
        
        
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
    
        dummy_result =  [[1],[1],[1],[0],[0],[1],[0],[0],[1],[0],[1],[0],[0]] #droite
        
        MachineLearningModel = NeuronNetwork(2, 1, 2, 12, 0.01,
                                             Cost_functions.binary_cross_entropy,
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun,
                                             Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                             Activation_functions.sigmoidLogisticFun, Activation_functions.der_sigmoidLogisticFun,
                                             False,Optimizer.ADAM, 0.9,0.999)

    
    
        for i in range (0, 5000, 1) :
            print("let's do model training")
            performance = MachineLearningModel.supervisedModelTrainingEpochExecution(dummy_input, dummy_result)
            #performance = MachineLearningModel.TDB_supervisedModelTrainingByBatchEpochExecution(dummy_input, dummy_result)
        
            if performance < 9.5e-2:
                break
    
        input_data = [16/20,2/20]
        print("input data is "+str(input_data))
        result = MachineLearningModel.executeModel(input_data)
        print("output data is "+str(result[0].output_value))
        if(result[0].output_value<1e-2):
            print("test 1 is a success")
            print()
            
        self.assertLess(result[0].output_value, 1e-2)
    
        input_data = [6/20,7/20]
        print("input data is "+str(input_data))
        result = MachineLearningModel.executeModel(input_data)
        print("output data is "+str(result[0].output_value))
        if(1-result[0].output_value<1e-2):
            print("test 2 is a success")
            print()
            
        self.assertLess(1-result[0].output_value, 1e-2)
    
        input_data = [11/20,6/20]
        print("input data is "+str(input_data))
        result = MachineLearningModel.executeModel(input_data)
        print("output data is "+str(result[0].output_value))
        if(result[0].output_value<1e-2):
            print("test 3 is a success")
            print()
            
        self.assertLess(result[0].output_value, 1e-2)
        
        
        pass
    
    
    def testMultiClassNeuronNetwork(self):
        
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
        
        MachineLearningModel = NeuronNetwork(3, 2, 8, 8, 1e-2,
                                         Cost_functions.categorical_cross_entropy, 
                                         Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                         Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                         Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun,
                                         True, Optimizer.ADAM, 0.9,0.999)
        
        for i in range (0, 500, 1) :
            print("For epoch : "+str(i))
            performance = MachineLearningModel.supervisedModelTrainingEpochExecution(dummy_input, dummy_result)
        
            if performance < 1e-3:
                break
        
        
        for i in range(0,len(dummy_result),1) :
            
            input_data = dummy_input[i]
            print("input data is "+str(input_data))
            MachineLearningModel.executeModel(input_data)
            result = MachineLearningModel.getNetworkOutput()
            print("output data is "+str(result))
            print("expected result is "+str(dummy_result[i]))
            
            if np.argmax(result) == np.argmax(dummy_result[i]):
                print("test "+str(i)+" is a success")
            print()
            self.assertEqual(np.argmax(result), np.argmax(dummy_result[i]))
        
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()