'''
Created on 22 oct. 2024

@author: SSM9
'''
import unittest

import numpy as np

from ArtificialNeuronNetwork.CNNComponents.Kernel import Kernel
from ArtificialNeuronNetwork import Activation_functions


class Test(unittest.TestCase):


    def testKernelMethods1D1Channel(self):
        
        mykernel = Kernel([1,3,1], Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
        
        mykernel.synaptic_weights = [-1,0,1]
        
        inputData = np.array([[0],[0],[1],[1],[1],[0],[0]])
        
        mykernel.processInputs(inputData)
        
        expected_result = np.array([[0],[1],[1],[0],[-1],[-1],[0]])
        
        print(mykernel.featureMap)
        
        self.assertTrue((mykernel.featureMap == expected_result).all())
                
        pass
    
    def testKernelMethods1D3Channel(self):
        
        mykernel = Kernel([1,3,3], Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
        
        mykernel.synaptic_weights = [-1,1,1,0,1,1,1,1,1]
        
        inputData = np.array([[0,0,0],[0,0,0],[1,1,1],[1,1,1],[1,1,1],[0,0,0],[0,0,0]])
        
        mykernel.processInputs(inputData)
        
        expected_result = np.array([[0],[3],[5],[6],[3],[1],[0]])
        
        print(np.squeeze(mykernel.featureMap))
        
        self.assertTrue((mykernel.featureMap == expected_result).all())
                
        pass
    
    def testKernelMethods2D(self):
        
        mykernel = Kernel([2,3,1], Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
        
        mykernel.synaptic_weights = [-1,0,1,-1,0,1,-1,0,1]
        
        inputData = np.array([[[0],[0],[0],[0],[0],[0]],
                              [[0],[0],[0],[0],[0],[0]],
                              [[0],[1],[1],[1],[1],[0]],
                              [[0],[1],[1],[1],[1],[0]],
                              [[0],[1],[1],[1],[1],[0]],
                              [[0],[0],[0],[0],[0],[0]],
                              [[0],[0],[0],[0],[0],[0]]])
        
        mykernel.processInputs(inputData)
        
        expected_result = np.array([[[0],[0],[0],[0],[0],[0]],
                                    [[1],[1],[0],[0],[-1],[-1]],
                                    [[2],[2],[0],[0],[-2],[-2]],
                                    [[3],[3],[0],[0],[-3],[-3]],
                                    [[2],[2],[0],[0],[-2],[-2]],
                                    [[1],[1],[0],[0],[-1],[-1]],
                                    [[0],[0],[0],[0],[0],[0]]])
        
        print(np.squeeze(mykernel.featureMap))
        
        self.assertTrue((mykernel.featureMap == expected_result).all())
                
        pass
    
    
    def testKernelMethodsBackPropagationWithoutActivation(self):
        
        mykernel = Kernel([2,3,1], Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
                
        inputData = np.array([[[0],[0],[0],[0],[0],[0]],
                              [[0],[0],[0],[0],[0],[0]],
                              [[0],[1],[1],[1],[1],[0]],
                              [[0],[1],[1],[1],[1],[0]],
                              [[0],[1],[1],[1],[1],[0]],
                              [[0],[0],[0],[0],[0],[0]],
                              [[0],[0],[0],[0],[0],[0]]])
                
        expected_result = np.array([[[0],[0],[0],[0],[0],[0]],
                                    [[1],[1],[0],[0],[-1],[-1]],
                                    [[2],[2],[0],[0],[-2],[-2]],
                                    [[3],[3],[0],[0],[-3],[-3]],
                                    [[2],[2],[0],[0],[-2],[-2]],
                                    [[1],[1],[0],[0],[-1],[-1]],
                                    [[0],[0],[0],[0],[0],[0]]])
        
        
        true_synaptic_weights = [-1,0,1,-1,0,1,-1,0,1]      
        
        
        alpha = 0.01
    
        for i in range (0,5000,1):
            
            mykernel.processInputs(inputData)
            
            #mean squared error loss function used to perform back propagation
            dL = np.array(-2*(expected_result-mykernel.featureMap))
            LossfunctionResult = np.mean(np.power(expected_result-mykernel.featureMap,2)) 
                
            mykernel.updateParametersFromErrorGrad(np.reshape(dL,(7,6)), alpha)
                
           
        print("error without activation function "+str(LossfunctionResult))
        self.assertLess(LossfunctionResult, 1e-5)
                
        pass
    
    def testKernelMethodsBackPropagationWithActivation(self):
        
        mykernel = Kernel([2,3,1], Activation_functions.parametricReLUFun, Activation_functions.der_parametricReLUFun)
                
        inputData = np.array([[[0],[0],[0],[0]],
                              [[1],[1],[1],[1]],
                              [[1],[1],[1],[1]],
                              [[1],[1],[1],[1]],
                              [[0],[0],[0],[0]]])
                
        expected_result = np.array([[[1],[0],[0],[-1]],
                                    [[2],[0],[0],[-2]],
                                    [[3],[0],[0],[-3]],
                                    [[2],[0],[0],[-2]],
                                    [[1],[0],[0],[-1]]])
                
        alpha = 0.02
    
        for i in range (0,5000,1):
            
            mykernel.processInputs(inputData)
            
            #mean squared error loss function used to perform back propagation
            dL = np.array(-2*(expected_result-mykernel.featureMap))
            LossfunctionResult = np.mean(np.power(expected_result-mykernel.featureMap,2)) 
                
            mykernel.updateParametersFromErrorGrad(np.reshape(dL,(5,4)), alpha)
                
            
        print("error with activation function "+str(LossfunctionResult))
        print(np.round(np.squeeze(mykernel.featureMap)))
        print(np.squeeze(mykernel.featureMap))
        self.assertLess(LossfunctionResult, 1e-5)
                
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()