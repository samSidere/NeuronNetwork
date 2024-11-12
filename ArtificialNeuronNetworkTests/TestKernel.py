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
        
        print(mykernel.featureMap)
        
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
        
        print(mykernel.featureMap)
        
        self.assertTrue((mykernel.featureMap == expected_result).all())
                
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()