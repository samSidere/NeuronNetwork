'''
Created on 23 oct. 2024

@author: SSM9
'''
import unittest

import numpy as np

from ArtificialNeuronNetwork.CNNComponents.Maxpooling import MaxPooling


class Test(unittest.TestCase):


    def testMaxPooling1D(self):
        
        myMaxPoolingWindow = MaxPooling([1,2])
                
        inputData = np.array([[3],[2],[5],[1],[6],[4]])
        
        myMaxPoolingWindow.processInputs(inputData)
        
        expected_result = np.array([[3],[5],[6]])
        
        print(myMaxPoolingWindow.maxPoolingOutput)
        
        self.assertTrue((myMaxPoolingWindow.maxPoolingOutput == expected_result).all())
        
        pass


    def testMaxPooling2D(self):
        
        myMaxPoolingWindow = MaxPooling([2,2])
                
        inputData = np.array([[[3],[8],[5],[1],[6],[3]],
                               [[0],[2],[2],[1],[6],[4]],
                               [[3],[2],[11],[3],[3],[1]],
                               [[3],[2],[5],[1],[9],[2]]])
        
        myMaxPoolingWindow.processInputs(inputData)
        
        expected_result = np.array([[[8],[5],[6]],
                                   [[3],[11],[9]]])
        
        print(myMaxPoolingWindow.maxPoolingOutput)
        
        self.assertTrue((myMaxPoolingWindow.maxPoolingOutput == expected_result).all())
        
        pass

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()