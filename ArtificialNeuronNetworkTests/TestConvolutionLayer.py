'''
Created on 25 oct. 2024

@author: SSM9
'''
import unittest

from ArtificialNeuronNetwork.CNNComponents.Kernel import Kernel
from ArtificialNeuronNetwork.CNNComponents.Maxpooling import MaxPooling   
from ArtificialNeuronNetwork import Activation_functions
from ArtificialNeuronNetwork.CNNComponents.ConvolutionLayer import ConvolutionLayer


import imageio.v3 as iio
import numpy as np


class Test(unittest.TestCase):


    def testConvLayer(self):
        
        filename = "E:\\users\\sami\\trash\\avatar.jpg"
        
        kernelParam0 = [0,0,0,-1,0,0,0,0,0,-1,0,0,6,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0]
        kernelParam1 = [0,0,0,0,-1,0,0,0,0,0,-1,0,0,6,0,0,-1,0,0,0,0,0,-1,0,0,0,0]
        kernelParam2 = [0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,6,0,0,-1,0,0,0,0,0,-1,0,0,0]
        
        
        img = iio.imread(filename, pilmode='RGB')
        
        #In case we have a gray scale image, it is necessary to format data for the kernels
        numberOfChannels = 3
        img = img.reshape((len(img),len(img[0]),numberOfChannels))        
        
        #Create kernels manually          
        mykernelR = Kernel([2,3,numberOfChannels], Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
        mykernelR.synaptic_weights = kernelParam0
        mykernelG = Kernel([2,3,numberOfChannels], Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
        mykernelG.synaptic_weights = kernelParam1
        mykernelB = Kernel([2,3,numberOfChannels], Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
        mykernelB.synaptic_weights = kernelParam2      
        
        #Process image to generate feature maps manually
        mykernelR.processInputs(img)
        mykernelG.processInputs(img)
        mykernelB.processInputs(img)
        
        
        #MaxPoolingLayer manually
        myPoolingLayerR = MaxPooling([2,4])
        myPoolingLayerG = MaxPooling([2,4])
        myPoolingLayerB = MaxPooling([2,4])
        
        myPoolingLayerR.processInputs(mykernelR.featureMap)
        myPoolingLayerG.processInputs(mykernelG.featureMap)
        myPoolingLayerB.processInputs(mykernelB.featureMap)
        
                
        #Concatenate manually the feature maps
        
        result = np.concatenate((myPoolingLayerR.maxPoolingOutput,myPoolingLayerG.maxPoolingOutput), axis=2)
        result = np.concatenate((result,myPoolingLayerB.maxPoolingOutput), axis=2)
        
        result = np.array(result)
        result = result.astype(np.uint8)
        
        #Create a convolution Layer
        myConvLayer = ConvolutionLayer( layerSize=3, kernelsShape = [2,3,numberOfChannels],
                                        activation_function = Activation_functions.reLUFun,
                                        der_activation_fun = Activation_functions.der_reLUFun,
                                        maxPooling = True, maxPoolingShape = [2,4])
        
        myConvLayer.kernels[0].synaptic_weights = kernelParam0
        myConvLayer.kernels[1].synaptic_weights = kernelParam1
        myConvLayer.kernels[2].synaptic_weights = kernelParam2
        
        #Process image with convolution layer
        myConvLayer.processInputs(img)
        
        result2 = myConvLayer.convLayerOutput.astype(np.uint8)
        
        self.assertTrue((result == result2).all()) 
        
        pass


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testConvLayer']
    unittest.main()