'''
Created on 23 oct. 2024

@author: SSM9
'''

from ArtificialNeuronNetwork import Activation_functions
from ArtificialNeuronNetwork.Neuron import Optimizer

from ArtificialNeuronNetwork.CNNComponents.Kernel import Kernel
from ArtificialNeuronNetwork.CNNComponents.Maxpooling import MaxPooling

import numpy as np


class ConvolutionLayer(object):
    '''
    The convolution Layer will apply a filter to the input in order to detect features that will allow a fully connected layer to attach the input to a class
    '''
    
    kernels = None
    layerSize = None
    kernelsDimension = None
    
    maxPooling = None
    maxPoolingNodes = None
    
    convLayerOutput = None
    
    #TBD traiter 1xN_Channels->1xN_Channels(SplitNConcat)
    #TBD Tester les features

    def __init__(self, layerSize=1, kernelsShape = [1,3,1],
                 activation_function = Activation_functions.reLUFun,
                 der_activation_fun = Activation_functions.der_reLUFun,
                 optimizer = Optimizer.ADAM, 
                 beta1 = 0.9, 
                 beta2 = 0.999,
                 maxPooling = False,
                 maxPoolingShape = [1,2]
                 ):
        
        
        self.layerSize = layerSize
        self.kernelsDimension = kernelsShape[0]
        
        self.kernels=[]
        
        self.maxPooling = maxPooling
        
        if(self.maxPooling == True):
            self.maxPoolingNodes = []
        else:
            self.maxPoolingNodes = None
        
        for i in range(0, self.layerSize, 1):
            self.kernels.append(Kernel(kernelsShape,
                                         activation_function,
                                         der_activation_fun,
                                         optimizer,
                                         beta1,
                                         beta2))
            if(self.maxPooling == True):
                self.maxPoolingNodes.append(MaxPooling(maxPoolingShape))
            
        
        self.convLayerOutput=[]
        
        return
    
    def processInputs(self, inputData):
        
        self.convLayerOutput=[]
        
        #Generate feature map associated to each kernel
        #Generate output by concatenating all the feature maps
        
        if self.maxPooling == True:
            for kernel, maxPoolingNode in zip(self.kernels, self.maxPoolingNodes) :
                kernel.processInputs(inputData)
                maxPoolingNode.processInputs(kernel.featureMap)
                
                if len(self.convLayerOutput) == 0:
                    self.convLayerOutput = maxPoolingNode.maxPoolingOutput
                else:
                    self.convLayerOutput = np.concatenate((self.convLayerOutput, maxPoolingNode.maxPoolingOutput), axis=self.kernelsDimension)
                
        else:
            for kernel, in zip(self.kernels) :
                kernel.processInputs(inputData)
                
                if len(self.convLayerOutput) == 0:
                    self.convLayerOutput = kernel.featureMap
                else:
                    self.convLayerOutput = np.concatenate((self.convLayerOutput, kernel.featureMap), axis=self.kernelsDimension)
            
       
        
        return