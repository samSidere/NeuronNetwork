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
    
    def flattenConvLayerOutput(self):
        return self.convLayerOutput.flatten()
    
    '''
    TODO : manage max pooling
    '''
    def backPropagationThroughLayer(self, nextLayer, correction_coeff):
        
        #Build errorMaps for each kernel
        #Compute error common gradient member for each pixel of each generated feature map
        #Get slices and rearrange them to build errorMaps
        #Do backpropagation through each kernel
         
        next_layer_errors_associated_to_each_featuremaps_pixel = np.zeros(self.convLayerOutput.size)
                
        #get for each pixel of each generated feature map of the current layer a table of weights of the next layer associated to its synaptic connection
        for i in range (0,np.shape(next_layer_errors_associated_to_each_featuremaps_pixel)[0],1):
            for j in range (0, nextLayer.layerSize, 1):
                next_layer_errors_associated_to_each_featuremaps_pixel[i] += nextLayer.neurons[j].previous_synaptic_weights[i]*nextLayer.neurons[j].error
        
        #get slices and rearrange them to build errorMaps
        offset = 0
        
        #errorMap init
        errorMap = np.zeros(self.convLayerOutput.shape)
        
        for i in range (0,self.layerSize,1):
            
            #extract and reshape slice
            errorMap[i] = np.reshape(next_layer_errors_associated_to_each_featuremaps_pixel[offset:offset+self.convLayerOutput[i].size],self.convLayerOutput[i].shape)
            
            self.kernels[i].updateParametersFromErrorGrad(errorMap[i],correction_coeff)
            
            #update offset for next slice
            offset += self.convLayerOutput[i].size
        
        return 
    