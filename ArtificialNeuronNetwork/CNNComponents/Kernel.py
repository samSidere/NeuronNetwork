'''
Created on 18 oct. 2024

@author: SSM9
'''

import copy as cp
import numpy as np

from ArtificialNeuronNetwork.Neuron import Neuron

from ArtificialNeuronNetwork.Neuron import Optimizer

from  ArtificialNeuronNetwork import Activation_functions 

class Kernel(Neuron):
    '''
    classdocs
    '''
    #Shape is an array (dimension, side length, number of channels)
    kernelDimension = None
    kernelSideLength = None
    numberOfChannels = None
    padding = None
    featureMap = None
    

    def __init__(self, shape = [1,3,1],
                 activation_function = Activation_functions.reLUFun,
                 der_activation_fun = Activation_functions.der_reLUFun,
                 optimizer = Optimizer.ADAM, 
                 beta1 = 0.9, 
                 beta2 = 0.999):
        
        #Build the parameters from shape
        #Check if side len is odd
        if(shape[1]%2==0):
            print("kernel side len not valid and has been increased by 1")
            shape[1]+=1
                
        self.kernelDimension = shape[0]
        self.kernelSideLength = shape[1]
        self.numberOfChannels = shape[2]
                            
        self.padding = int(self.kernelSideLength/2)
        
        synaptic_weights = np.ones((self.kernelSideLength**self.kernelDimension)*self.numberOfChannels)
                
        Neuron.__init__(self, synaptic_weights=synaptic_weights, 
                        activation_function = activation_function, 
                        der_activation_function = der_activation_fun, 
                        bias = 0, 
                        optimizer = optimizer, 
                        beta1 = beta1, 
                        beta2 = beta2) 
        
    def processInputs(self, inputData):
        
        #Generate kernel input based on the padding paramater and inputData shape
        formattedInputData = self.padInputData(inputData)
        
        #Initialize  feature map => dimension feature map is dimension inputData-1
        featureMapShape = list(inputData.shape)

        featureMapShape[len(featureMapShape)-1]=1

        featureMapShape = tuple(featureMapShape)

        #each axis size depends on the input data axis size
        self.featureMap = np.zeros(featureMapShape)
                
        #formattedInputData is the real input of the Kernel
        #The Kernel will parse the input data from and generate feature map
        #for each pixel -> map surrounding pixels to kernel neuron inputs -> process input -> put resulting pixel in the feature map
        if self.kernelDimension == 1 :
            for i in range (0, len(self.featureMap), 1):
                kernel = formattedInputData[i:i+self.kernelSideLength]
                
                self.mapKernel2Neuron(kernel)
        
                Neuron.processInputs(self)
                
                #Save pixel in feature map
                self.featureMap[i][0]= self.output_value
                
        elif self.kernelDimension == 2 :
            
            for i in range (0, len(self.featureMap), 1):
                for j in range (0, len(self.featureMap[0]), 1):
                    kernel = formattedInputData[i:i+self.kernelSideLength,j:j+self.kernelSideLength]
                
                    self.mapKernel2Neuron(kernel)
        
                    Neuron.processInputs(self)
                
                    #Save pixel in feature map
                    self.featureMap[i][j][0]= self.output_value
            
        else:
            print("number of dimensions not taken in charge")
            return            
            
        return
    
    def padInputData(self, inputData):

        pad_width = []

        for i in range (0,self.kernelDimension,1):
            pad_width.append((self.padding,self.padding))
    
        pad_width.append((0,0))
        
        formattedInputData = np.pad(inputData,pad_width, 'constant', constant_values=(0))
        
        return formattedInputData
    
    def iterativeParsing(self, array):
        if array.ndim > 1 :
            #Parse next dimension
            for side in array:
                self.iterativeParsing(side)
        else :
            #Return slice
            print("Pixel content is"+str(array))  
        
        return
    
    def mapKernel2Neuron(self, kernel):
        
        self.input_values = kernel.flatten()
        
        return    