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
    TODO = implement back propagation
    TODO = implement test using conv_tests example
    TODO = Attention à la surcharge des algos
    '''
    #Shape is an array (dimension, side length, number of channels)
    kernelDimension = None
    kernelSideLength = None
    numberOfChannels = None
    padding = None
    featureMap = None
    dfeatureMap = None #will store the derivative of feature map to include activation function
    
    formattedInputData = None
    

    def __init__(self, shape = [1,3,1],
                 activation_function = Activation_functions.linearActivationFun,
                 der_activation_fun = Activation_functions.der_linearActivationFun,
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
        
        synaptic_weights = np.random.randn((self.kernelSideLength**self.kernelDimension)*self.numberOfChannels)
                
        Neuron.__init__(self, synaptic_weights=synaptic_weights, 
                        activation_function = activation_function, 
                        der_activation_function = der_activation_fun, 
                        bias = 0, 
                        optimizer = optimizer, 
                        beta1 = beta1, 
                        beta2 = beta2) 
        
        #depending on the chosen optimizer the param table shape will be different (TBD optimizer params might need to be reset)
        if(self.optimizer == Optimizer.ADAM):
            if self.kernelDimension == 1 :
                self.optimizer_params = np.zeros((2, self.kernelSideLength, self.numberOfChannels), dtype=float)
            elif self.kernelDimension == 2 :
                self.optimizer_params = np.zeros((2, self.kernelSideLength, self.kernelSideLength, self.numberOfChannels), dtype=float)
            
        else :
            if self.kernelDimension == 1 :
                self.optimizer_params = np.zeros((self.kernelSideLength, self.numberOfChannels), dtype=float)
            elif self.kernelDimension == 2 :
                self.optimizer_params = np.zeros((self.kernelSideLength, self.kernelSideLength, self.numberOfChannels), dtype=float)
            
            
    def processInputs(self, inputData):
                
        #Generate kernel input based on the padding paramater and inputData shape
        self.padInputData(inputData)
        
        #Initialize  feature map => dimension feature map is dimension inputData-1
        featureMapShape = list(inputData.shape)

        featureMapShape[len(featureMapShape)-1]=1

        featureMapShape = tuple(featureMapShape)

        #each axis size depends on the input data axis size
        self.featureMap = np.zeros(featureMapShape)
        self.dfeatureMap = np.zeros(featureMapShape)
                
        #formattedInputData is the real input of the Kernel
        #The Kernel will parse the input data from and generate feature map
        #for each pixel -> map surrounding pixels to kernel neuron inputs -> process input -> put resulting pixel in the feature map
        if self.kernelDimension == 1 :
            for i in range (0, len(self.featureMap), 1):
                kernel = self.formattedInputData[i:i+self.kernelSideLength]
                
                self.mapKernel2Neuron(kernel)
        
                Neuron.processInputs(self)
                
                #Save pixel in feature map
                self.featureMap[i][0]= self.output_value
                self.dfeatureMap[i][0]= Neuron.computeDerivativeOfInput(self)
                
        elif self.kernelDimension == 2 :
            
            for i in range (0, len(self.featureMap), 1):
                for j in range (0, len(self.featureMap[0]), 1):
                    kernel = self.formattedInputData[i:i+self.kernelSideLength,j:j+self.kernelSideLength]
                
                    self.mapKernel2Neuron(kernel)
        
                    Neuron.processInputs(self)
                
                    #Save pixel in feature map
                    self.featureMap[i][j][0]= self.output_value
                    self.dfeatureMap[i][j][0]= Neuron.computeDerivativeOfInput(self)
            
        else:
            print("number of dimensions not taken in charge")
            return            
            
        return
    
    def padInputData(self, inputData):

        pad_width = []

        for i in range (0,self.kernelDimension,1):
            pad_width.append((self.padding,self.padding))
    
        pad_width.append((0,0))
        
        self.formattedInputData = np.pad(inputData,pad_width, 'constant', constant_values=(0))
        
        return self.formattedInputData
    
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
    
    
    def getParameterNewValue(self, paramOldValue,correction_coeff,grad_error):
        
        
        if(self.optimizer == Optimizer.SGD):
            paramNewValue = paramOldValue - correction_coeff*grad_error
            
        elif (self.optimizer == Optimizer.MOMENTUM):
            self.optimizer_params = self.beta1*self.optimizer_params + correction_coeff*grad_error
            paramNewValue = paramOldValue - self.optimizer_params
            
        elif(self.optimizer == Optimizer.NAG):
            #Nesterov Accelerated Gradient
            '''
            TDB - Improve NAG implementation based on the official definition of the optimizer
            Here’s the gradient descent stage:
            ϕt+1=θt−εt∇f(θt)
            
            followed by the momentum-like stage:
            θt+1=ϕt+1+μt(ϕt+1−ϕt)
            
            TODO Fix this algorithm implementation
            
            '''
            phi_t_1 = paramOldValue - correction_coeff*grad_error
            paramNewValue = phi_t_1 + self.beta1*(phi_t_1 - self.optimizer_params)
            
            self.optimizer_params = phi_t_1
        
        elif(self.optimizer == Optimizer.RMSProp):
            self.optimizer_params = self.beta2*self.optimizer_params+(1-self.beta2)*np.pow(grad_error,2)
            paramNewValue = paramOldValue - (correction_coeff/np.sqrt(self.optimizer_params+1e-8))*grad_error
            
        elif(self.optimizer == Optimizer.ADAM):
            
            #Compute momentum
            self.optimizer_params[0] = self.beta1*self.optimizer_params[0] + (1-self.beta1)*grad_error
            mt_cap = self.optimizer_params[0]/(1-self.beta1**self.num_steps)
            
            
            #Compute RMS propagation
            self.optimizer_params[1] = self.beta2*self.optimizer_params[1]+(1-self.beta2)*(grad_error)**2
            vt_cap = self.optimizer_params[1]/(1-self.beta2**self.num_steps)
            
            paramNewValue = paramOldValue - ((correction_coeff*mt_cap)/np.sqrt(vt_cap+1e-8))
                            
        else:
            paramNewValue = paramOldValue - correction_coeff*grad_error
            
        return paramNewValue
        
    #Update weights and bias from error computed from next layer TODO : refactor all this TODO : refactor all this capability and infrastructure
    
    
    '''
    TODO : validate simple back propagation - introduce activation function - propagate error to previous layer - introduce optimizers
    '''
    def updateParametersFromErrorGrad(self, errorMap, correction_coeff):
        
        #save previous weights before refreshing their value
        self.previous_synaptic_weights = self.synaptic_weights
        
        W = self.synaptic_weights
        
        #product of each element of the error map with the derivatives of feature maps to introduce activation function
        errorMap = errorMap*np.squeeze(self.dfeatureMap)
        
        #Reshape kernel
        if self.kernelDimension == 1 :
            W= np.reshape(W, (self.kernelSideLength, self.numberOfChannels))
            
            # Retrieving dimensions from W's shape
            f = self.kernelSideLength
            
            # Retrieving dimensions from dH's shape
            width = len(errorMap)
            
            # Initializing dX, dW with the correct shapes
            dX = np.zeros(self.formattedInputData.shape)
            dW = np.zeros(W.shape)
            
            for w in range(width):
                    dX[w:w+f] += W * errorMap[w]#dH[h,w]
                    dW +=  self.formattedInputData[w:w+f] * errorMap[w]#X[h:h+f, w:w+f] * dH[h,w]
            
            
        elif self.kernelDimension == 2 :
            W= np.reshape(W, (self.kernelSideLength, self.kernelSideLength, self.numberOfChannels))
            
            # Retrieving dimensions from W's shape
            f = self.kernelSideLength
            
            # Retrieving dimensions from dH's shape
            (height, width) = errorMap.shape
            
            # Initializing dX, dW with the correct shapes
            dX = np.zeros(self.formattedInputData.shape)
            dW = np.zeros(W.shape)
            
            for h in range(height):
                for w in range(width):
                    dX[h:h+f, w:w+f] += W * errorMap[h,w]#dH[h,w]
                    dW +=  self.formattedInputData[h:h+f, w:w+f] * errorMap[h,w]#X[h:h+f, w:w+f] * dH[h,w]
            
        else :
            print("number of dimensions not taken in charge")
            return
        
        '''
        TODO : implement optimization algorithms
        TODO : manage dX to propagate error to previous Layer
        '''
       
        W=self.getParameterNewValue(W, correction_coeff, dW)
        #W = W-correction_coeff*dW
        
        self.synaptic_weights = W.flatten()
                
        return 