'''
Created on 23 oct. 2024

@author: SSM9
'''

import numpy as np

class MaxPooling(object):
    '''
    classdocs
    '''
    
    maxPoolingOutput = None
    
    poolDimension = None
    poolSize = None


    def __init__(self, shape = [1,2]):
        '''
        Constructor
        '''
        
        self.poolDimension = shape[0]
        self.poolSize = shape[1]
        
    
    def processInputs(self, inputData):        
        #Initialize  feature map => dimension feature map is dimension inputData-1
        outputShape = list(inputData.shape)
        
        for i in range (0, len(outputShape)-1, 1):
            outputShape[i]=int(outputShape[i]/self.poolSize)

        outputShape = tuple(outputShape)
                
        self.maxPoolingOutput = np.zeros(outputShape)
                
        if self.poolDimension == 1:
            
            if(len(inputData)%self.poolSize != 0):
                print("dimension of input incompatible with pooling data")
            
            for i in range (0, len(self.maxPoolingOutput), 1):
                poolingWindow = inputData[i*self.poolSize:i*self.poolSize+self.poolSize]
                
                self.maxPoolingOutput[i]= np.max(poolingWindow)
            
        elif self.poolDimension == 2 :
                        
            if(len(inputData)%self.poolSize != 0 or len(inputData[0])%self.poolSize != 0):
                print("dimension of input incompatible with pooling data")
                
            for i in range (0, len(self.maxPoolingOutput), 1):
                for j in range (0, len(self.maxPoolingOutput[0]), 1):
                    poolingWindow = inputData[i*self.poolSize:i*self.poolSize+self.poolSize,j*self.poolSize:j*self.poolSize+self.poolSize]
                
                    self.maxPoolingOutput[i][j]= np.max(poolingWindow)
            
        else:
            print("number of dimensions not taken in charge")
            return  
        
        return
        