'''
Created on 16 août 2024

@author: SSM9
'''
import numpy as np

from ArtificialNeuronNetwork import Activation_functions
import json
from ArtificialNeuronNetwork.Activation_functions import der_neuronInhibitionFun
from enum import Enum

class Neuron(object):
    
    '''
    This class describes a neuron architecture based on perceptron definition
    TODO : add a way to manage features like inhibit backward propagation
    '''
    synaptic_weights = None
    previous_synaptic_weights = None #used for backward propagation computation
    activation_function= None
    der_activation_function = None
    
    bias = None
    
    input_values=None
    output_value=None
    error=None
    optimizer = None
    momentum_param = None
    gamma = None
    
    
    def __init__(self, synaptic_weights=[], 
                 activation_function=Activation_functions.neuronInhibitionFun, 
                 der_activation_function=der_neuronInhibitionFun, 
                 bias=0, 
                 optimizer = None,
                 gamma = 0) :
        
        self.synaptic_weights = synaptic_weights
        self.previous_synaptic_weights = synaptic_weights
        self.activation_function = activation_function
        self.der_activation_function = der_activation_function
        self.bias = bias
        
        self.input_values = np.empty(len(self.synaptic_weights), dtype=float)
        
        self.output_value = 0
            
        self.error = 0
        
        if(optimizer==None):
            self.optimizer = Optimizer.SGD
        else:
            self.optimizer = optimizer
        
        self.momentum_param = 0
        self.gamma = gamma
    
    def processInputs(self):
        '''
        Faire la somme
        Utiliser la fonction d'activation
        '''
        
        if len(self.input_values)!=len(self.synaptic_weights):
            print('input_vector len ('+str(len(self.input_values))+
                  ') is not compatible with the number of neuron dendrites ('+str(len(self.synaptic_weights))+')')
        else:
            combination_function_result = self.bias;
            '''
            for v,w in zip(self.input_values,self.synaptic_weights):
                combination_function_result = combination_function_result+np.multiply(v,w)
            '''
            combination_function_result+=np.dot(self.input_values,self.synaptic_weights)
            self.output_value = self.activation_function(combination_function_result)
                   
            
    #Compute error from Next Layer TODO : refactor all this capability
    def computeErrorFromNextLayer(self, next_layer_weights_associated_to_self,next_layer_errors):
        self.error = np.float64(0)
        
        self.error = np.dot(next_layer_weights_associated_to_self,next_layer_errors)
        
        return 
    
    #Refresh parameter using the selected optimizer algorith
    def getParameterNewValue(self,paramOldValue,correction_coeff,grad_error):
        
        paramNewValue=0
        
        if(self.optimizer == Optimizer.SGD):
            paramNewValue = paramOldValue - correction_coeff*grad_error
            
        elif (self.optimizer == Optimizer.MOMENTUM):
            self.momentum_param = self.gamma*self.momentum_param + correction_coeff*grad_error
            paramNewValue = paramOldValue - self.momentum_param
            
        elif(self.optimizer == Optimizer.NAG):
            #Nesterov Accelerated Gradient
            self.momentum_param = self.gamma*self.momentum_param + correction_coeff*grad_error*(paramOldValue-self.gamma*self.momentum_param)
            paramNewValue = paramOldValue - self.momentum_param
            
        else:
            paramNewValue = paramOldValue - correction_coeff*grad_error
            
        return paramNewValue
        
    #Update weights and bias from error computed from next layer TODO : refactor all this TODO : refactor all this capability and infrastructure
    def updateParametersFromNextLayer(self, next_layer_errors, correction_coeff, next_layer_weights_associated_to_self):
        
        #save previous weights before refreshing their value
        self.previous_synaptic_weights = self.synaptic_weights
        
        self.computeErrorFromNextLayer(next_layer_weights_associated_to_self, next_layer_errors)
                
        #refresh each weight of the neuron using the gradient method
        for i in range (0, len(self.synaptic_weights),1):
            
            grad_err_weight_i = self.error*(self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)+self.bias))*self.input_values[i]
            
            self.synaptic_weights[i]=self.getParameterNewValue(self.synaptic_weights[i], correction_coeff, grad_err_weight_i)
        
        #refresh bias
        grad_err_weight_bias = self.error*(self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)+self.bias))
        self.bias = self.getParameterNewValue(self.bias, correction_coeff, grad_err_weight_bias)
           
        return
    
    #Update weights and bias from error computed from output error (only for output layer neurons) TODO : refactor all this capability
    def updateParametersFromOutputError(self, error, correction_coeff):
        
        #save previous weights before refreshing their value
        self.previous_synaptic_weights = self.synaptic_weights
        
        self.error = np.float64(error)
                        
        #refresh each weight of the neuron using the gradient method
        for i in range (0, len(self.synaptic_weights),1):
                        
            grad_err_weight_i = self.error*(self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)+self.bias))*self.input_values[i]
            
            self.synaptic_weights[i]= self.getParameterNewValue(self.synaptic_weights[i], correction_coeff, grad_err_weight_i)
        
        #refresh bias
        grad_err_weight_bias = self.error*(self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)+self.bias))
        self.bias = self.getParameterNewValue(self.bias, correction_coeff, grad_err_weight_bias)
           
        return
    
    def verbose(self):
        print("I am a neuron with the following parameters \n synaptic weights="+str(self.synaptic_weights)
              +"\n bias="+str(self.bias))

    def getHyperParameters(self, directCall=True):
        
        if(directCall==True):
            hyperParams = json.dumps(NeuronHyperParameters(self.synaptic_weights, 
                                                           self.bias, 
                                                           self.activation_function.__name__, 
                                                           self.der_activation_function.__name__,
                                                           self.optimizer.name,
                                                           self.gamma).__dict__)
        else:
            hyperParams = NeuronHyperParameters(self.synaptic_weights, 
                                                self.bias, 
                                                self.activation_function.__name__, 
                                                self.der_activation_function.__name__,
                                                self.optimizer.name,
                                                self.gamma).__dict__
            
        return hyperParams
    
    def loadHyperParameters(self, hyperParamsJson):
        
        hyperParamsReceiverObject = json.loads(hyperParamsJson)
        
        self.synaptic_weights = hyperParamsReceiverObject["synaptic_weights"]
        self.previous_synaptic_weights = self.synaptic_weights
        self.activation_function = Activation_functions.getFunctionByName(hyperParamsReceiverObject["activation_function"])
        self.der_activation_function = Activation_functions.getFunctionByName(hyperParamsReceiverObject["der_activation_function"])
        self.bias = hyperParamsReceiverObject["bias"]
        self.optimizer = Optimizer[hyperParamsReceiverObject["optimizer"]]
        self.gamma=hyperParamsReceiverObject["gamma"]
        
        self.input_values = np.empty(len(self.synaptic_weights), dtype=float)
       
        self.output_value = 0
            
        self.error = 0
        self.momentum_param = 0
        
        
        
        
class NeuronHyperParameters(object):
    
    synaptic_weights = None
    bias = None
    activation_function= None
    der_activation_function = None
    
    optimizer = None
    gamma = None
        
    def __init__(self, synaptic_weights, 
                 bias, activation_function, 
                 der_activation_function,
                 optimizer,
                 gamma):
        
        self.synaptic_weights=[]
        for w in synaptic_weights :
            self.synaptic_weights.append(w)
            
        self.bias = bias
        self.optimizer=optimizer
        self.gamma=gamma
        self.activation_function = activation_function
        self.der_activation_function = der_activation_function        
        
        
class Optimizer(Enum):
    SGD = 0,
    MOMENTUM = 1,
    NAG = 2,
    RMSProp = 3,
    ADAM = 4,
    ADAMW = 5
    
