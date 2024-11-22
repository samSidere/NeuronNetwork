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
    error_function_gradient = None
    
    bias = None
    
    input_values=None
    output_value=None
    error=None
    optimizer = None
    optimizer_params = None
    beta1 = None
    beta2 = None
    num_steps=None
    
    
    def __init__(self, synaptic_weights=[], 
                 activation_function=Activation_functions.neuronInhibitionFun, 
                 der_activation_function=der_neuronInhibitionFun, 
                 bias=0, 
                 optimizer = None,
                 beta1 = 0,
                 beta2 = 0,
                 error_function_gradient = None) :
        
        self.synaptic_weights = synaptic_weights
        self.previous_synaptic_weights = synaptic_weights
        self.activation_function = activation_function
        self.der_activation_function = der_activation_function
        
        if(error_function_gradient==None):
            self.error_function_gradient = ErrorFunctionGradient.MEAN_SQUARED_ERROR_LOSS
        else:
            self.error_function_gradient = error_function_gradient
        
        self.bias = bias
        
        self.input_values = np.empty(len(self.synaptic_weights), dtype=float)
        
        self.output_value = 0
            
        self.error = 0
        
        if(optimizer==None):
            self.optimizer = Optimizer.SGD
        else:
            self.optimizer = optimizer
        
        #depending on the chosen optimizer the param table shape will be different (TBD optimizer params might need to be reset)
        if(self.optimizer == Optimizer.ADAM):
            self.optimizer_params = np.zeros((2,len(self.synaptic_weights)+1), dtype=float)
        else :
            self.optimizer_params = np.zeros(len(self.synaptic_weights)+1, dtype=float)
        
        self.num_steps = 1
        
        self.beta1 = beta1
        self.beta2 = beta2
    
    def processInputs(self):
        '''
        Faire la somme
        Utiliser la fonction d'activation
        '''
        
        if len(self.input_values)!=len(self.synaptic_weights):
            print('input_vector len ('+str(len(self.input_values))+
                  ') is not compatible with the number of neuron dendrites ('+str(len(self.synaptic_weights))+')')
        else:
                        
            combination_function_result = np.dot(self.input_values,self.synaptic_weights)+self.bias
            
            self.output_value = self.activation_function(combination_function_result)
                   
    def computeDerivativeOfInput(self):
        
        if len(self.input_values)!=len(self.synaptic_weights):
            print('input_vector len ('+str(len(self.input_values))+
                  ') is not compatible with the number of neuron dendrites ('+str(len(self.synaptic_weights))+')')
            
            return
        else:
                        
            derivative = self.der_activation_function(np.dot(self.input_values,self.synaptic_weights)+self.bias)
            return derivative
                   
            
    #Compute error from Next Layer TODO : refactor all this capability
    def getErrorFromNextLayer(self, next_layer_weights_associated_to_self,next_layer_errors):
        self.error = np.float64(0)
        
        self.error = np.dot(next_layer_weights_associated_to_self,next_layer_errors)
        
        return 

    #Compute error gradient
    def computeErrorFunctionGradient(self, dendrite_index=None, error=0):
        
        if dendrite_index != len(self.synaptic_weights):
            DJ = error*(self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)
                                                     +self.bias))*self.input_values[dendrite_index]
          
        else:
            DJ = error*(self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)
                                                     +self.bias))
        
        return DJ

    #Refresh parameter using the selected optimizer algorith
    #TODO implement other optimizers
    def getParameterNewValue(self, dendrite_index, paramOldValue,correction_coeff,grad_error):
        
        paramNewValue=0
        
        if(self.optimizer == Optimizer.SGD):
            paramNewValue = paramOldValue - correction_coeff*grad_error
            
        elif (self.optimizer == Optimizer.MOMENTUM):
            self.optimizer_params[dendrite_index] = self.beta1*self.optimizer_params[dendrite_index] + correction_coeff*grad_error
            paramNewValue = paramOldValue - self.optimizer_params[dendrite_index]
            
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
            paramNewValue = phi_t_1 + self.beta1*(phi_t_1 - self.optimizer_params[dendrite_index])
            
            self.optimizer_params[dendrite_index] = phi_t_1
        
        elif(self.optimizer == Optimizer.RMSProp):
            self.optimizer_params[dendrite_index] = self.beta2*self.optimizer_params[dendrite_index]+(1-self.beta2)*(grad_error)**2
            paramNewValue = paramOldValue - (correction_coeff/np.sqrt(self.optimizer_params[dendrite_index]+1e-8))*grad_error
            
        elif(self.optimizer == Optimizer.ADAM):
            
            #Compute momentum
            self.optimizer_params[0][dendrite_index] = self.beta1*self.optimizer_params[0][dendrite_index] + (1-self.beta1)*grad_error
            mt_cap = self.optimizer_params[0][dendrite_index]/(1-self.beta1**self.num_steps)
            
            
            #Compute RMS propagation
            self.optimizer_params[1][dendrite_index] = self.beta2*self.optimizer_params[1][dendrite_index]+(1-self.beta2)*(grad_error)**2
            vt_cap = self.optimizer_params[1][dendrite_index]/(1-self.beta2**self.num_steps)
            
            paramNewValue = paramOldValue - ((correction_coeff*mt_cap)/np.sqrt(vt_cap+1e-8))
                            
        else:
            paramNewValue = paramOldValue - correction_coeff*grad_error
            
        return paramNewValue
        
    
    
    #Update weights and bias from error computed from next layer TODO : refactor all this TODO : refactor all this capability and infrastructure
    def updateParametersFromNextLayer(self, next_layer_errors, correction_coeff, next_layer_weights_associated_to_self):
        
        #save previous weights before refreshing their value
        self.previous_synaptic_weights = self.synaptic_weights
        
        self.getErrorFromNextLayer(next_layer_weights_associated_to_self, next_layer_errors)
                
        self.upateParameters(correction_coeff)
           
        return
    
    #Update weights and bias from error computed from output error (only for output layer neurons) TODO : refactor all this capability
    def updateParametersFromOutputError(self, error, correction_coeff):
        
        #save previous weights before refreshing their value
        self.previous_synaptic_weights = self.synaptic_weights
        
        self.error = np.float64(error)
        
        self.upateParameters(correction_coeff)
           
        return
    
    def upateParameters(self, correction_coeff):
         
        #refresh each weight of the neuron using the gradient descent method
        for i in range (0, len(self.synaptic_weights)+1,1):
                                    
            grad_err_param_i = self.computeErrorFunctionGradient(i,self.error)
            
            # Index = len(self.synaptic_weights) means we are updating the bias
            if(i != len(self.synaptic_weights)):
                #refresh weights
                self.synaptic_weights[i]= self.getParameterNewValue(i,self.synaptic_weights[i], correction_coeff, grad_err_param_i)
            else:
                #refresh bias
                self.bias = self.getParameterNewValue(i,self.bias, correction_coeff, grad_err_param_i)
        
        self.num_steps+=1
           
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
                                                           self.beta1,
                                                           self.beta2,
                                                           self.error_function_gradient.name).__dict__)
        else:
            hyperParams = NeuronHyperParameters(self.synaptic_weights, 
                                                self.bias, 
                                                self.activation_function.__name__, 
                                                self.der_activation_function.__name__,
                                                self.optimizer.name,
                                                self.beta1,
                                                self.beta2,
                                                self.error_function_gradient.name).__dict__
            
        return hyperParams
    
    def loadHyperParameters(self, hyperParamsJson):
        
        hyperParamsReceiverObject = json.loads(hyperParamsJson)
        
        self.synaptic_weights = hyperParamsReceiverObject["synaptic_weights"]
        self.previous_synaptic_weights = self.synaptic_weights
        self.activation_function = Activation_functions.getFunctionByName(hyperParamsReceiverObject["activation_function"])
        self.der_activation_function = Activation_functions.getFunctionByName(hyperParamsReceiverObject["der_activation_function"])
        self.error_function_gradient = ErrorFunctionGradient[hyperParamsReceiverObject["error_function_gradient"]]
        self.bias = hyperParamsReceiverObject["bias"]
        self.optimizer = Optimizer[hyperParamsReceiverObject["optimizer"]]
        self.beta1=hyperParamsReceiverObject["beta1"]
        self.beta2=hyperParamsReceiverObject["beta2"]
        
        self.input_values = np.empty(len(self.synaptic_weights), dtype=float)
       
        self.output_value = 0
            
        self.error = 0
        
        #depending on the chosen optimizer the param table shape will be different
        if(self.optimizer == Optimizer.ADAM):
            self.optimizer_params = np.zeros((2,len(self.synaptic_weights)+1), dtype=float)
        else :
            self.optimizer_params = np.zeros(len(self.synaptic_weights)+1, dtype=float)
        
        self.num_steps = 1
        
        
class NeuronHyperParameters(object):
    
    synaptic_weights = None
    bias = None
    activation_function= None
    der_activation_function = None
    error_function_gradient = None
    
    optimizer = None
    beta1 = None
    beta2 = None
        
    def __init__(self, synaptic_weights, 
                 bias, activation_function, 
                 der_activation_function,
                 optimizer,
                 beta1,
                 beta2,
                 error_function_gradient):
        
        self.synaptic_weights=[]
        for w in synaptic_weights :
            self.synaptic_weights.append(w)
            
        self.bias = bias
        self.optimizer=optimizer
        self.beta1=beta1
        self.beta2=beta2
        self.activation_function = activation_function
        self.der_activation_function = der_activation_function        
        self.error_function_gradient = error_function_gradient
        
class Optimizer(Enum):
    SGD = 0,
    MOMENTUM = 1,
    NAG = 2,
    RMSProp = 3,
    ADAM = 4,
    ADAMW = 5
    
class ErrorFunctionGradient(Enum):
    MEAN_SQUARED_ERROR_LOSS = 0,
    BINARY_CROSS_ENTROPY_LOSS = 1,
    CATEGORICAL_CROSS_ENTROPY_LOSS = 2
