'''
Created on 16 août 2024

@author: SSM9
'''
import numpy as np

from ArtificialNeuronNetwork import Activation_functions

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
    
    def __init__(self, synaptic_weights, activation_function, der_activation_function, bias):
        
        self.synaptic_weights = synaptic_weights
        self.previous_synaptic_weights = synaptic_weights
        self.activation_function = activation_function
        self.der_activation_function = der_activation_function
        self.bias = bias
        
        self.input_values = np.empty(len(self.synaptic_weights), dtype=float)
        if self.activation_function ==  Activation_functions.softmax :
            self.output_value = []
        else :
            self.output_value = 0
            
        self.error = 0
    
    def processInputs(self):
        if self.activation_function ==  Activation_functions.softmax :
            self.doSoftmax()
        else :
            self.doRegularProcessing()

            
    
    def doRegularProcessing(self):
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
            
    
    #softmax allow to generate a probability associated to each input of the neuron
    def doSoftmax(self):
        '''
        Faire la somme totale des proba
        Générer le vecteur définissant la proba de chaque entrée
        '''
        divider = 0
        
        for input_value in self.input_values :
            divider = divider+np.exp(input_value)
                        
        for input_value in self.input_values :
            self.output_value.append(np.exp(input_value)/divider)
        
            
    #Compute error from Next Layer TODO : refactor all this capability
    def computeErrorFromNextLayer(self, next_layer_weights_associated_to_self,next_layer_errors):
        self.error = np.float64(0)
        
        self.error = np.dot(next_layer_weights_associated_to_self,next_layer_errors)
        
        return 
        
    #Update weights and bias from error computed from next layer TODO : refactor all this TODO : refactor all this capability and infrastructure
    def updateParametersFromNextLayer(self, next_layer_errors, correction_coeff, next_layer_weights_associated_to_self):
        
        #save previous weights before refreshing their value
        self.previous_synaptic_weights = self.synaptic_weights
        
        self.computeErrorFromNextLayer(next_layer_weights_associated_to_self, next_layer_errors)
                
        #refresh each weight of the neuron using the gradient method
        for i in range (0, len(self.synaptic_weights),1):
            
            grad_err_weight_i = self.error*(-self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)+self.bias))*self.input_values[i]
            
            self.synaptic_weights[i]=self.synaptic_weights[i]+(-correction_coeff*grad_err_weight_i)
        
        #refresh bias
        grad_err_weight_bias = self.error*(-self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)+self.bias))
        self.bias = self.bias+(-correction_coeff*grad_err_weight_bias)
           
        return
    
    #Update weights and bias from error computed from output error (only for output layer neurons) TODO : refactor all this capability
    def updateParametersFromOutputError(self, error, correction_coeff):
        
        #save previous weights before refreshing their value
        self.previous_synaptic_weights = self.synaptic_weights
        
        self.error = np.float64(error)
                        
        #refresh each weight of the neuron using the gradient method
        for i in range (0, len(self.synaptic_weights),1):
                        
            grad_err_weight_i = self.error*(-self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)+self.bias))*self.input_values[i]
            
            self.synaptic_weights[i]=self.synaptic_weights[i]+(-correction_coeff*grad_err_weight_i)
        
        #refresh bias
        grad_err_weight_bias = self.error*(-self.der_activation_function(np.dot(self.synaptic_weights,self.input_values)+self.bias))
        self.bias = self.bias+(-correction_coeff*grad_err_weight_bias)
           
        return
    
    def verbose(self):
        print("I am a neuron with the following parameters \n synaptic weights="+str(self.synaptic_weights)
              +"\n bias="+str(self.bias))

    