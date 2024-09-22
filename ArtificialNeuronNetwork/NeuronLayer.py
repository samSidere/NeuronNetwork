'''
Created on 20 août 2024

@author: SSM9
'''


from ArtificialNeuronNetwork.Neuron import Neuron
import ArtificialNeuronNetwork.Activation_functions as Activation_functions
import numpy as np
import json

class NeuronLayer(object):
       
    layerSize = None
    dendritePerNeuron = None
    
    neurons = None
    isInputLayer = None
    
    def __init__(self, layerSize=0, dendritePerNeuron=0, activationFunction=Activation_functions.neuronInhibitionFun, der_activationFunction=Activation_functions.der_neuronInhibitionFun, neurons_bias=0, isInputLayer=False):
        self.layerSize = layerSize
        self.dendritePerNeuron = dendritePerNeuron
        self.neurons=[]
        self.isInputLayer = isInputLayer
        
        for i in range (0,self.layerSize,1):
            if self.isInputLayer:
                #Input layer weight are always ones and only one input per neuron is allowed
                self.neurons.append(Neuron(np.ones(1),activationFunction,der_activationFunction,neurons_bias))
            else:
                self.neurons.append(Neuron(np.random.randn(self.dendritePerNeuron),activationFunction,der_activationFunction,neurons_bias))
        
    
    def connectLayerToInputData(self,input_data):
        #Process input Layer
        #The input layer takes raw input from the domain. No computation (except formatting) is performed at this layer. Nodes here just pass on the information (features) to the hidden layer. 
        for i in range (0,self.layerSize,1):
            #refresh inputs
            self.neurons[i].input_values = [input_data[i]]
            
        #Do feedforward propagation for current Layer
        self.feedForwardPropagationThroughLayer()
        return
    
    
    def connectLayerToPreviousLayer(self, previousLayer):
        
        #Do feedforward propagation for current Layer
        
        for i in range (0,self.layerSize,1):
            #refresh input by defining synaptic connection between previous layer and current layer
            for j in range (0, previousLayer.layerSize,1):
                self.neurons[i].input_values[j] = previousLayer.neurons[j].output_value        
        return 
    
    def feedForwardPropagationThroughLayer(self):
        
        #Do feedforward propagation for current Layer
        for i in range (0,self.layerSize,1):
            #process
            self.neurons[i].processInputs()
        
        return 
    
    #TODO : refactor all this capability and infrastructure
    def backPropagationThroughLayer(self, nextLayer, correction_coeff):
        
        if(self.isInputLayer):
            return
        
        next_layer_errors=[]
        next_layer_previous_weights_associated_to_each_current_layer_neuron= []
        
        #get next layer errors and weights associated to each neuron
        for i in range (0, nextLayer.layerSize, 1):
            next_layer_errors.append(nextLayer.neurons[i].error)
            
        
        #get for each neuron of the current layer a table of weights of the next layer associated to its synaptic connection
        for i in range (0, self.layerSize,1):
            
            #init the temporary weights table
            next_layer_previous_weights_associated_to_neuron_i = []
            
            #Parse all of the current neurons synaptic connections to get the associated weights
            for j in range (0, nextLayer.layerSize, 1):
                next_layer_previous_weights_associated_to_neuron_i.append(nextLayer.neurons[j].previous_synaptic_weights[i])
        
            next_layer_previous_weights_associated_to_each_current_layer_neuron.append(next_layer_previous_weights_associated_to_neuron_i)
            
        
        #Do back propagation for current Layer
        for i in range (0,self.layerSize,1):            
            #update weights and store errors into each neuron of the layer
            self.neurons[i].updateParametersFromNextLayer(next_layer_errors, correction_coeff, next_layer_previous_weights_associated_to_each_current_layer_neuron[i])
        return 
      
    def backPropagationThroughOuptputLayer(self, errors, correction_coeff, outputIndex = None):
        
        if(self.isInputLayer):
            return
        
        
        
        if(outputIndex == None):
            #Do back propagation for current Layer
            for i in range (0,self.layerSize,1):
                #TODO build weights table for "previous" layer before their update
                #update weights and store errors into each neuron of the layer
                self.neurons[i].updateParametersFromOutputError( errors[i], correction_coeff)
                return 
        else :
            for neuron in self.neurons:
                neuron.error=0
            self.neurons[outputIndex].updateParametersFromOutputError( errors, correction_coeff)
            return 
            
    def printLayerOutput(self):
        
        for neuron in self.neurons:
            print(str(neuron.output_value))
        
        return
    
    
    def getHyperParameters(self, directCall=True):
        
        if(directCall==True):
            hyperParams = json.dumps(LayerHyperParameters(self.layerSize, self.dendritePerNeuron, self.neurons, self.isInputLayer).__dict__)
        else:
            hyperParams = LayerHyperParameters(self.layerSize, self.dendritePerNeuron, self.neurons, self.isInputLayer).__dict__
        
        return hyperParams
    
    def loadHyperParameters(self, hyperParamsJson):
        
        hyperParamsReceiverObject = json.loads(hyperParamsJson)
                
        self.layerSize = hyperParamsReceiverObject["layerSize"]
        self.dendritePerNeuron = hyperParamsReceiverObject["dendritePerNeuron"]
        self.isInputLayer = hyperParamsReceiverObject["isInputLayer"]
        
        for i in range (0,self.layerSize,1):
            self.neurons.append(Neuron())
            self.neurons[i].loadHyperParameters(json.dumps(hyperParamsReceiverObject["neurons"][i]))
        
    
class LayerHyperParameters(object):
    
    layerSize = None
    dendritePerNeuron = None
    
    neurons = None
    isInputLayer = None
    
    def __init__(self, layerSize, dendritePerNeuron, neurons, isInputLayer):
        self.layerSize = layerSize
        self.dendritePerNeuron = dendritePerNeuron
        self.neurons = []
        self.isInputLayer = isInputLayer
        
        for neuron in neurons :
            self.neurons.append(neuron.getHyperParameters(directCall=False))
           