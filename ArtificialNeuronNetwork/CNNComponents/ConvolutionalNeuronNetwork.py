'''
Created on 19 nov. 2024

@author: SSM9
'''

from ArtificialNeuronNetwork.CNNComponents.ConvolutionLayer import ConvolutionLayer
from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork

from ArtificialNeuronNetwork import Activation_functions
from ArtificialNeuronNetwork import Cost_functions

from ArtificialNeuronNetwork.Neuron import Optimizer

import numpy as np

class ConvolutionalNeuronNetwork(object):
    '''
    the convolution network will connect convolution layers to a fully connected Network
    '''
    convolutionLayers = None
    fullyConnectedNetwork = None
    
    inputsShape = None
    
    number_of_classes = None
    
    correction_coeff = None
    

    def __init__(self, inputShape = (50,50,1)):
        
        kernelShape = [2,3,1]
        
        numberOfKernels = 1
        
        #Build the parameters from shape
        #Check if side len is odd
        if(kernelShape[1]%2==0):
            print("kernel side len not valid and has been increased by 1")
            kernelShape[1]+=1
            
            
        paddingLen = int(kernelShape[1]/2)
        
        self.inputsShape = inputShape
        self.number_of_classes= 2
                
        featureMapsShape = (self.inputsShape[0], self.inputsShape[1])
        
        self.correction_coeff = 1e-2
        
        self.convolutionLayers = ConvolutionLayer(layerSize=numberOfKernels, kernelsShape = kernelShape,
                                                  activation_function = Activation_functions.reLUFun,
                                                  der_activation_fun = Activation_functions.der_reLUFun,
                                                  optimizer = Optimizer.ADAM, 
                                                  beta1 = 0.9, 
                                                  beta2 = 0.999,
                                                  maxPooling = False,
                                                  maxPoolingShape = [1,2]
                                                  )
        
        self.fullyConnectedNetwork = NeuronNetwork(number_of_inputs = featureMapsShape[0]*featureMapsShape[1]*numberOfKernels, 
                                                   number_of_outputs = self.number_of_classes, 
                                                   network_depth = 8, 
                                                   neurons_per_hidden_layer = 8, 
                                                   correction_coeff = self.correction_coeff, 
                                                   cost_function = Cost_functions.categorical_cross_entropy, 
                                                   input_layer_activation_function = Activation_functions.linearActivationFun, 
                                                   input_layer_der_activation_function = Activation_functions.der_linearActivationFun, 
                                                   hidden_layers_activation_function = Activation_functions.reLUFun, 
                                                   hidden_layer_der_activation_function = Activation_functions.der_reLUFun, 
                                                   output_layer_activation_function = Activation_functions.linearActivationFun, 
                                                   output_layer_der_activation_function = Activation_functions.der_linearActivationFun, 
                                                   softmax_output = True, 
                                                   optimizer = Optimizer.ADAM, 
                                                   beta1 = 0.9, 
                                                   beta2 = 0.999)

        
        
    def executeModel(self, inputData):
        
        #On va faire attention aux données d'entrée / Est qu'on crop les images nous même? => mise en forme considérée acquise
        
        if(inputData.shape != self.inputsShape):
            print("input data shape is not compatible with the network configuration")
            return
        
        self.convolutionLayers.processInputs(inputData)
        
        convolutionResult = self.convolutionLayers.flattenConvLayerOutput()
        
        self.fullyConnectedNetwork.executeModel(convolutionResult)     
         
        return self.fullyConnectedNetwork.getNetworkOutput()
    
    
    def getNetworkOutput(self):
        return self.fullyConnectedNetwork.getNetworkOutput()
    
    
    def supervisedModelTrainingEpochExecution(self, input_data_set, expected_results):
        
        #Check data consistency
        if len(input_data_set)!= len(expected_results):
            print("Error : input data set("+str(len(input_data_set))+") and expected results ("+str(len(expected_results))+") sizes are different")
            return
        elif self.number_of_classes!= len(expected_results[0]):
            print("Error : number of classes ("+str(self.number_of_classes)+") and expected results size("+str(len(expected_results[0]))+") are different")
            return  
        
        #init computed results table
        expected_results = np.array(expected_results)
        computed_results = np.zeros((len(input_data_set),self.number_of_classes))
        
        #pour chaque élément du jeu de test
        #feedforward au travers des couches de convolution
        #feedforward au tavers de la FC
        #Back propagation dans la FC
        #récupération de la première couche (dernière de la backpropagation) de la FC
        #Transmission de la couche à la dernière convolution layer
        #back propagation
        #etc
        
        for i in range (0, len(input_data_set), 1):
                        
            self.convolutionLayers.processInputs(input_data_set[i])
            
            convolutionResult = self.convolutionLayers.flattenConvLayerOutput()
            
            computed_results[i] = self.fullyConnectedNetwork.trainModelOnOneSample(convolutionResult, expected_results[i])
            
            #performance += self.fullyConnectedNetwork.supervisedModelTrainingEpochExecution(convolutionResult, expected_results[i])
            
            #get layer to transmit to convolution layers
            if(self.fullyConnectedNetwork.network_depth == 0 or self.fullyConnectedNetwork.neurons_per_hidden_layer == 0):
                FC_last_layer = self.fullyConnectedNetwork.output_layer
            else:
                FC_last_layer = self.fullyConnectedNetwork.hidden_layers[0]
                        
            self.convolutionLayers.backPropagationThroughLayer(FC_last_layer, self.correction_coeff)
            
        cost_function_results = self.fullyConnectedNetwork.computeCostFunctionResult(expected_results,computed_results)
                
        return cost_function_results