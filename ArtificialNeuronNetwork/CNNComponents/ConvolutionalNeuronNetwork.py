'''
Created on 19 nov. 2024

@author: SSM9
'''

from ArtificialNeuronNetwork.CNNComponents.ConvolutionLayer import ConvolutionLayer
from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork

from ArtificialNeuronNetwork import Activation_functions
from ArtificialNeuronNetwork import Cost_functions

from ArtificialNeuronNetwork.Neuron import Optimizer

class ConvolutionalNeuronNetwork(object):
    '''
    the convolution network will connect convolution layers to a fully connected Network
    '''
    convolutionLayers = None
    fullyConnectedNetwork = None
    
    inputsShape = None
    

    def __init__(self):
        
        kernelShape = [1,3,1]
        
        numberOfKernels = 1
        
        #Build the parameters from shape
        #Check if side len is odd
        if(kernelShape[1]%2==0):
            print("kernel side len not valid and has been increased by 1")
            kernelShape[1]+=1
            
            
        paddingLen = int(self.kernelSideLength/2)
        
        self.inputsShape = (50,50)
                
        featureMapsShape = (self.inputsShape[0]+ paddingLen, self.inputsShape[1]+ paddingLen)
        
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
                                                   number_of_outputs = 2, 
                                                   network_depth = 8, 
                                                   neurons_per_hidden_layer = 8, 
                                                   correction_coeff = 1e-2, 
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
        
        performance = 0
        
        #pour chaque élément du jeu de test
        #feedforward au travers des couches de convolution
        #feedforward au tavers de la FC
        #Back propagation dans la FC
        #récupération de la première couche (dernière de la backpropagation) de la FC
        #Transmission de la couche à la dernière convolution layer
        #back propagation
        #etc
        
        
        
        return performance