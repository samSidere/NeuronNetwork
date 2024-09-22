'''
Created on 21 août 2024

@author: SSM9
'''
from ArtificialNeuronNetwork.NeuronLayer import NeuronLayer
from ArtificialNeuronNetwork.Neuron import Neuron
import ArtificialNeuronNetwork.Cost_functions as Cost_functions
import ArtificialNeuronNetwork.Activation_functions as Activation_functions
import numpy as np
import json

class NeuronNetwork(object):
    
    '''
    input Layer = couche d'entrée
    hidden Layers = tableau de couches de neurones cachées
    output Layer = couche de sortie
    '''
    input_layer=None
    hidden_layers=None
    output_layer=None
    
    number_of_inputs=None
    number_of_outputs=None
    network_depth=None
    neurons_per_hidden_layer=None
        
    cost_function=None
    correction_coeff=None
    

    '''
    Initialise le réseau de neurone utilisé comme ML model
    '''
    def __init__(self, 
                 number_of_inputs=0, 
                 number_of_outputs=0, 
                 network_depth=0,
                 neurons_per_hidden_layer=0,
                 correction_coeff=1, 
                 cost_function=Cost_functions.mean_squared_error, 
                 input_layer_activation_function=Activation_functions.neuronInhibitionFun, input_layer_der_activation_function=Activation_functions.der_neuronInhibitionFun,
                 hidden_layers_activation_function=Activation_functions.neuronInhibitionFun, hidden_layer_der_activation_function=Activation_functions.der_neuronInhibitionFun,
                 output_layer_activation_function=Activation_functions.neuronInhibitionFun, output_layer_der_activation_function=Activation_functions.der_neuronInhibitionFun ):
        
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.network_depth = network_depth
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        
        
        
        self.input_layer = NeuronLayer(self.number_of_inputs, 1, input_layer_activation_function, input_layer_der_activation_function, 0, True)
        
        self.hidden_layers=[]
        
        if self.network_depth==0 or self.neurons_per_hidden_layer == 0:
            self.output_layer = NeuronLayer(self.number_of_outputs, self.number_of_inputs, output_layer_activation_function, output_layer_der_activation_function, 1, False)
        elif self.network_depth > 0 and self.neurons_per_hidden_layer > 0:
            
            for i in range (0,self.network_depth,1):
                if i==0:
                    self.hidden_layers.append(NeuronLayer(self.neurons_per_hidden_layer, self.number_of_inputs, hidden_layers_activation_function, hidden_layer_der_activation_function, 0, False))
                else:
                    self.hidden_layers.append(NeuronLayer(self.neurons_per_hidden_layer, self.neurons_per_hidden_layer, hidden_layers_activation_function, hidden_layer_der_activation_function, 0, False))
            
            self.output_layer = NeuronLayer(self.number_of_outputs, self.neurons_per_hidden_layer, output_layer_activation_function, output_layer_der_activation_function, 1, False)
        else:
            print("wrong value for network depth and/or number of neurons per layer")
         
        self.cost_function = cost_function
        self.correction_coeff = correction_coeff
        
    '''
    Execute sur une un jeu d'entrées le modèle courant
    '''    
    def executeModel(self, input_data):
        
        #Connect input data to input Layer
        self.input_layer.connectLayerToInputData(input_data)
                
        #Perform feed forward propagation
        self.feedForwardPropagation()
        
        #return model result (output layer neurons)               
        return self.output_layer.neurons
    
    '''
    Execute sur une un jeu d'entrées le modèle courant
    ''' 
    def feedForwardPropagation(self):
        ###############################################################################################################################################
        #
        #
        #                                            FEED FORWARD
        # Feedforward Propagation - the flow of information occurs in the forward direction. The input is used to calculate some intermediate function in the hidden layer, which is then used to calculate an output. 
        ###############################################################################################################################################
        if self.network_depth==0 or self.neurons_per_hidden_layer == 0:
            #in the case there is no hidden layer
            self.output_layer.connectLayerToPreviousLayer(self.input_layer)
            self.output_layer.feedForwardPropagationThroughLayer()
            return
        else :
            #parse hidden layer and perform computation
            for i in range (0,self.network_depth,1):
                if i==0:
                    self.hidden_layers[i].connectLayerToPreviousLayer(self.input_layer)
                else:
                    self.hidden_layers[i].connectLayerToPreviousLayer(self.hidden_layers[i-1])
                
                self.hidden_layers[i].feedForwardPropagationThroughLayer()
            
            self.output_layer.connectLayerToPreviousLayer(self.hidden_layers[self.network_depth-1])
            self.output_layer.feedForwardPropagationThroughLayer()
        return
    
    '''
    Renvoie la sortie du réseau dans un tenseur
    '''
    def getNetworkOutput(self):
        
        if self.number_of_outputs > 1 :
            output_data = []
        
            for neuron in self.output_layer.neurons:
                output_data.append(neuron.output_value)
                
        else :
            output_data = self.output_layer.neurons[0].output_value
            
        return output_data
    
      
    '''
    Calcule les performance du réseau
    '''
    def computeNetworkPerformance (self, computed_result, expected_results):
        
        #Compute cost function for a batch of result #TODO Check if there is an issue in case we have more than one output for other cost function
        cost_function_results=self.cost_function(expected_results, computed_result)
        print("cost function result from batch simulation ="+str(cost_function_results))        
        return cost_function_results
   
    '''
    Entraine le modèle
    '''
    def supervisedModelTrainingEpochExecution(self, input_data_set, expected_results):
        
        '''
        print("input layer w")
        for neuron in self.input_layer.neurons :
            print(str(neuron.synaptic_weights))
        print("hidden layers w")
        for layer in self.hidden_layers :
            for neuron in layer.neurons :
                print(str(neuron.synaptic_weights))
        print("output layer w")
        for neuron in self.output_layer.neurons :
            print(str(neuron.synaptic_weights))
        #'''
        
        #Check data consistency
        if len(input_data_set)!= len(expected_results[0]):
            print("Error : input data set("+str(len(input_data_set))+") and expected results ("+str(len(expected_results[0]))+") sizes are different")
            return
        elif self.number_of_outputs!= len(expected_results):
            print("Error : number of output ("+str(self.number_of_outputs)+") and expected results size("+str(len(expected_results))+") are different")
            return        
        
        #init computed results table
        computed_results = np.zeros((self.number_of_outputs,len(input_data_set)))
                
        #Execute model for each data
        for i in range (0,len(input_data_set),1):
            
            errors=np.zeros(self.number_of_outputs)
                        
            self.executeModel(input_data_set[i])
            
            #Store output values into the computed results table for each output neurons            
            for j in range(0, self.number_of_outputs,1):
                computed_results[j][i]=self.output_layer.neurons[j].output_value
                
            for j in range(0, self.number_of_outputs,1):
                errors[j]=expected_results[j][i]-computed_results[j][i]
            
            #print("errors :"+str(errors))   
            self.updateModelParameters(errors)
            
            
        '''
        print("input layer w")
        for neuron in self.input_layer.neurons :
            print(str(neuron.synaptic_weights))
        print("hidden layers w")
        for layer in self.hidden_layers :
            for neuron in layer.neurons :
                print(str(neuron.synaptic_weights))
        print("output layer w")
        for neuron in self.output_layer.neurons :
            print(str(neuron.synaptic_weights))
        #'''
        print("expected results are : "+str(expected_results))
        print("computed results are : "+str(computed_results))
        
        cost_function_results = self.computeNetworkPerformance(computed_results, expected_results)
        
        '''
        #Update model parameters with back propagation and gradient descent algorithm after each epoch for each output
        for i in range (0, len(input_data_set), 1) :
            errors=np.zeros(self.number_of_outputs)
            
            for j in range(0, self.number_of_outputs,1):
                errors[j]=expected_results[j][i]-computed_results[j][i]
            
            #print("errors :"+str(errors))   
            self.updateModelParameters(errors)
        '''
        
        return
    
    '''
    Met à jour jour les paramètres du réseau en propageant les erreurs vers la couche d'entrée
    '''    
    def updateModelParameters(self, errors, outputIndex = None):
        
        #Connect errors to ouput Layer
        self.output_layer.backPropagationThroughOuptputLayer(errors, self.correction_coeff, outputIndex)
                
        #print("tbd")
        
        #Perform back propagation
        self.backPropagation()
        
        #print("tbd")
        
        #print("tbd")
        
        return
    
    '''
    Execute sur une un jeu d'entrées le modèle courant
    ''' 
    def backPropagation(self):
        ###############################################################################################################################################
        #
        #
        #                                            BACK PROPAGATION (ADJUST WEIGHTS)
        #Backpropagation - the weights of the network connections are repeatedly adjusted to minimize the difference between the actual output vector of the net and the desired output vector.
        #To put it simply—backpropagation aims to minimize the cost function by adjusting the network’s weights and biases. The cost function gradients determine the level of adjustment with respect to parameters like activation function, weights, bias, etc.
        ###############################################################################################################################################
        
        if self.network_depth==0 or self.neurons_per_hidden_layer == 0:
            #in the case there is no hidden layer input parameters are not updated
            #self.input_layer.backPropagationThroughLayer(self.output_layer, self.correction_coeff)
            return
        else :
            #parse hidden layer backward and perform computation
            for i in range (0,self.network_depth,1):
                if i==0:
                    self.hidden_layers[self.network_depth-1-i].backPropagationThroughLayer(self.output_layer, self.correction_coeff)
                else:
                    self.hidden_layers[self.network_depth-1-i].backPropagationThroughLayer(self.hidden_layers[self.network_depth-i], self.correction_coeff)
            
            #input parameters are not updated
            #self.input_layer.backPropagationThroughLayer(self.hidden_layers[0], self.correction_coeff)
        return
    
    def saveNetworkParameterIntofile(self, filename="HyperParameters"):
        hyperParams = self.getHyperParameters(True)
        
        file = open(filename,"w")
        file.write(hyperParams)
        file.close()
    
    def loadNetworkParameterFromfile(self, filename="HyperParameters"):
        file = open(filename,"r")
        self.loadHyperParameters(file.read())
        file.close()
    
    
    def getHyperParameters(self, directCall=True):
        
        if(directCall==True):
            hyperParams = json.dumps(NetworkHyperParameters(self.input_layer, 
                                                            self.hidden_layers, 
                                                            self.output_layer, 
                                                            self.number_of_inputs,
                                                            self.number_of_outputs,
                                                            self.network_depth,
                                                            self.neurons_per_hidden_layer,
                                                            self.cost_function.__name__,
                                                            self.correction_coeff).__dict__)
        else:
            hyperParams = NetworkHyperParameters(self.input_layer, 
                                                            self.hidden_layers, 
                                                            self.output_layer, 
                                                            self.number_of_inputs,
                                                            self.number_of_outputs,
                                                            self.network_depth,
                                                            self.neurons_per_hidden_layer,
                                                            self.cost_function.__name__,
                                                            self.correction_coeff).__dict__
        
        return hyperParams
    
    def loadHyperParameters(self, hyperParamsJson):
        
        hyperParamsReceiverObject = json.loads(hyperParamsJson)
        
        self.number_of_inputs = hyperParamsReceiverObject["number_of_inputs"]
        self.number_of_outputs = hyperParamsReceiverObject["number_of_outputs"]
        self.network_depth = hyperParamsReceiverObject["network_depth"]
        self.neurons_per_hidden_layer = hyperParamsReceiverObject["neurons_per_hidden_layer"]
        
        self.cost_function = Cost_functions.getFunctionByName(hyperParamsReceiverObject["cost_function"])
        self.correction_coeff = hyperParamsReceiverObject["correction_coeff"]
        
        self.input_layer = NeuronLayer()
        self.input_layer.loadHyperParameters(json.dumps(hyperParamsReceiverObject["input_layer"]))
        
        self.output_layer = NeuronLayer()
        self.output_layer.loadHyperParameters(json.dumps(hyperParamsReceiverObject["output_layer"]))
        
        if self.network_depth > 0 and self.neurons_per_hidden_layer > 0:
            for i in range (0,self.network_depth,1):
                self.hidden_layers.append(NeuronLayer())
                self.hidden_layers[i].loadHyperParameters(json.dumps(hyperParamsReceiverObject["hidden_layers"][i]))
               
        
    
class NetworkHyperParameters(object):
    
    input_layer=[]
    hidden_layers=[]
    output_layer=[]
    
    number_of_inputs=None
    number_of_outputs=None
    network_depth=None
    neurons_per_hidden_layer=None
        
    cost_function=None
    correction_coeff=None
    
    def __init__(self, input_layer, hidden_layers, output_layer, number_of_inputs,number_of_outputs,network_depth,neurons_per_hidden_layer,cost_function,correction_coeff):
        
        self.hidden_layers=[]
        
    
        self.number_of_inputs=number_of_inputs
        self.number_of_outputs=number_of_outputs
        self.network_depth=network_depth
        self.neurons_per_hidden_layer=neurons_per_hidden_layer
        
        self.cost_function=cost_function
        self.correction_coeff=correction_coeff
        
        
        self.input_layer=input_layer.getHyperParameters(directCall=False)
        self.output_layer=output_layer.getHyperParameters(directCall=False)
        for layer in hidden_layers :
            self.hidden_layers.append(layer.getHyperParameters(directCall=False))
            