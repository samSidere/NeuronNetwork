'''
Created on 21 août 2024

@author: SSM9
'''
from ArtificialNeuronNetwork.NeuronLayer import NeuronLayer
import ArtificialNeuronNetwork.Cost_functions as Cost_functions
import ArtificialNeuronNetwork.Activation_functions as Activation_functions
from ArtificialNeuronNetwork.Neuron import ErrorFunctionGradient
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
    
    softmax_output=False
    

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
                 output_layer_activation_function=Activation_functions.neuronInhibitionFun, output_layer_der_activation_function=Activation_functions.der_neuronInhibitionFun,
                 softmax_output=False,
                 optimizer=None,
                 beta1=0,
                 beta2=0):
        
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.network_depth = network_depth
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        
        self.correction_coeff = correction_coeff
        self.softmax_output = softmax_output
        
        self.cost_function = cost_function
        
        if(self.cost_function == Cost_functions.mean_squared_error):
            error_function_gradient = ErrorFunctionGradient.MEAN_SQUARED_ERROR_LOSS
        elif (self.cost_function == Cost_functions.binary_cross_entropy):
            error_function_gradient = ErrorFunctionGradient.BINARY_CROSS_ENTROPY_LOSS
        elif (self.cost_function == Cost_functions.categorical_cross_entropy):
            error_function_gradient = ErrorFunctionGradient.CATEGORICAL_CROSS_ENTROPY_LOSS
        else :
            print('warning Error function is not specified at Neuron Level, make sure the Neuron computation mechanism is correct' )
            error_function_gradient = ErrorFunctionGradient.MEAN_SQUARED_ERROR_LOSS
                
        self.input_layer = NeuronLayer(self.number_of_inputs, 1, input_layer_activation_function, input_layer_der_activation_function, 0, True, optimizer, beta1, beta2 , error_function_gradient)
        
        self.hidden_layers=[]
        
        if self.network_depth==0 or self.neurons_per_hidden_layer == 0:
            self.output_layer = NeuronLayer(self.number_of_outputs, self.number_of_inputs, output_layer_activation_function, output_layer_der_activation_function, 1, False, optimizer, beta1, beta2 , error_function_gradient)
        elif self.network_depth > 0 and self.neurons_per_hidden_layer > 0:
            
            for i in range (0,self.network_depth,1):
                if i==0:
                    self.hidden_layers.append(NeuronLayer(self.neurons_per_hidden_layer, self.number_of_inputs, hidden_layers_activation_function, hidden_layer_der_activation_function, 0, False, optimizer, beta1, beta2 , error_function_gradient))
                else:
                    self.hidden_layers.append(NeuronLayer(self.neurons_per_hidden_layer, self.neurons_per_hidden_layer, hidden_layers_activation_function, hidden_layer_der_activation_function, 0, False, optimizer, beta1, beta2 , error_function_gradient))
            
            self.output_layer = NeuronLayer(self.number_of_outputs, self.neurons_per_hidden_layer, output_layer_activation_function, output_layer_der_activation_function, 1, False, optimizer, beta1, beta2 , error_function_gradient)
        else:
            print("wrong value for network depth and/or number of neurons per layer")
        
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
        
        if self.softmax_output :
            output_data = doStableSoftmax(output_data)
        
        return output_data
    
      
    '''
    Calcule les performance du réseau à l'aide de la fonction de coût
    '''
    def computeCostFunctionResult (self, expected_results,computed_result):
        
        #Compute cost function for a batch of result #TODO Check if there is an issue in case we have more than one output for other cost function
        cost_function_results=self.cost_function(expected_results, computed_result)
        print("cost function result from batch simulation ="+str(cost_function_results))        
        return cost_function_results
   
   
   
    '''
    Calcule l'erreur qui sera propagée dans le réseau pour la propagation inverse. Cette erreur correspond à au gradientΘ de la fonction de perte associée au réseau.
    L(Θ)' => Bien entendu ce gradient est celui d'une fonction composée mais les autres facteurs de cette fonctions sont ajoutées lors du calcul de l'erreur dans chaque neurone
    #'''
    
    def computeErrorUsingLossFunction(self, expected_result, computed_result):
                                
        if(self.cost_function == Cost_functions.mean_squared_error):
            return -2*(expected_result-computed_result)
        
        elif (self.cost_function == Cost_functions.binary_cross_entropy):
            return -(expected_result/computed_result)+(1-expected_result)/(1-computed_result)
        
        elif (self.cost_function == Cost_functions.categorical_cross_entropy):
            #This formula is specific to multi class problems where expected result is always a vector of type [0...010...0] (only one selected class)
            return computed_result-expected_result
        
        else:
            return -2*(expected_result-computed_result)
        
        
    '''
    Entraine le modèle
    '''
    def supervisedModelTrainingEpochExecution(self, input_data_set, expected_results):
        
        #Check data consistency
        if len(input_data_set)!= len(expected_results):
            print("Error : input data set("+str(len(input_data_set))+") and expected results ("+str(len(expected_results))+") sizes are different")
            return
        elif self.number_of_outputs!= len(expected_results[0]):
            print("Error : number of output ("+str(self.number_of_outputs)+") and expected results size("+str(len(expected_results[0]))+") are different")
            return        
        
        #init computed results table
        expected_results = np.array(expected_results)
        computed_results = np.zeros((len(input_data_set),self.number_of_outputs))
                
        #Execute model for each data
        for i in range (0,len(input_data_set),1):
            
            errors=np.zeros(self.number_of_outputs)
                        
            self.executeModel(input_data_set[i])
            
            #Store output values into the computed results table for each output neurons            
            for j in range(0, self.number_of_outputs,1):
                computed_results[i][j]=self.output_layer.neurons[j].output_value
                
            if self.softmax_output :
                computed_results[i] = doStableSoftmax(computed_results[i])
                errors=self.computeErrorUsingLossFunction(expected_results[i],computed_results[i])
            else:
                for j in range(0, self.number_of_outputs,1):
                    errors[j]=self.computeErrorUsingLossFunction(expected_results[i][j],computed_results[i][j])
            
            
            #print("errors :"+str(errors))   
            self.updateModelParameters(errors)
            
        #print("expected results are : "+str(expected_results))
        #print("computed results are : "+str(computed_results))
        cost_function_results = self.computeCostFunctionResult(expected_results,computed_results)     
                
        return cost_function_results
    
    '''
    TODO improve this method
    '''
    def TDB_supervisedModelTrainingByBatchEpochExecution(self, input_data_set, expected_results):
        
        #Check data consistency
        if len(input_data_set)!= len(expected_results):
            print("Error : input data set("+str(len(input_data_set))+") and expected results ("+str(len(expected_results))+") sizes are different")
            return
        elif self.number_of_outputs!= len(expected_results[0]):
            print("Error : number of output ("+str(self.number_of_outputs)+") and expected results size("+str(len(expected_results[0]))+") are different")
            return        
        
        #init computed results table
        expected_results = np.array(expected_results)
        computed_results = np.zeros((len(input_data_set),self.number_of_outputs))
        
        epoch_errors = np.zeros((len(input_data_set),self.number_of_outputs))
                        
        #Execute model for each data
        for i in range (0,len(input_data_set),1):
            
            errors=np.zeros(self.number_of_outputs)
                        
            self.executeModel(input_data_set[i])
            
            #Store output values into the computed results table for each output neurons            
            for j in range(0, self.number_of_outputs,1):
                computed_results[i][j]=self.output_layer.neurons[j].output_value
                
            if self.softmax_output :
                computed_results[i] = doStableSoftmax(computed_results[i])
                errors=self.computeErrorUsingLossFunction(expected_results[i],computed_results[i])
            else:
                for j in range(0, self.number_of_outputs,1):
                    errors[j]=self.computeErrorUsingLossFunction(expected_results[i][j],computed_results[i][j])
            
            
            epoch_errors[i] = errors
        
        epoch_errors = np.transpose(epoch_errors)
        
        errors = np.zeros(self.number_of_outputs)
                    
        for i in range(0, self.number_of_outputs, 1):
            errors[i]=np.sum(epoch_errors[i])           
              
        self.updateModelParameters(errors)
            
        #print("expected results are : "+str(expected_results))
        #print("computed results are : "+str(computed_results))
        cost_function_results = self.computeCostFunctionResult(expected_results,computed_results)        
        
        return cost_function_results
        
    
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
                                                            self.correction_coeff,
                                                            self.softmax_output).__dict__)
        else:
            hyperParams = NetworkHyperParameters(self.input_layer, 
                                                            self.hidden_layers, 
                                                            self.output_layer, 
                                                            self.number_of_inputs,
                                                            self.number_of_outputs,
                                                            self.network_depth,
                                                            self.neurons_per_hidden_layer,
                                                            self.cost_function.__name__,
                                                            self.correction_coeff,
                                                            self.softmax_output).__dict__
        
        return hyperParams
    
    def loadHyperParameters(self, hyperParamsJson):
        
        hyperParamsReceiverObject = json.loads(hyperParamsJson)
        
        self.number_of_inputs = hyperParamsReceiverObject["number_of_inputs"]
        self.number_of_outputs = hyperParamsReceiverObject["number_of_outputs"]
        self.network_depth = hyperParamsReceiverObject["network_depth"]
        self.neurons_per_hidden_layer = hyperParamsReceiverObject["neurons_per_hidden_layer"]
        
        self.cost_function = Cost_functions.getFunctionByName(hyperParamsReceiverObject["cost_function"])
        self.correction_coeff = hyperParamsReceiverObject["correction_coeff"]
        self.softmax_output = hyperParamsReceiverObject["softmax_output"]
        
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
    softmax_output=False
    
    def __init__(self, input_layer, 
                 hidden_layers, 
                 output_layer, 
                 number_of_inputs,
                 number_of_outputs,
                 network_depth,
                 neurons_per_hidden_layer,
                 cost_function,
                 correction_coeff,
                 softmax_output):
        
        self.hidden_layers=[]
        
    
        self.number_of_inputs=number_of_inputs
        self.number_of_outputs=number_of_outputs
        self.network_depth=network_depth
        self.neurons_per_hidden_layer=neurons_per_hidden_layer
        
        self.cost_function=cost_function
        self.correction_coeff=correction_coeff
        self.softmax_output=softmax_output
        
        
        self.input_layer=input_layer.getHyperParameters(directCall=False)
        self.output_layer=output_layer.getHyperParameters(directCall=False)
        for layer in hidden_layers :
            self.hidden_layers.append(layer.getHyperParameters(directCall=False))
            
'''
Softmax Function
Before exploring the ins and outs of the Softmax activation function, we should focus on its building block—the sigmoid/logistic activation function that works on calculating probability values. 

The output of the sigmoid function was in the range of 0 to 1, which can be thought of as probability. 

But—

This function faces certain problems.

Let’s suppose we have five output values of 0.8, 0.9, 0.7, 0.8, and 0.6, respectively. How can we move forward with it?

The answer is: We can’t.

The above values don’t make sense as the sum of all the classes/output probabilities should be equal to 1. 

You see, the Softmax function is described as a combination of multiple sigmoids. 

It calculates the relative probabilities. Similar to the sigmoid/logistic activation function, the SoftMax function returns the probability of each class. 

It is most commonly used as an activation function for the last layer of the neural network in the case of multi-class classification. 

Let’s go over a simple example together.

Assume that you have three classes, meaning that there would be three neurons in the output layer. Now, suppose that your output from the neurons is [1.8, 0.9, 0.68].

Applying the softmax function over these values to give a probabilistic view will result in the following outcome: [0.58, 0.23, 0.19]. 

The function returns 1 for the largest probability index while it returns 0 for the other two array indexes. Here, giving full weight to index 0 and no weight to index 1 and index 2. So the output would be the class corresponding to the 1st neuron(index 0) out of three.

You can see now how softmax activation function make things easy for multi-class classification problems.

'''
def doSoftmax(X):
    '''
    Faire la somme totale des proba
    Générer le vecteur définissant la proba de chaque entrée
    #TODO implement stable version
    '''
    return np.exp(X)/np.sum(np.exp(X))

def doStableSoftmax(X):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftx = X - np.max(X)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)