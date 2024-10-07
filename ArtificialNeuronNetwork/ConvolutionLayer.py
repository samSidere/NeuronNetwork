'''
Created on 1 oct. 2024

@author: SSM9
'''

from ArtificialNeuronNetwork.NeuronLayer import NeuronLayer

class ConvolutionLayer(NeuronLayer):
    '''
    The convolution Layer will apply a filter to the input in order to detect features that will allow a fully connected layer to attach the input to a class
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        #super().__init__(layerSize, dendritePerNeuron, activationFunction, der_activationFunction, neurons_bias, isInputLayer)
        