'''  
Created on 3 sept. 2024

@author: SSM9
'''

import numpy as np
import copy as cp
import random

from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork
import ArtificialNeuronNetwork.Activation_functions as Activation_functions
import  ArtificialNeuronNetwork.Cost_functions as Cost_functions


class StochasticPolicyGradientLearningAgent(object):
        
    stateSize = None
    actionSetSize = None
    gammaDiscount = None
    
    agentPolicyNetwork = None
    
    
    def __init__(self, stateSize = 1 , actionSetSize = 1, gammaDiscount=0, filename=None ):
        
        self.stateSize = stateSize
        self.actionSetSize = actionSetSize
        self.gammaDiscount = gammaDiscount
        
        
        #Policy Network is the neural network at the center of the agent. It will compute stochastic policy P(a|st) in order to drive agent decisions
        #In this environment, it consists in one neuron network in charge of computing each P(a|st) output
        if filename==None :
            self.agentPolicyNetwork = NeuronNetwork(self.stateSize, self.actionSetSize, 2, 12, 0.01,
                                             Cost_functions.categorical_cross_entropy, 
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                             Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
        else :
            self.agentPolicyNetwork = NeuronNetwork()
            self.agentPolicyNetwork.loadNetworkParameterFromfile(filename)
            self.stateSize = self.agentPolicyNetwork.number_of_inputs
            self.actionSetSize = self.agentPolicyNetwork.number_of_outputs
        
    
    
    #This function is a vanilla feedforward propagation to compute Stochastic policies of each sampled action
    def computeStochasticPolicy(self, state):
        #Connect state to agent Stochastic Policy Network input
        self.agentPolicyNetwork.executeModel(state)
        
        #Use softmax to return probability distribution as output
        networkOutput = Cost_functions.doSoftmax(self.agentPolicyNetwork.getNetworkOutput())
        
        return networkOutput
        
    
    #This function will select an action based on the highest policy value computed by the agent's policy network            
    def sampleActionBasedOnStochasticPolicy(self, state):
        
        sampledPolicyValues = self.computeStochasticPolicy(state)
        
        return np.argmax(sampledPolicyValues)
    
    
    #This function let the user train the Policy Network using a Monte Carlo Reinforce stochastic policy gradient algorithm on one trajectory
    def updatePolicyNetworkParametersOneStepMonteCarloReinforceLearning(self, sampledPolicy, episodeComputedPolicies, episodeReturn):
        
        '''
        #Compute gradient of objective function J(Θ)
        #update NetworkParametersBased J(Θ) Gradient Ascend Method
        '''
        
        
        self.agentPolicyNetwork.updateModelParameters(#TBD)
        
        
        return
    
                
    def saveAgentParamtoFile(self, filename):
        
        self.agentPolicyNetwork.saveNetworkParameterIntofile(filename)
        
        return