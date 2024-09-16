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


class ParallelReinforcementDeepLearningAgent(object):
    '''
        Reinforcement Deep Learning Based agent is a Agent that will learn its parameters Action-Value Functions using ANN
        
        Our goal in deep Q-learning is to solve the action-value function Q(s,a). 
        Why? If the AI agent knows Q(s,a) then the agent will consider the given objective (like winning a chess game versus a human player 
        or playing Atari’s Breakout) solved because the knowledge of Q(s,a) enables the agent to determine the quality of any possible action in any given state. 
        With that knowledge, the agent could behave accordingly and in perpetuity.
        
    '''
    
    stateSize = None
    actionSetSize = None
    gammaDiscount = None
    
    agentQNetwork = []
    agentTargetNetwork = []
    
    agentOutput = []
    

    def __init__(self, stateSize = 1 , actionSetSize = 1, gammaDiscount=0 ):
        
        self.stateSize = stateSize
        self.actionSetSize = actionSetSize
        self.gammaDiscount = gammaDiscount
        
        
        #Q network is the neural network at the center of the agent. It will compute action value functions in order to drive agent decisions
        #In this environment, it consists in N parallel neuron network in charge of computing each Q(s,a) action pair output
        for i in range (0,self.actionSetSize,1):
            self.agentQNetwork.append(NeuronNetwork(self.stateSize, 1, 3, 12, 0.0005,
                                             Cost_functions.mean_squared_error, 
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                             Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun))
            
            #Target network is in charge of computation of TD target that the network will use for training. It uses a copy of Q Net parameters and is updated every N epochs
            self.agentTargetNetwork.append(cp.deepcopy(self.agentQNetwork[i]))
        
        
        self.agentOutput = np.zeros(self.actionSetSize)
    
    def copyParamToAgentTargetNetwork(self) : 
        self.agentTargetNework = cp.deepcopy(self.agentQNetwork)        
        
    def computeActionValueFunctions(self, state):
        #Connect state to agent Q network input
        for i in range (0,self.actionSetSize,1) :
            self.agentQNetwork[i].executeModel(state)
            self.agentOutput[i] = self.agentQNetwork[i].getNetworkOutput()
                
    def getSelectedActionFollowingGreedyPolicy(self):
        return np.argmax(self.agentOutput)
    
    def getSelectedActionFollowingRandomPolicy(self):
        action_probability_based_on_policy = np.ones(self.actionSetSize)*1/(self.actionSetSize)
        
        chosen_action = np.random.choice(self.actionSetSize, 1, p=action_probability_based_on_policy)[0]
        
        return chosen_action
    
    def getSelectedActionFollowingEpsilonGreedyPolicy(self, Epsilon):
        
        if random.random() <= Epsilon :
            chosen_action = self.getSelectedActionFollowingRandomPolicy()
        else :
            chosen_action = self.getSelectedActionFollowingGreedyPolicy()
                    
        return chosen_action
    
    def updateActionValueFunctionsNetworkParameter(self, actionValueFunctionToUpdateIndex, state, reward, finalStateReached):
        
        QTarget = reward
        targetNetOutput = np.zeros(self.actionSetSize)
        
        if finalStateReached == False :
            #Compute Q(s',a') based on the next state
            for i in range (0,self.actionSetSize,1) :
                self.agentTargetNetwork[i].executeModel(state)
                targetNetOutput[i] = self.agentTargetNetwork[i].getNetworkOutput()
        
            #Get argmaxQ(s',a')
            argmaxQ = np.max(targetNetOutput)
            
            QTarget += self.gammaDiscount*argmaxQ
        
        #print("QTarget is :"+str(QTarget))
        
        Qsa = self.agentOutput[actionValueFunctionToUpdateIndex]
        
        #Compute QNetwork Error gradient for the parameter update algorithm /Normalement l'algorithme de back propagation prend le gradient de l'erreur E' quadratique mais celle de cet algorithme utilise une fonction d'erreur appelée Li(theta i)
        #I have to build a vector of zeros where only the updated Q(s,a) will back propagate an error
        Error = 2*(QTarget-Qsa)
        
        #print("Qsa is :"+str(Qsa))
        print('Error is :'+str(Error))
        
        self.agentQNetwork[actionValueFunctionToUpdateIndex].updateModelParameters([Error])
        
        return
    
    def updateTDTargetNetworkParameters(self):
        for i in range (0,self.actionSetSize,1):
            self.agentTargetNetwork[i] = cp.deepcopy(self.agentQNetwork[i])
            
    