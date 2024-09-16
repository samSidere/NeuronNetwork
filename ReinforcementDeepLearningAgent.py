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


class ReinforcementDeepLearningAgent(object):
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
    
    agentQNetwork = None
    agentTargetNetwork = None
    
    agentOutput = []
    

    def __init__(self, stateSize = 1 , actionSetSize = 1, gammaDiscount=0 ):
        
        self.stateSize = stateSize
        self.actionSetSize = actionSetSize
        self.gammaDiscount = gammaDiscount
        
        
        #Q network is the neural network at the center of the agent. It will compute action value functions in order to drive agent decisions
        #In this environment, it consists in one neuron network in charge of computing each Q(s,a) action pair output
        
        self.agentQNetwork = NeuronNetwork(self.stateSize, self.actionSetSize, 2, 12, 0.001,
                                             Cost_functions.mean_squared_error, 
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                             Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
        
        #Target network is in charge of computation of TD target that the network will use for training. It uses a copy of Q Net parameters and is updated every N epochs
        self.agentTargetNetwork = cp.deepcopy(self.agentQNetwork)
        
        self.agentOutput = np.zeros(self.actionSetSize)
    
    def copyParamToAgentTargetNetwork(self) : 
        self.agentTargetNework = cp.deepcopy(self.agentQNetwork)        
    
    def refreshAgentActionValueFunctions(self, state):
        #contrary to compute. refresh will save Q(s,a) values to agent output to let user select one action based on policy and perform interaction with environment
        self.agentOutput = self.computeQNetworkActionValueFunctions(state)
        return self.agentOutput
        
    def computeQNetworkActionValueFunctions(self, state):
        #Connect state to agent Q network input
        self.agentQNetwork.executeModel(state)
        return self.agentQNetwork.getNetworkOutput()
        
    def computeTargetNetworkActionValueFunctions(self, state):
        #Connect state to agent Q network input
        self.agentTargetNetwork.executeModel(state)
        return self.agentTargetNetwork.getNetworkOutput()
        
                
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
    
    def updateQNetworkParametersUsingDeepQLearning(self, entryState, chosenAction, reward, resultingState, finalStateReached):
        
        #ActionValueFunctions = Q(s,a)
        #First compute Qsa S->QNetwork-> Qsa
        Qst0_at0 = self.computeQNetworkActionValueFunctions(entryState)[chosenAction]
        
        
        #Second compute S'->TargetNetwork -> Qs'a' and get argmax(Q(s'a')) if S' is not a final state. Otherwise TDTarget stays equal to reward
        if finalStateReached == False :
            #Compute Q(s',a') based on the next state
            Qst1_at1 = self.computeTargetNetworkActionValueFunctions(resultingState)
        
            #Get argmaxQ(s',a')
            argmaxQst1_at1 = np.max(Qst1_at1)
        else:
            argmaxQst1_at1 = 0   
        
        #Compute TD Target -> R+Gamma*Qs'a'
        TDTarget = reward + self.gammaDiscount*argmaxQst1_at1
                
        #Compute Error -> TDTarget - Qsa
                
        #Compute QNetwork Error gradient for the parameter update algorithm /Normalement l'algorithme de back propagation prend le gradient de l'erreur E' quadratique mais celle de cet algorithme utilise une fonction d'erreur appelée Li(theta i)
        #I have to build a vector of zeros where only the updated Q(s,a) will back propagate an error
        Error = np.zeros(self.actionSetSize)
        Error[chosenAction] = 2*(TDTarget-Qst0_at0)
        
        print('Error is :'+str(np.square(TDTarget-Qst0_at0)))        
        self.agentQNetwork.updateModelParameters(Error)
        
        
        return
    
    def updateQNetworkParametersUsingDoubleDeepQLearning(self, entryState, chosenAction, reward, resultingState, finalStateReached):
        
        #ActionValueFunctions = Q(s,a)
        #First compute Qsa S->QNetwork-> Qsa
        Qst0_at0 = self.computeQNetworkActionValueFunctions(entryState)[chosenAction]
        
        
        #Second compute S'->TargetNetwork -> Qs'a' and get argmax(Q(s'a')) if S' is not a final state. Otherwise TDTarget stays equal to reward
        if finalStateReached == False :
            Qst1_at1_fromQNet = self.computeQNetworkActionValueFunctions(resultingState)
            
            #Compute Q(s',a') based on the next state
            #First QNetwork will compute Q(S',a) in order to identify the A that it would have chosen following the greedy policy for state s'
            Qst1_at1_fromTargetNet = self.computeTargetNetworkActionValueFunctions(resultingState)
        
            #Get Q(s',a') from the Target Network based on the action that the Q Network would have chosen previously
            argmaxQst1_at1 = Qst1_at1_fromTargetNet[np.argmax(Qst1_at1_fromQNet)]
        else:
            argmaxQst1_at1 = 0   
        
        #Compute TD Target -> R+Gamma*Qs'a'
        TDTarget = reward + self.gammaDiscount*argmaxQst1_at1
                
        #Compute Error -> TDTarget - Qsa
                
        #Compute QNetwork Error gradient for the parameter update algorithm /Normalement l'algorithme de back propagation prend le gradient de l'erreur E' quadratique mais celle de cet algorithme utilise une fonction d'erreur appelée Li(theta i)
        #I have to build a vector of zeros where only the updated Q(s,a) will back propagate an error
        Error = np.zeros(self.actionSetSize)
        Error[chosenAction] = 2*(TDTarget-Qst0_at0)
        
        print('Error is :'+str(np.square(TDTarget-Qst0_at0)))        
        self.agentQNetwork.updateModelParameters(Error)
        
        
        return
    
    def updateQNetworkParametersFromLastTransitionUsingDeepQLearning(self, actionValueFunctionToUpdateIndex, state, reward, finalStateReached):
        
        #ActionValueFunctions = Q(s,a)
        #First compute Qsa S->QNetwork-> Qsa
        #Second compute S'->TargetNetwork -> Qs'a'
        #Compute TD Target -> R+Gamma*Qs'a'
        #Compute Error -> TDTarget - Qsa
        
        TDTarget = reward
        targetNetOutput = np.zeros(self.actionSetSize)
        
        if finalStateReached == False :
            #Compute Q(s',a') based on the next state
            targetNetOutput = self.computeTargetNetworkActionValueFunctions(state)
        
            #Get argmaxQ(s',a')
            argmaxQ = np.max(targetNetOutput)
            
            TDTarget += self.gammaDiscount*argmaxQ
        
        #print("QTarget is :"+str(QTarget))
        
        Qsa = self.agentOutput[actionValueFunctionToUpdateIndex]
        
        #Compute QNetwork Error gradient for the parameter update algorithm /Normalement l'algorithme de back propagation prend le gradient de l'erreur E' quadratique mais celle de cet algorithme utilise une fonction d'erreur appelée Li(theta i)
        #I have to build a vector of zeros where only the updated Q(s,a) will back propagate an error
        Error = np.zeros(self.actionSetSize)
        Error[actionValueFunctionToUpdateIndex] = 2*(TDTarget-Qsa)
        
        #print("Qsa is :"+str(Qsa))
        print('Error is :'+str(np.square(TDTarget-Qsa)))
        self.agentQNetwork.updateModelParameters(Error)
        
        return
    
    def updateTargetNetworkParameters(self):
        self.agentTargetNetwork = cp.deepcopy(self.agentQNetwork)
            
    