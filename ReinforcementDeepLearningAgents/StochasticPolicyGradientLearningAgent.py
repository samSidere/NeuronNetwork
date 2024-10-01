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
        
        #print('network output before softmax is : '+str(self.agentPolicyNetwork.getNetworkOutput()))
        
        #Use softmax to return probability distribution as output
        networkOutput = Cost_functions.doSoftmax(self.agentPolicyNetwork.getNetworkOutput())
        
        return networkOutput
        
    
    #This function will select an action based on the highest policy value computed by the agent's policy network            
    def sampleActionBasedOnStochasticPolicy(self, state):
        
        sampledPolicyValues = self.computeStochasticPolicy(state)
        
        return np.argmax(sampledPolicyValues)
    
    #This function will select an action based on the highest policy value computed by the agent's policy network            
    def sampleActionBasedOnStochasticPolicyEpsilonGreedy(self, state, Epsilon):
        
        sampledPolicyValues = self.computeStochasticPolicy(state)
        
        if random.random() <= Epsilon :
            
            action_probability_based_on_policy = np.ones(self.actionSetSize)*1/(self.actionSetSize)
        
            chosen_action = np.random.choice(self.actionSetSize, 1, p=action_probability_based_on_policy)[0]
        else :
            chosen_action = np.argmax(sampledPolicyValues)
        
        return chosen_action
    
    
    #This function let the user train the Policy Network using a Monte Carlo Reinforce stochastic policy gradient algorithm on one trajectory
    def updatePolicyNetworkParametersOneEpisodeMonteCarloReinforceLearning(self, episodeComputedPolicies, episodeSelectedPolicies, episodeRewards):
        
        '''
        #For each step of the episode
        #Compute gradient of objective function J(Θ)
        #update NetworkParametersBased J(Θ) Gradient Ascend Method
        #
        # Trick has to be put in place in order to push error through network
        # We will hack the w(t+1)=w(t).-alpha.(ytrue-ypred).-der_activation_fun(sum(wi.xi)).xi formula used to perform gradient descent in order to do the ascent
        # the  w(t+1) = w(t) -alpha * -der_linear_activation_fun(sum(wi.xi)*xi*(GradJ(Θ)/xi)
        # => w(t+1) = w(t) -alpha * -1*xi*(GradJ(Θ)/xi)
        # => w(t+1) = w(t) +alpha*xi*(GradJ(Θ)/xi)
        '''
        
        #compute δij as a diagonal unit matrix
        δij= np.diag(np.diag(np.ones((self.actionSetSize,self.actionSetSize))))
        
        #partial factor will store a factor that is used to compute the derivative of logπ(at|st)
        partialFactor = np.float64(0)
        
        #in this algorithm we update Θ for each step of the episode that has been played        
        for t in range(0,len(episodeComputedPolicies),1):
            
            #Compute initial return Gt=sum t=0->T (γ**t*Rt)
            episodeReturn = np.float64(0)
            for i in range (t,len(episodeRewards),1) :
                episodeReturn+=episodeRewards[i]*np.power(self.gammaDiscount,i-t)
            
            #print("episode return is at step "+str(t)+" : "+str(episodeReturn))
            
            #Compute gradient of objective function J(Θ) **based on the trick
            partialFactor+=1/episodeSelectedPolicies[t]
            
            gradJΘCommonFactors=partialFactor*episodeReturn
            
            #Create the Jacobian matrix of the π(at|st) that the network computed in the episode
            #Compute Si matrix
            Si=np.array(episodeComputedPolicies[t])
            for i in range(1,self.actionSetSize,1):
                Si=np.concatenate((Si,np.array(episodeComputedPolicies[t])))
            
            Si=Si.reshape(self.actionSetSize,self.actionSetSize)
            
            #Compute Sj matrix
            Sj=Si.transpose()
            
            #Compute JacobianMatrix of the policyResult(t) that we will use to update Θ at this step
            gradJΘt=(Si*(δij-Sj))*gradJΘCommonFactors
            #We want to pass the partial derivatives of the function per class
            gradJΘt=gradJΘt.transpose()
                        
            #Then for each class of the softmax vector update the output layer and do back propagation
            for i in range (0,self.actionSetSize,1):
                self.agentPolicyNetwork.updateModelParameters(gradJΘt[i])
            
        return
    
                
    def saveAgentParamtoFile(self, filename):
        
        self.agentPolicyNetwork.saveNetworkParameterIntofile(filename)
        
        return