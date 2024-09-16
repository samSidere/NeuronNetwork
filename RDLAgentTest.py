'''
Created on 4 sept. 2024

@author: SSM9
'''

from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork
from ArtificialNeuronNetwork.Neuron import Neuron
from ArtificialNeuronNetwork import Cost_functions
from ArtificialNeuronNetwork import Activation_functions

from Environment_Test import EnvironmentEmulator

import numpy as np

import copy as cp



if __name__ == '__main__':
    
    #Creation of the Agent
    
    Gamma_discount = 0
    
    #Q network is the neural network at the center of the agent. It will compute action value functions in order to drive agent decisions
    
    agentQNetwork = NeuronNetwork(2, 1, 2, 12, 0.001,
                                         Cost_functions.mean_squared_error, 
                                         Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                         Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                         Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun)
    
    
    
    #Target network is in charge of computation of TD target that the network will use for training. It uses a copy of Q Net parameters and is updated every N epochs
    agentTargetNework = cp.deepcopy(agentQNetwork)   
    
    for epoch in range (0,1,1) :
    
        #Target network is in charge of computation of TD target that the network will use for training. It uses a copy of Q Net parameters and is updated every N epochs
        agentTargetNework = cp.deepcopy(agentQNetwork)
     
        Score = 0
        Number_of_Steps = 0
        Epsilon = np.float64(0)
        #Gamma_discount = epoch/200
        
        #Reset environment state
        #Creation of the environment
        game = EnvironmentEmulator()      
        
        for i in range (0,10000,1) :
            
            #Reset environment state
            #Creation of the environment
            game = EnvironmentEmulator()  
                                
            #Connection of the environment to the agent
            input_data = game.get_environment_state()
            
            #print("state of the environment is :"+str(input_data))
            
            output_data = agentQNetwork.executeModel(input_data)
            
            '''
            for i in range(0, len(output_data), 1) :
                
                print("Q(s,a) : "+str(output_data[i].output_value))
                
            print('index of the argmaxQ(s,a) '+str(np.argmax(agentQNetwork.getNetworkOutput())))
            #'''
            #'''
            Epsilon = 1/np.sqrt(i+1)
            action_probability_based_on_policy = np.ones(agentQNetwork.number_of_outputs)/Epsilon/(agentQNetwork.number_of_outputs-1)
            action_probability_based_on_policy[np.argmax(agentQNetwork.getNetworkOutput)]=1-Epsilon
            
            
            chosen_action_based_on_e_greedy_policy = np.random.choice(agentQNetwork.number_of_outputs, 1, p=action_probability_based_on_policy)
            
            #print("Epsilon = "+str(Epsilon))
            #print("Chosen action = "+str(chosen_action_based_on_e_greedy_policy))
                                             
            game.actions[chosen_action_based_on_e_greedy_policy]()
            Qsa = agentQNetwork.getNetworkOutput()[chosen_action_based_on_e_greedy_policy]
            #'''
            
            '''
            game.actions[np.argmax(agentQNetwork.getNetworkOutput)]()
                        
            Qsa = agentQNetwork.getNetworkOutput()[np.argmax(agentQNetwork.getNetworkOutput)]
            #'''
            
            '''
            game.actions[0]()
                        
            Qsa = agentQNetwork.getNetworkOutput()[0]
            #'''
            
            if game.game_in_progress == True :
            
                #input_data = game.get_environment_state()
            
                #game.printMap()
            
                #print("state of the environment is :"+str(input_data))
            
                #Compute argmaxQ(s',a') from Target Network
                #output_data = agentTargetNework.executeModel(input_data)
                agentTargetNework.executeModel(input_data)
                
                '''
                for i in range(0, len(output_data), 1) :
                
                    print("Q(s,a) : "+str(output_data[i].output_value))
                
                print('index of the argmaxQ(s,a) '+str(np.argmax(agentQNetwork.getNetworkOutput())))
                #'''
                
                #Get argmaxQ(s',a')
                argmaxQ = np.max(agentTargetNework.getNetworkOutput())
            
                QTarget = game.reward#+Gamma_discount*argmaxQ
            else :
                QTarget = game.reward
            
            print("Qsa is :"+str(Qsa))
            print("QTarget is :"+str(QTarget))
            #Compute QNetwork Error gradient for the parameter update algorithm /Normalement l'algorithme de back propagation prend le gradient de l'erreur E' quadratique mais celle de cet algorithme utilise une fonction d'erreur appelée Li(theta i)
            #I have to build a vector of zeros where only the updated Q(s,a) will back propagate an error
            Error = 2*(QTarget-Qsa)
            
            print('Error is :'+str(Error))
            
            Score += game.reward
            Number_of_Steps+=1          
              
            
                   
            agentQNetwork.updateModelParameters(Error,np.argmax(agentQNetwork.getNetworkOutput()))
            
            if game.game_in_progress == False :
                break
            
            #game.printPosition()
        
        #game.printMap()
        print('Score for epoch '+str(epoch)+' : '+str(Score))
        print('Number of steps for epoch '+str(epoch)+' : '+str(Number_of_Steps))
    #'''    
    #show one game
    Score = 0
    Number_of_Steps = 0
        
    #Reset environment state
    #Creation of the environment
    game = EnvironmentEmulator()      
        
    for i in range (0,1000,1) :
                    
        #Connection of the environment to the agent
        input_data = game.get_environment_state()
            
        #print("state of the environment is :"+str(input_data))
            
        output_data = agentQNetwork.executeModel(input_data)
        
        
        for i in range(0, len(output_data), 1) :
               
            print("Q(s,a) : "+str(output_data[i].output_value))
                
        
        print(str(agentQNetwork.getNetworkOutput()))
        print('index of the argmaxQ(s,a) '+str(np.argmax(agentQNetwork.getNetworkOutput())))
            
        game.actions[np.argmax(agentQNetwork.getNetworkOutput())]()
            
        Score += game.reward
        Number_of_Steps+=1          
            
        
        
        game.printMap()
        if game.game_in_progress == False :
            break
        
        input('next turn')
        
    print('Score for last game '+str(epoch)+' : '+str(Score))
    print('Number of steps for last game '+str(epoch)+' : '+str(Number_of_Steps))
    #'''
    
    pass