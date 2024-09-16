'''
Created on 6 sept. 2024

@author: SSM9
'''

import numpy as np

from ReinforcementDeepLearningAgent import ReinforcementDeepLearningAgent
from EnvironmentEmulator import EnvironmentEmulator
from ExperienceReplayMemory import ExperienceReplayMemory

if __name__ == '__main__':
    
    #init memory of <s,a,s',r> tuple to train the DQL agent
    memorySize = 1000
    experienceReplayMemory =    ExperienceReplayMemory(memorySize)
    batchSize = 32
    
    myAgent = ReinforcementDeepLearningAgent(7,4,0.2) 
    
    #Reset environment state
    #Creation of the environment
    #game = EnvironmentEmulator() 
        
    for epoch in range (0, 1000, 1):
        
        #Reset environment state
        #Creation of the environment
        game = EnvironmentEmulator() 
        
        Epsilon = 1/np.sqrt(epoch+1)
                
        myAgent.gammaDiscount = 0.2
            
        for i in range (0, 10, 1):
                        
            if game.game_in_progress == False :
                game = EnvironmentEmulator()
                #break    
            
            currentState = game.get_environment_state()
            
            myAgent.refreshAgentActionValueFunctions(currentState)
            
            chosenAction = myAgent.getSelectedActionFollowingEpsilonGreedyPolicy(Epsilon)
            
            game.actions[chosenAction]()
            
            resultingState = game.get_environment_state()
            
            reward = game.reward
                        
            finalStateReached = not game.game_in_progress
            
            #update parameters based on the latest transition befor experience replay
            myAgent.updateQNetworkParametersUsingDoubleDeepQLearning(currentState, chosenAction, reward, resultingState, finalStateReached)
            
            #Experience replay = after taking an action we save <S,S',A,R> into a memory and we perform network training on a batch of saved transitions.           
            #store Transition into Experience replay memory
            experienceReplayMemory.appendTransition(currentState, chosenAction, reward, resultingState, finalStateReached)
            
            
            #Start Training based on a batch of previous Experience
            #get batch from memory
            batch = experienceReplayMemory.getRandomBatch(batchSize)
            
            #update network parameters for each element of the batch
            for transition in batch :
                myAgent.updateQNetworkParametersUsingDoubleDeepQLearning(transition.currentState, transition.chosenAction, transition.reward, transition.resultingState, transition.resultingStateFinal)
        
        
        myAgent.updateTargetNetworkParameters()
        
    
    #Reset environment state
    #Creation of the environment
    game = EnvironmentEmulator() 

    game.printMap()
    
    for i in range (0, 2000, 1):
                    
        myAgent.refreshAgentActionValueFunctions(game.get_environment_state())
        
        print("Q(s,a) of the agent are :"+str(myAgent.agentOutput))
            
        chosenAction = myAgent.getSelectedActionFollowingGreedyPolicy()
            
        print("Selected action is the number :"+str(chosenAction))
            
        game.actions[chosenAction]()
        
        game.printMap()
        
        if game.game_in_progress == False :
            break
        
        input('next turn')
    
    
    pass
