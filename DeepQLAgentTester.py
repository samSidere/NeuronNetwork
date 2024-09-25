'''
Created on 6 sept. 2024

@author: SSM9
'''

import numpy as np

from ReinforcementDeepLearningAgents.DeepQLearningAgent import DeepQLearningAgent
from EnvironmentEmulator import EnvironmentEmulator
from ReinforcementDeepLearningAgents.ExperienceReplayMemory import ExperienceReplayMemory



if __name__ == '__main__':
    
    #init memory of <s,a,s',r> tuple to train the DQL agent
    memorySize = 100
    experienceReplayMemory =    ExperienceReplayMemory(memorySize)
    batchSize = 32
    
    print('Do you want to load an previously created Agent?')
    filename = input("Insert your Agent parameters file path")
    
    if filename =="":
        myAgent = DeepQLearningAgent(6,4,0.2)
    else :
        myAgent = DeepQLearningAgent(gammaDiscount=0.1,filename=filename)
        
    print('Do you want to change learning rate?')
    rate = input("Insert learning rate alpha")
    
    if rate !="":
        myAgent.agentQNetwork.correction_coeff = np.float64(rate)
        myAgent.agentTargetNetwork.correction_coeff = np.float64(rate)
    
    counter = 0
    victory_rate = 0
    
    max_epoch = 100
    max_game_duration = 30
    
    TargetNetRefreshrate = 30
    
    for epoch in range (0, max_epoch, 1):
        
        #Reset environment state
        #Creation of the environment
        game = EnvironmentEmulator()
        score = 0
        number_of_turn=max_game_duration
        Epsilon = 1/np.sqrt(counter+1)
        
        if counter%TargetNetRefreshrate==0 :
            myAgent.updateTargetNetworkParameters()
            
        for i in range (0, max_game_duration, 1):
            
            counter +=1
                        
            if game.game_in_progress == False :
                number_of_turn=i
                break    
            
            currentState = game.get_environment_state()
            
            Qsa_vector = myAgent.computeQNetworkActionValueFunctions(currentState)
            
            chosenAction = myAgent.getSelectedActionFollowingEpsilonGreedyPolicy(Epsilon, Qsa_vector)
            
            game.actions[chosenAction]()
            
            #Update kpi
            score += game.reward
            if game.reward == 100/1000 : #value for victory
                victory_rate+=1
                
            
            resultingState = game.get_environment_state()
            
            reward = game.reward
                        
            finalStateReached = not game.game_in_progress
            
            #update parameters based on the latest transition before experience replay
            myAgent.updateQNetworkParametersUsingDoubleDeepQLearning(currentState, chosenAction, reward, resultingState, finalStateReached)
            
            
            #Experience replay = after taking an action we save <S,S',A,R> into a memory and we perform network training on a batch of saved transitions.           
            #store Transition into Experience replay memory
            experienceReplayMemory.appendTransition(currentState, chosenAction, reward, resultingState, finalStateReached)
            
            if(counter>memorySize):
                #Start Training based on a batch of previous Experience
                #get batch from memory
                batch = experienceReplayMemory.getRandomBatch(batchSize)
            
                #update network parameters for each element of the batch
                for transition in batch :
                    myAgent.updateQNetworkParametersUsingDoubleDeepQLearning(transition.currentState, transition.chosenAction, transition.reward, transition.resultingState, transition.resultingStateFinal)
                
        print("for epoch "+str(epoch)+"score is "+str(score)+" \t in "+str(number_of_turn)+"\t turns and victory rate is "+str(victory_rate/max_epoch))
        
        
    print('Do you want to save this Agent?')
    filename = input("Insert your Agent parameters file path")
    
    if filename !="":
        myAgent.saveAgentParamtoFile(filename)
    
    #Reset environment state for Testing
    #Creation of the environment
    game = EnvironmentEmulator() 

    game.printMap()
    
    for i in range (0, 2000, 1):
                    
        Qsa_vector = myAgent.computeQNetworkActionValueFunctions(game.get_environment_state())
        
        print("Q(s,a) of the agent are :"+str(Qsa_vector))
            
        chosenAction = myAgent.getSelectedActionFollowingGreedyPolicy(Qsa_vector)
            
        print("Selected action is the number :"+str(chosenAction))
            
        game.actions[chosenAction]()
        
        game.printMap()
        
        if game.game_in_progress == False :
            break
        
        input('next turn')
    
    
    pass
