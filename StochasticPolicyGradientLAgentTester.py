'''
Created on 6 sept. 2024

@author: SSM9
'''

import numpy as np

from ReinforcementDeepLearningAgents.StochasticPolicyGradientLearningAgent import StochasticPolicyGradientLearningAgent
from EnvironmentEmulator import EnvironmentEmulator




if __name__ == '__main__':
        
    print('Do you want to load an previously created Agent?')
    filename = input("Insert your Agent parameters file path")
    
    if filename =="":
        myAgent = StochasticPolicyGradientLearningAgent(6,4,0.6)
    else :
        myAgent = StochasticPolicyGradientLearningAgent(gammaDiscount=0.6,filename=filename)
        
    print('Do you want to change learning rate?')
    rate = input("Insert learning rate alpha")
    
    if rate !="":
        myAgent.agentPolicyNetwork.correction_coeff = np.float64(rate)
    
    counter = 0
    victory_rate = 0
    
    max_episodes = 1000
    max_game_duration = 20
    
    
    for episode in range (0, max_episodes, 1):
        
        #Reset environment state
        #Creation of the environment
        game = EnvironmentEmulator()
        score = 0
        number_of_turn=max_game_duration
                
        #Create the History of the episode in order to feed the training algorithm
        episodeComputedPolicies = []
        episodeSelectedPolicies = []
        episodeRewards = []
        
        Epsilon = 1/np.sqrt(episode+1)
        
        for i in range (0, max_game_duration, 1):
            
            counter +=1
                        
            if game.game_in_progress == False :
                number_of_turn=i
                break    
            
            currentState = game.get_environment_state()
            
            computedPoliciesValues = myAgent.computeStochasticPolicy(currentState)
                        
            episodeComputedPolicies.append(computedPoliciesValues)
            
            chosenAction = myAgent.sampleActionBasedOnStochasticPolicy(currentState)
            
            episodeSelectedPolicies.append(computedPoliciesValues[chosenAction])
            
            game.actions[chosenAction]()
            
            episodeRewards.append(game.reward)
            
            #Update kpi
            score += game.reward
            if game.reward == 100/1000 : #value for victory
                victory_rate+=1
                
            finalStateReached = not game.game_in_progress
            
            
        print("for episode "+str(episode)+" : score is "+str(score)+" \t in "+str(number_of_turn)+"\t turns and victory rate is "+str((victory_rate/max_episodes)*100)+'%')
        #update parameters based on the results of the episode
        myAgent.updatePolicyNetworkParametersOneEpisodeMonteCarloReinforceLearning(episodeComputedPolicies, episodeSelectedPolicies, episodeRewards)
        
        
    print('Do you want to save this Agent?')
    filename = input("Insert your Agent parameters file path")
    
    if filename !="":
        myAgent.saveAgentParamtoFile(filename)
    
    #Reset environment state for Testing
    #Creation of the environment
    game = EnvironmentEmulator() 

    game.printMap()
    
    for i in range (0, 2000, 1):
        
        chosenAction = myAgent.sampleActionBasedOnStochasticPolicy(game.get_environment_state())
            
        print("Selected action is the number :"+str(chosenAction))
            
        game.actions[chosenAction]()
        
        game.printMap()
        
        if game.game_in_progress == False :
            break
        
        input('next turn')
    
    
    pass
