'''
Created on 6 sept. 2024

@author: SSM9
'''

from ReinforcementDeepLearningAgents.DeepQLearningAgent import DeepQLearningAgent
from EnvironmentEmulator import EnvironmentEmulator

if __name__ == '__main__':
    
    print('Do you want to load an previously created Agent?')
    filename = input("Insert your Agent parameters file path")
    
    if filename =="":
        pass
    else :
        myAgent = DeepQLearningAgent(filename=filename)
    
       
    counter = 0
    victory_rate = 0
    
    #Reset environment state for Testing
    #Creation of the environment
    game = EnvironmentEmulator() 

    game.printMap()
    
    while True :
                    
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
