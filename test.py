'''
Created on 5 sept. 2024

@author: SSM9
'''

import numpy as np
from ExperienceReplayMemory import ExperienceReplayMemory

if __name__ == '__main__':
    
    memory = ExperienceReplayMemory(10)
    
    print(str(len(memory.memory))+'\n')

    for i in range (0,20,1) :
        memory.appendTransition(i, i, i, i)
    
    print(str(len(memory.memory))+'\n')
    
    toto=memory.getRandomBatch(5)
    
    for item in toto :
        print(str(item.currentState))
    
    print('\n'+str(len(memory.memory))+'\n')
    
    for i in range (70,120,1) :
        memory.appendTransition(i, i, i, i)
        
    print('\n'+str(len(memory.memory))+'\n')
    
    toto=memory.getRandomBatch(25)
    
    for item in toto :
        print(str(item.currentState))
        
    print('\n'+str(len(memory.memory))+'\n')
    
    
    pass