'''
Created on 13 sept. 2024

@author: SSM9
'''
import numpy as np
import random
from collections import deque

class ExperienceReplayMemory(object):
    
    memory = None

    def __init__(self, capacity):
        
        self.memory = deque(maxlen=capacity)
        

    def appendTransition(self,currentState, chosenAction, reward, resultingState, resultingStateFinal=False):
        self.memory.append(Transition(currentState, chosenAction, reward, resultingState, resultingStateFinal))
        
 
    
    def getRandomBatch(self, size):
        
        if size > len(self.memory):
            batch = random.sample(list(self.memory), len(self.memory))
        else:
            batch = random.sample(list(self.memory), size)
                    
        return batch
        
        
class Transition (object):
    
    currentState = None
    resultingState = None
    chosenAction = None
    reward = None
    resultingStateFinal = None

    def __init__(self, currentState, chosenAction, reward, resultingState, resultingStateFinal=False):
        self.currentState = currentState
        self.chosenAction = chosenAction
        self.reward = reward
        self.resultingState = resultingState
        self.resultingStateFinal=resultingStateFinal
        

        