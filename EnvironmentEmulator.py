'''
Created on 4 sept. 2024

@author: SSM9
'''

import numpy as np
from pip._vendor.typing_extensions import Self

class EnvironmentEmulator(object):
    
    map = None
    
    goalPosition = [2,2]
    
    '''
    original_map = np.array([
            ['+','+'],
            ['+','+'],
            ['+','V'],
            ])
    
    '''
    original_map = np.array([
            ['+','+','+'],
            ['+','+','+'],
            ['+','+','V'],
            ['+','+','+'],
            ['+','+','+'],
            ])
    
    '''
        original_map = np.array([
            ['+','+','+','+','+'],
            ['+','+','+','+','+'],
            ['+','+','+','+','V'],
            ['+','+','+','+','+'],
            ['+','+','+','+','+'],
            ])
    
    original_map = np.array([
            ['+','+','+','+','+','+','+','+'],
            ['+','+','+','+','+','+','+','+'],
            ['+','+','+','+','+','+','+','+'],
            ['+','+','+','+','+','+','+','+'],
            ['+','+','+','+','+','+','+','V'],
            ['+','+','+','+','+','+','+','+'],
            ['+','+','+','+','+','+','+','+'],
            ])
    
    
    original_map = np.array([
            ['+','B','+','+','+','B','+','+'],
            ['+','+','B','+','O','O','O','+'],
            ['O','O','+','+','O','+','+','B'],
            ['B','+','O','+','+','O','+','O'],
            ['+','+','+','O','+','O','+','V'],
            ['B','B','B','+','+','O','+','+'],
            ['+','O','O','+','+','+','+','B'],
            ])
    #'''
    
    position = None
    previousDistance = None
    distance = None
    reward = None
    game_in_progress = None
    
    actions = None
     

    def __init__(self):
        
        self.position = Position()
        self.map = np.copy(self.original_map)
        
        self.actions = [self.move_up, self.move_down, self.move_left, self.move_right]
        
        self.game_in_progress = True
        
        self.placePositionToken()
        
        self.reward = 0
        self.computeDistance()
        self.previousDistance = self.distance
        
        #self.printMap()
    
    def printPosition(self):
        print('Current Position = x:'+str(self.position.x)+', y:'+str(self.position.y))
        
    def printMap(self):
        print('Map:')
        print(str(self.map))
        print('Current Position = x:'+str(self.position.x)+', y:'+str(self.position.y))
        print('Current reward = '+str(self.reward))
    
    def computeDistance(self):
        self.previousDistance = self.distance
        self.distance = (abs(self.position.y-self.goalPosition[1])+abs(self.position.x-self.goalPosition[0]))
    
    def computeCurrentReward(self):
        
        reward = self.computeReward(self.position.x,self.position.y)
        
        return reward
    
    def computeReward(self, x, y):
        
        if x<0 or y<0 or x>=len(self.map[0]) or y>=len(self.map) :
            reward=-2
        elif self.original_map[y][x]=='B':
            reward =  2
        elif self.original_map[y][x]=='O':
            reward =  -10
        elif self.original_map[y][x]=='V':
            reward =  100
        else :
            '''
            self.computeDistance()
            if self.previousDistance >= self.distance :
                reward = +1
            else :
                reward = -2
            '''
            reward = -1
        
        reward = reward
        
        return reward/1000
    
    def computeOutOfBoundReward(self):
        
        outboundReward = -10
        
        reward = outboundReward
        
        return reward/1000
    
    def placePositionToken(self):
        self.map = np.copy(self.original_map)
        self.map[self.position.y][self.position.x]='X'
        
        if self.original_map[self.position.y][self.position.x]=='V' or self.original_map[self.position.y][self.position.x]=='O':
            self.game_in_progress = False
    
    def moveToken(self, x = None ,y = None ):
        if self.game_in_progress == False :
            return
        
        if x == None :
            x = self.position.x
        
        if y == None :
            y = self.position.y
        
                
        if x<0 or y<0 or x>=len(self.map[0]) or y>=len(self.map) :
            return
        else :
            self.position.x = x
            self.position.y = y
            
        return
        
    
    def move_up(self):
        
        if self.game_in_progress == False :
            return
                
        if self.position.y==0:
            self.reward = self.computeOutOfBoundReward()
            #self.game_in_progress = False
            
        else :
            self.position.y = self.position.y-1
            self.reward = self.computeCurrentReward()
                    
        self.placePositionToken()
        
        return self.reward
    
    def move_down(self):
        
        if self.game_in_progress == False :
            return
                
        if self.position.y==len(self.map)-1:
            self.reward = self.computeOutOfBoundReward()
            #self.game_in_progress = False
            
        else :
            self.position.y = self.position.y+1
            self.reward = self.computeCurrentReward()

        
        self.placePositionToken()
        
        return self.reward
    
    
    def move_left(self):
        
        if self.game_in_progress == False :
            return
                
        if self.position.x==0:
            self.reward = self.computeOutOfBoundReward()
            #self.game_in_progress = False
            
        else :
            self.position.x = self.position.x-1
            self.reward = self.computeCurrentReward()
        
        
        self.placePositionToken()
        
        return self.reward
    
    def move_right(self):
        
        if self.game_in_progress == False :
            return
                
        if self.position.x==len(self.map[0])-1:
            self.reward = self.computeOutOfBoundReward()
            #self.game_in_progress = False
            
        else :
            self.position.x = self.position.x+1
            self.reward = self.computeCurrentReward()
        
        self.placePositionToken()
        
        return self.reward
    
    #This function acts as the interpreter system state return function in charge of feeding the agent with inputs
    def get_environment_state(self):
        #We send back to the network the current coordinates and the direct reward that you can get from 
        
        height = (len(self.map)-1)
        width = (len(self.map[0])-1)
                
        #state = [self.position.x/width, self.position.y/height, self.distance/(height+width)]
         
        x= self.position.x
        y = self.position.y
        
        self.computeDistance()
        
        up = self.computeState(x, y-1)
        down = self.computeState(x, y+1)
        left = self.computeState(x-1, y)
        right = self.computeState(x+1, y)
        
        #state = [self.position.x/(width), self.position.y/(height), self.distance/((height+width)),up,down,left,right]
        state = [self.position.x/(10*width), self.position.y/(10*height),up,down,left,right]
        
        return state
    
    
    def computeState(self, x, y):
        
        #+ = 0
        #bonus B = 1
        #Victory V = 2
        #hole O or offlimit = 3
        
        if x<0 or y<0 or x>=len(self.map[0]) or y>=len(self.map) :
            positionType =3
        elif self.original_map[y][x]=='B':
            positionType =1
        elif self.original_map[y][x]=='O':
            positionType =4
        elif self.original_map[y][x]=='V':
            positionType =2
        else :
            positionType=0
                
        return positionType/40
    
class Position(object):
    
    x = None
    y = None
    
    def __init__(self):
        self.x=0
        self.y=0
        
        