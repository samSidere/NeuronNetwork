'''
Created on 17 sept. 2024

@author: SSM9
'''

import numpy as np

from ArtificialNeuronNetwork.Cost_functions import categorical_cross_entropy,\
    categorical_cross_entropy_Loss



if __name__ == '__main__':
    
    A=[1,0,0,0,0]
    B=[0.996,0.001,0.001,0.001,0.001]
    
    result=categorical_cross_entropy_Loss(A, B)
    
    print('For A ='+str(A)+'And B = '+str(B)+'result is '+str(result))
    
    A=[[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]]
    B=[[0.996,0.001,0.001,0.001,0.001],[0.996,0.001,0.001,0.001,0.001],[0.996,0.001,0.001,0.001,0.001],[0.996,0.001,0.001,0.001,0.001],[0.996,0.001,0.001,0.001,0.001]]
    
    print('For A ='+str(A)+'And B = '+str(B)+'result is '+str(result))
    
    
    pass

