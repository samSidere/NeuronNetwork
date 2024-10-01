'''
Created on 17 sept. 2024

@author: SSM9
'''

import numpy as np

from ArtificialNeuronNetwork.Cost_functions import categorical_cross_entropy
from ArtificialNeuronNetwork.Cost_functions import doSoftmax



if __name__ == '__main__':
    
    toto = np.array([1,0,0,0])
    print('DeepQLAgentTester is '+str(toto))
    toto = doSoftmax(toto)
    print('DeepQLAgentTester is '+str(toto))
    toto = np.sum(toto)
    print('DeepQLAgentTester is '+str(toto))
    
    #'''
    y_true= [0, 0, 2, 1, 2, 1, 1, 0, 2, 1]
    y_true = [y_true,y_true,y_true]
    y_pred = np.transpose([[ 0.9777, -1.7423, -0.6411],
        [ 0.1729, -0.2791, -0.1739],
        [-0.7644,  1.8711, -1.0929],
        [ 1.0369, -0.6144, -0.2865],
        [ 1.1093,  0.1779,  0.4427],
        [ 0.2944, -0.8224, -0.0402],
        [ 0.5885,  1.7270,  1.5069],
        [-0.2755,  0.7618,  0.3418],
        [-0.4669,  1.8668,  0.6125],
        [-0.2345,  1.1015, -2.4268]])
        
    '''
    
    # y_true: True Probability Distribution
    y_true = [1, 0, 0, 0, 0]
    y_true = [y_true,y_true,y_true,y_true,y_true]
 
    # y_pred: Predicted values for each calss
    y_pred = [10, 5, 3, 1, 4]
    y_pred = [y_pred,y_pred,y_pred,y_pred,y_pred]
    #'''
    
    
    #y_true=doSoftmax(y_true)
    for i in range(0,len(y_pred),1):
        y_true[i]=doSoftmax(y_true[i])
        y_pred[i]=doSoftmax(y_pred[i])
    
    print('y_true is '+str(y_true))
    print('y_pred is '+str(y_pred))
    #print('y_predT is '+str(np.transpose(np.log(y_pred))))
    
    print('sum per class is '+str(np.diagonal(np.dot(y_true, np.transpose(np.log(y_pred))))))
    result = categorical_cross_entropy(y_true, y_pred)
    
    print('result is '+str(result))
    
    
    pass

