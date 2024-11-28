'''
Created on 22 nov. 2024

@author: SSM9
'''

from ArtificialNeuronNetwork.CNNComponents.ConvolutionalNeuronNetwork import ConvolutionalNeuronNetwork

import imageio.v3 as iio
import numpy as np

import os
from enum import Enum

class Forme(Enum):
    carre = [1,0]
    cercle = [0,1]

if __name__ == '__main__':
    
    myCNN = ConvolutionalNeuronNetwork()
    
    #import dataSet
    directory = "E:\\users\\sami\\trash\\dataset"
    
    files = os.listdir(directory)
    
    #performTraining
    input_data_set = []#zeros((len(files),50,50,3))
    expected_results = []
    
    for file in files:
        if file.endswith(".png"):
            
            label = os.path.basename(file).split('.')[0]            
            img = iio.imread(os.path.join(directory, file), pilmode='L')
            
            img = img.reshape(50,50,1)
            
            input_data_set.append(img)
            expected_results.append(Forme[label]._value_)
    
            # Prints only text file present in My Folder
            #print(file)
            print(os.path.basename(file).split('.')[0])
            #print(img)
    
    input_data_set = np.array(input_data_set)
    expected_results = np.array(expected_results)
    
    for i in range (0, 5000, 1) :
        print("let's do model training")
        performance = myCNN.supervisedModelTrainingEpochExecution(input_data_set, expected_results)
        
        if performance < 1e-3:
            break
        
    
    #test 1
    result = myCNN.executeModel(input_data_set[0])
    
    if result.all() == np.array([1,0]).all():
        print('carré')
    else:
        print('cercle')
    
    pass


    