'''
Created on 20 sept. 2024

@author: SSM9
'''
from ArtificialNeuronNetwork.Neuron import Neuron
from ArtificialNeuronNetwork.NeuronLayer import NeuronLayer
from ArtificialNeuronNetwork.NeuronNetwork import NeuronNetwork
import ArtificialNeuronNetwork.Activation_functions as Activation_functions
import ArtificialNeuronNetwork.Cost_functions as Cost_functions
from ArtificialNeuronNetwork.Neuron import Optimizer

def my_function():
    pass

class MyClass(object):
    def method(self):
        pass


if __name__ == '__main__':
    print("=====================================================Precondition Tests===================================================")
    
    print(my_function.__name__)         # gives "my_function"
    print(MyClass.method.__name__)      # gives "method"

    print(my_function.__qualname__)     # gives "my_function"
    print(MyClass.method.__qualname__)  # gives "MyClass.method"
    
    print("=====================================================Save and Load Tests For Neuron Level===================================================")
    toto = Neuron([0,1,2,3,7],
                  Activation_functions.linearActivationFun,Activation_functions.der_linearActivationFun,
                  66,
                  Optimizer.MOMENTUM,
                  0.3)
        
    print("DeepQLAgentTester hyper params are"+toto.getHyperParameters())
    
    tata = Neuron()
    
    print("tata hyper params before loading are"+tata.getHyperParameters())
    
    tata.loadHyperParameters(toto.getHyperParameters())
    
    print("tata content is "+str(tata.__dict__))
    print("tata hyper params after loading are"+tata.getHyperParameters())
    
    print()
    print()
    
    if(tata.getHyperParameters()==toto.getHyperParameters()):
        print("Neuron save and load test was a success")
    else :
        print("Neuron save and load test was a failure")
    
    print("=====================================================Save and Load Tests For Layer Level===================================================")
    toto = NeuronLayer(10, 2, Activation_functions.sigmoidLogisticFun, Activation_functions.der_sigmoidLogisticFun, 79, False, Optimizer.MOMENTUM, 0.3)
        
    print("DeepQLAgentTester hyper params are"+toto.getHyperParameters())
    
    tata = NeuronLayer()
    
    print("tata hyper params before loading are"+tata.getHyperParameters())
    
    tata.loadHyperParameters(toto.getHyperParameters())
    
    print("tata content is "+str(tata.__dict__))
    print("tata hyper params after loading are"+tata.getHyperParameters())
    
    print()
    print()
    
    if(tata.getHyperParameters()==toto.getHyperParameters()):
        print("NeuronLayer save and load test was a success")
    else :
        print("NeuronLayer save and load test was a failure")
        
        
    print("=====================================================Save and Load Tests For Network Level===================================================")
    toto = NeuronNetwork(5, 2, 2, 12, 0.01,
                                             Cost_functions.mean_squared_error, 
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                             Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun,
                                             True,
                                             Optimizer.MOMENTUM, 0.3)
        
    print("DeepQLAgentTester hyper params are"+toto.getHyperParameters())
    
    tata = NeuronNetwork()
    
    print("tata hyper params before loading are"+tata.getHyperParameters())
    
    tata.loadHyperParameters(toto.getHyperParameters())
    
    print("tata content is "+str(tata.__dict__))
    print("tata hyper params after loading are"+tata.getHyperParameters())
    
    print()
    print()
    
    if(tata.getHyperParameters()==toto.getHyperParameters()):
        print("NeuronNetwork save and load test was a success")
    else :
        print("NeuronNetwork save and load test was a failure")
    
    print("=====================================================Save and Load Tests For Network Level Using a file===================================================")
    toto = NeuronNetwork(5, 2, 2, 12, 0.01,
                                             Cost_functions.mean_squared_error, 
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun, 
                                             Activation_functions.reLUFun, Activation_functions.der_reLUFun,
                                             Activation_functions.linearActivationFun, Activation_functions.der_linearActivationFun,
                                             True,
                                             Optimizer.MOMENTUM, 0.3)
    
    print("DeepQLAgentTester hyper params are"+toto.getHyperParameters())
    toto.saveNetworkParameterIntofile("E:\\users\\sami\\trash\\dump.json")
    
    tata = NeuronNetwork()
    
    print("tata hyper params before loading are"+tata.getHyperParameters())
    
    tata.loadNetworkParameterFromfile("E:\\users\\sami\\trash\\dump.json")
    
    print("tata hyper params after loading are"+tata.getHyperParameters())
    
    print()
    print()
    
    if(tata.getHyperParameters()==toto.getHyperParameters()):
        print("NeuronNetwork from file save and load test was a success")
    else :
        print("NeuronNetwork from file save and load test was a failure")
    
    pass