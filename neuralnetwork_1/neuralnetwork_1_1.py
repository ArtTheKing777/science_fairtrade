'''
Created on 16 jan. 2019

@author: arttheking
'''
import numpy as np
#from posix import sync



def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(2)

synaptic_weights = 2 * np.random.random((3,1)) - 1

print('random starting sy naptic_weights: ')
print(synaptic_weights)

for iteration in range(20000):
    
    input_layer = training_inputs
    
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    
    error = training_outputs - outputs
    
    adjustments = error * sigmoid_derivative(outputs)
    
    synaptic_weights += np.dot(input_layer.T, adjustments)
    if iteration % 3000 == 0:
        print(adjustments,"op,",iteration,"it")
    
print('synaptic_weight after training: ')  
print(synaptic_weights)

print('output after training: ')
print(outputs)
