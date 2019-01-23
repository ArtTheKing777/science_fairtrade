'''
Created on 17 jan. 2019

@author: arttheking
'''
import numpy as np
from random import randrange

x_Box = 800
y_Box = 600



#x1 = #ball location x frame 1 when oppont it
#y1 = #ball location training_outputs frame 1 when oppont it
#x2 = #ball location x frame 2 when oppont it
#y2 = #ball location training_outputs frame 2 when oppont it
def chooseInputs():
    x1 = randrange(0,x_Box)
    y1 = randrange(0,y_Box)
    x2 = x1 + randrange(1,10)
    y2 = y1 + randrange(-10,10)

    return np.array([x1,y1,x2,y2])


allowed_error = 0.1 # 1% is acceptable..
doelperc = 50
aantalfout = 0
aantalgoed = 0

def calculateY(x1,y1,x2,y2):
    y_end = y1-(y1-y2)*((x_Box-x1)/(x2-x1))
    bounces = y_end // y_Box
    flip = bounces % 2
    y_Rend = y_end % y_Box
    if flip == 1:
        result = y_Box - y_Rend
    else:
        result = y_Rend
    
    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


synaptic_weights_toN1 = 2 * np.random.random((4,1)) - 1
synaptic_weights_toN2 = 2 * np.random.random((4,1)) - 1
synaptic_weights_toN3 = 2 * np.random.random((4,1)) - 1
synaptic_weights_toOut = 2 * np.random.random((3,1)) - 1

synaptic_weights_from_IN1 = np.array([synaptic_weights_toN1[0],synaptic_weights_toN2[0],synaptic_weights_toN3[0]])
synaptic_weights_from_IN2 = np.array([synaptic_weights_toN1[1],synaptic_weights_toN2[1],synaptic_weights_toN3[1]])
synaptic_weights_from_IN3 = np.array([synaptic_weights_toN1[2],synaptic_weights_toN2[2],synaptic_weights_toN3[2]])
synaptic_weights_from_IN4 = np.array([synaptic_weights_toN1[3],synaptic_weights_toN2[3],synaptic_weights_toN3[3]])


#print ("all synaptic and bis things: ")
#print(synaptic_weights_toN1)
#print(synaptic_weights_toN2)
#print(synaptic_weights_toN3)
#print(synaptic_weights_toOut)
#print(biases)

while True:
    # inputs = chooseInputs()
    # training_outputs = calculateY(inputs[0], inputs[1], inputs[2], inputs[3]) / y_Box # x1, y1, x2, y2
    
    inputs = chooseInputs()
    training_outputs = calculateY(inputs[0], inputs[1], inputs[2], inputs[3]) / y_Box # x1, y1, x2, y2

    #1 OK
    output_N1 = sigmoid(np.dot(inputs, synaptic_weights_toN1) )[0]
    output_N2 = sigmoid(np.dot(inputs, synaptic_weights_toN2) )[0]
    output_N3 = sigmoid(np.dot(inputs, synaptic_weights_toN3) )[0]
    
    #2
    N_Layer = np.array([output_N1, output_N2, output_N3])
    output = sigmoid(np.dot(synaptic_weights_toOut.T, N_Layer))
    
    #weight and biases modifacation------------------------------      
    #3
    o_error = training_outputs - output
    o_delta = o_error * sigmoid_derivative(output)*factor
    
    N1_error = synaptic_weights_toOut[0]*o_delta
    N1_delta = N1_error * sigmoid_derivative(output_N1)*factor
    N2_error = synaptic_weights_toOut[1]*o_delta
    N2_delta = N2_error * sigmoid_derivative(output_N2)*factor
    N3_error = synaptic_weights_toOut[2]*o_delta
    N3_delta = N3_error * sigmoid_derivative(output_N3)*factor
    
    #self.W1 += X.T.dot(self.z2_delta)

    synaptic_weights_toN1 += np.dot(N_Layer[0],  N1_delta)
    synaptic_weights_toN2 += np.dot(N_Layer[1],  N2_delta)
    synaptic_weights_toN3 += np.dot(N_Layer[2],  N3_delta)
    
    #synaptic_weights_from_IN1 += np.dot(inputs[0].T, (2*error *sigmoid_derivative(output)))
    #synaptic_weights_from_IN2 += np.dot(inputs[1].T, (2*error *sigmoid_derivative(output)))
    #synaptic_weights_from_IN3 += np.dot(inputs[2].T, (2*error *sigmoid_derivative(output)))
    #synaptic_weights_from_IN4 += np.dot(inputs[3].T, (2*error *sigmoid_derivative(output)))
    
    #  self.W2 += self.z2.T.dot(self.o_delta)
    synaptic_weights_toOut[0] += np.dot(output,  o_delta)
    synaptic_weights_toOut[1] += np.dot(output , o_delta)
    synaptic_weights_toOut[2] += np.dot(output , o_delta)
    
    
    #synaptic_weights_toOut[0] += np.dot(N_Layer[0], (2*error *o_delta))
    #synaptic_weights_toOut[1] += np.dot(N_Layer[1], (2*error *o_delta))
    #synaptic_weights_toOut[2] += np.dot(N_Layer[2], (2*error *o_delta))


    if training_outputs-allowed_error<= output <=training_outputs+allowed_error:
        aantalgoed += 1
    else:
        aantalfout += 1
    
    if (aantalgoed+aantalfout) % 1000 == 0:
        perc = 100*aantalgoed/(aantalgoed+aantalfout)
        print("Hoe goed is het netwerk:",perc)
        print(synaptic_weights_toN1[2])
        aantalgoed=0
        aantalfout=0

        if perc > doelperc:                
            break # break the loop