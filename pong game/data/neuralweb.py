import numpy as np
from random import randrange

# grootte van het speelveld
x_Box = 800
y_Box = 600

training_size = 100 # geeft aan hoeveel training inputs hij heeft
hidden_size = 10 # geeft aan hoeveel neuronen de hidden layer heeft
training_cycles = 50 # geeft aan hoe vaak de training moet worden uitgevoerd

def chooseInputs():
    x1 = randrange(0,x_Box)
    y1 = randrange(0,y_Box)
    x2 = x1 + randrange(1,10)
    y2 = y1 + randrange(-10,10)

    return [x1,y1,x2,y2]

def calculateY(x1,y1,x2,y2,predictLeft=True):
    if predictLeft:
        x1 = x_Box - x1
        x2 = x_Box - x2
 
    # kleine driehoek ABC
    AB = x2-x1
    BC = y2-y1
    
    # grote driehoek ADE
    AD = x_Box - x1
    factor = AD / AB
    DE = BC * factor

    y_end = y1 + DE

    while not (0 <= y_end <= y_Box):
        #bounce
        if y_end < 0: 
            y_end = -y_end
        else:
            y_end = y_Box - y_end          
    
    return y_end


########

def makeTrainingSet():
    training_input_list = []
    training_output_list = []
    for i in range(training_size):
        training_input = chooseInputs()
        training_output = calculateY(training_input[0],training_input[1],training_input[2],training_input[3])
        training_input_list.append(training_input)
        training_output_list.append([training_output])
    
    training_inputs = np.array(training_input_list, dtype=float)
    training_outputs = np.array(training_output_list, dtype=float)
    
    # scale units
    training_inputs = training_inputs / x_Box
    training_outputs = training_outputs/ y_Box
    return (training_inputs, training_outputs)

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 4
    self.outputSize = 1
    self.hiddenSize = hidden_size

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (3x2) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  def forward(self, training_inputs):
    #forward propagation through our network
    self.z = np.dot(training_inputs, self.W1) # dot product of training_inputs (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o 

  def sigmoid(self, s):
    # activation function 
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, training_inputs, training_outputs, o):
    # backward propgate through the network
    self.o_error = training_outputs - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += training_inputs.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train (self, training_inputs, training_outputs, cycles = 1):
    for cycle in range(cycles):
        o = self.forward(training_inputs)
        self.backward(training_inputs, training_outputs, o)

# hoofdprogramma om te trainen
NN = Neural_Network()
training_inputs, training_outputs = makeTrainingSet()
NN.train(training_inputs, training_outputs, training_cycles)

print ("Resultaat:")
print ("Input: \n" + str(training_inputs) )
print ("Actual Output: \n" + str(training_outputs) )
print ("Predicted Output: \n" + str(NN.forward(training_inputs))) 
print ("Goed: \n", (int)((1-(np.mean(np.square(training_outputs - NN.forward(training_inputs)))))*100),"%") # mean sum squared loss

# beweeg batje naar...
def calculateY_NN(x1,y1,x2,y2, continuousLearning=False, networkPlaysLeft= False):

    if networkPlaysLeft:
        x1 = x_Box - x1
        x2 = x_Box - x2
        
    inputs = [[x1/x_Box,y1/y_Box,x2/x_Box,y2/y_Box]]
    np_inputs = np.array(inputs, dtype=float)
    outputs = NN.forward(np_inputs)
    output = outputs[0]
    outputvalue_neuron0 = output[0] * y_Box
    
    if continuousLearning:
        desiredY = calculateY(x1,y1,x2,y2)/y_Box
        desiredOutputs = np.array([[desiredY]], dtype=float)
        NN.train(np_inputs,desiredOutputs)
            
    return (int)(outputvalue_neuron0)
