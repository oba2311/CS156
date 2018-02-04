#156.14.1.py

import numpy as np
from neupy import algorithms

def draw_bin_image(image_matrix):
    for row in image_matrix.tolist():
        print('| ' + ' '.join(' *'[val] for val in row))

zero = np.matrix([
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0
])

one = np.matrix([
    0, 1, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0
])

two = np.matrix([
    1, 1, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0,
    0, 1, 1, 0, 0,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
])

draw_bin_image(zero.reshape((6, 5)))

data = np.concatenate([zero, one, two], axis=0)

dhnet = algorithms.DiscreteHopfieldNetwork(mode='sync')
dhnet.train(data)


half_zero = np.matrix([
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
])
draw_bin_image(half_zero.reshape((6, 5)))







half_two = np.matrix([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 1, 1, 0, 0,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
])
draw_bin_image(half_two.reshape((6, 5)))


result = dhnet.predict(half_zero)
draw_bin_image(result.reshape((6, 5)))







result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))


half_two = np.matrix([
    1, 1, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
])

result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))

from neupy import environment
environment.reproducible()

dhnet.mode = 'async'
dhnet.n_times = 400

result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))






result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))



from neupy import environment
environment.reproducible()

dhnet.mode = 'async'
dhnet.n_times = 400

result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))






result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))



# from neupy import plots
# import matplotlib.pyplot as plt

# plt.figure(figsize=(14, 12))
# plt.title("Hinton diagram")
# plots.hinton(dhnet.weight)
# plt.show()


#--------------------

# Reference from Valyo: Based on the tutorial here: http://codeaffectionate.blogspot.kr/2013/05/fun-with-hopfield-and-numpy.html
# and some comments and the noise-adding fucntions taken from here https://github.com/tomstafford/emerge/blob/master/lecture4.ipynb

from numpy import zeros, outer, diag_indices, array, vectorize, dot
from pylab import imshow, cm, show
import numpy as np
A = """
.XXX.
X...X
XXXXX
X...X
X...X
"""
 
Z = """
XXXXX
...X.
..X..
.X...
XXXXX
"""
def to_pattern(letter):
    return array([+1 if c=='X' else -1 for c in letter.replace('\n','')])


def display(pattern):
    imshow(pattern.reshape((5,5)),cmap=cm.binary, interpolation='nearest')
    show()
patterns = array([to_pattern(A), to_pattern(Z)])


display(patterns[0])

display(patterns[1])

def train(patterns):
    #This trains a network to remember the patterns it is given
    r, c = patterns.shape #take the patters and make them vectors
                        #There is a neuron for each pixel in the patterns
    
    W = zeros((c,c))  #there is a weight between each neuron in the network
    for p in patterns: # for each pattern
        W = W + outer(p,p) # change the weights to reflect the correlation between pixels
    W[diag_indices(c)] = 0 # neurons are not connected to themselves (ie the weight is 0)
    return W/r #send back the normalised weights


def recall(W, patterns, steps=5):
    sgn = vectorize(lambda x: -1 if x<0 else +1) # convert input pattern into a -1/+1 pattern
    for _ in range(steps): # for N iterations, 5 works well in our example
        patterns = sgn(dot(patterns,W)) #adjust the neuron activity to reflect the weights
    return patterns


def hopfield_energy(W, patterns):
    return array([-0.5*dot(dot(p.T,W),p) for p in patterns])

weights = train(patterns)


patterns_recall = recall(weights, patterns, 5)

display(patterns_recall[1])

def degrade(patterns,noise):
    # Adds noise to a pattern
    sgn=np.vectorize(lambda x: x*-1 if np.random.random()<noise else x)
    out=sgn(patterns)
    return out

def degrade_weights(W,noise):
    # Resets a proportion of the weights in the network
    sgn=vectorize(lambda x: 0 if random()<noise else x)
    return sgn(W)

patterns_damaged = degrade(patterns, 0.1)

display(patterns_damaged[1])

