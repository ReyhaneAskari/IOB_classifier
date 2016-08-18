import theano
import theano.tensor as T
import numpy as np
from blocks.bricks import Linear
from blocks.initialization import IsotropicGaussian, Constant


# Model Definition (Computational Graph :D )
input = T.tensor3("input") # shape = l * n * 300
linear = Linear(name="linear1", input_dim=300, output_dim=500)
linear.weights_init = IsotropicGaussian(0.01)
linear.biases_init = Constant(0)
linear.initialize()
output = linear.apply(input)

# Function Definition 
f = theano.function([input],[output])

# Feeding Function
l = 6
n = 100
INPUT = np.random.randn(l, n, 300)
OUTPUT = f(INPUT)

print INPUT.shape
print OUTPUT[0].shape

import ipdb; ipdb.set_trace()

