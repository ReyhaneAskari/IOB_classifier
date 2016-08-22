import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import Linear, Tanh, Softmax
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from load_DB import load_data, get_vectors
import matplotlib.pyplot as plt


learning_rate = 1
num_epochs = 300
batchSize = 100


def sgd(cost, params):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * learning_rate])
    return updates


# Computational Graph
input = T.tensor3('input')
mask = T.fmatrix('mask')
target = T.tensor3('target')
linear1 = Linear(name='linear1', input_dim=300, output_dim=128)
recurrent = SimpleRecurrent(name='recurrent', activation=Tanh(), dim=128)
linear2 = Linear(name='linear2', input_dim=128, output_dim=9)
softmax = Softmax()
bricks = [linear1, recurrent, linear2]
for brick in bricks:
    brick.weights_init = IsotropicGaussian(0.01)
    brick.biases_init = Constant(0)
    brick.initialize()

linear1_output = linear1.apply(input)
recurrent_output = recurrent.apply(linear1_output, mask=mask)
linear2_output = linear2.apply(recurrent_output)
shape = linear2_output.shape  # 100 * 29*9
output = softmax.apply(linear2_output.reshape((-1, 9))).reshape(shape) # hameye dimension ha be gheyr az yeki k oon 9 hast.


# Cost and Functions
cost = T.nnet.categorical_crossentropy(output, target)  # 100 x 29
cost = cost * mask
cost = cost.mean()


params = Model(cost).parameters
updates = sgd(cost, params)
f_train = theano.function(
    inputs=[input, mask, target],
    outputs=cost,
    updates=updates,
    allow_input_downcast=True)
f_valid = theano.function(
    inputs=[input, mask, target],
    outputs=cost,
    allow_input_downcast=True)
f_predict = theano.function(
    inputs=[input,mask],
    outputs=output,
    allow_input_downcast=True) #32 va 64 (GPU & CPU)            

# Training
import os  
INPUTS, MASKS, TARGETS, word_2_vec, index_to_target = load_data()
if not os.path.isfile('./RNN_best_params.npy'):
    print "hmmm... it seems like you are running the script for the first time, let me train it first :)"
    params_per_epoch = []    
    num_exmaples = INPUTS.shape[0]
    num_batches = num_exmaples / batchSize
    all_mean_costs_train = []
    all_mean_costs_valid = []
    for i in range(num_epochs):
        Train_COSTS = []
        Valid_COSTS = []
        for j in range(num_batches):
            idx = j * batchSize
            INPUT = INPUTS[idx: idx + batchSize]
            MASK = MASKS[idx: idx + batchSize]
            TARGET = TARGETS[idx: idx + batchSize]
            # Data in training set
            if j < num_batches * 0.8:
                COST = f_train(INPUT, MASK, TARGET)
                Train_COSTS.append(COST)
            # Data in validation set
            else:
                COST = f_valid(INPUT, MASK, TARGET)
                Valid_COSTS.append(COST)
        print 'Mean Train Cost : ' + str(np.mean(Train_COSTS)) + ', Mean Valid Cost : ' + str(np.mean(Valid_COSTS))
        all_mean_costs_train.append(np.mean(Train_COSTS))
        all_mean_costs_valid.append(np.mean(Valid_COSTS))    
        # np.save('params_epoch_' + str(i), [p.get_value() for p in params])
        params_per_epoch.append([p.get_value() for p in params])

    min_valid_cost_index = np.argmin(all_mean_costs_valid)
    np.save("RNN_best_params", params_per_epoch[min_valid_cost_index])
else:
    RNN_best_params = np.load("RNN_best_params.npy")
    for saved_param, param in zip(RNN_best_params,params):
        param.set_value(saved_param)
    while True: 
        sample_sentence  = raw_input('Type a query (type "exit" to exit): \n')
        if sample_sentence == "exit":
            break
        sample_input =[sample_sentence.strip().lower().split()]
        sentence_vec = get_vectors(sample_input, word_2_vec)[0]
        MASK = np.ones(sentence_vec.shape[:2])
        OUTPUT = f_predict(sentence_vec, MASK)
        predicted_indexes = np.argmax(OUTPUT, axis=2)[0]
        predicted_targets = [index_to_target[index] for index in predicted_indexes]
        print predicted_targets
    # import ipdb; ipdb.set_trace()
