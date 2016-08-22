import numpy as np
import pycrfsuite
from load_DB import load_data, get_vectors

INPUTS, MASKS, TARGETS, word_2_vec, index_to_target = load_data()

trainer = pycrfsuite.Trainer(verbose=True, algorithm='l2sgd')

for i in range(1000):
    # trainer expects string and not int!
    T = [index_to_target[i] for i in np.argmax(TARGETS[i], axis=1)]
    length = int(np.sum(MASKS[i]))
    trainer.append(INPUTS[i][:length], T[:length])

trainer.set_params({
    # 'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 5000,  # stop earlier
    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

trainer.train('crf_model')
tagger = pycrfsuite.Tagger()
tagger.open('crf_model')
while True: 
    sample_sentence  = raw_input('Type a query (type "exit" to exit): \n')
    if sample_sentence == "exit":
        break
    sample_input =[sample_sentence.strip().lower().split()]
    sentence_vec = get_vectors(sample_input, word_2_vec)[0]          
    print tagger.tag(sentence_vec[0])

    
    
