import numpy as np
import pycrfsuite
from load_DB import load_data

INPUTS, MASKS, TARGETS = load_data()

trainer = pycrfsuite.Trainer(verbose=True, algorithm='l2sgd')

for i in range(800):
    # trainer expects string and not int!
    T = [str(i) for i in np.argmax(TARGETS[i], axis=1)]
    length = int(np.sum(MASKS[i]))
    trainer.append(INPUTS[i][:length], T[:length])

trainer.set_params({
    # 'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 5000,  # stop earlier
    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

print trainer.params()
trainer.train('crf_model')
print len(trainer.logparser.iterations), trainer.logparser.iterations[-1]

tagger = pycrfsuite.Tagger()
tagger.open('crf_model')

for i in np.arange(800, 1000):
    length = int(np.sum(MASKS[i]))
    INPUT = INPUTS[i][:length]
    TARGET = np.argmax(TARGETS[i], axis=1)[:length]
    print("Predicted:", ' '.join(tagger.tag(INPUT)))
    print("Correct:  ", ' '.join([str(i) for i in TARGET]))
