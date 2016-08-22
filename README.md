# IOB_classifier

We compare two sequence models to build a classifier system that is capable of labeling IOB (Inside, Outside, Beginning) tags given an input sentence. We used Simple Recurrent Neural Networks (SRNs) and Conditional Random Fields (CRFs) as the sequence models.

## How to test

Run train.py to test the SRN model. The model needs to be trained first. After running it for the first time it will creat a file named RNN_best_params.npy. To test the model, run the same script twice and enter a sentence. The classifier will return the IOB tages.

Run train_crf.py to test the CRF model. Enter a sample sentence and the classifier will return the IOB tags. 
