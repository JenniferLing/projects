## DeepSenseSpotting

- *pretrained_word_embeddings.py*: Given an input corpus and an output path, the word embeddings are created. word2vec CBOW, word2vec Skip-Gram and Glove vectors are trained and saved to the output directory. The number of dimensions, training epochs and the window size can be adapted.

- *mlp.py*: Performs cross validation on a dataset and the features created by sensespotting_2. The architecture is an MLP with several hidden layers (can be adapted)

- *context_nn.py*: Needs the word embeddings as input and also perform CV on the dataset and features created by sensespotting_2. The architecture consists of an embedding layer followed by three bidirectional LSTM layers. After dropout, the output is concatenated with the SenseSpotting features and five fully-connected layers are applied. The final prediction is computed with sigmoid.
