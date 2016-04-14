# irony_and_sarcasm_in_twitter

Twitter crawler can be used to get tweets with a specific hashtag (such as #irony).
Twitter4J is used to connect with Twitter API.

Tweet classification can be used to classify ironic, sarcastic and non-figurative (regular) tweets in a two-class-labelling-process.
In corpora the tweets are sorted by their function and their label (e.g. train/irony.csv) and used ID files to identify the tweets.
Additionally, all tweets have to exist in tagged form (-> corpora/tagged_tweets) before classification. 
The tags are created with the TwitIE tagger using additional modifications for preprocessing and tokenizing (see tagger folder).
The resources folder contains lists of stopwords, lists of positive and negative words, etc. The files in this folder are used for feature extraction.

The classification will be started via classification/main.py. 
There train and test corpus can be changed and learning or only feature extraction (creating an ARFF file) can be started.
First, create_ID_file.py creates the necessary ID files (each line contains a tweet ID and its class).
classification/preprocessLearning.py creates the data sets which are loaded and preprocessed (tokenized with tokenizer.py) in corpus.py and tweet.py.
In classification/learning_sparse.py you define which features you want to use and if you want to save a model.
classification/features.py contains the implementation of the features and the feature extraction process. 
In feature_config.py you can add specific regular expressions for emoticons, acronyms, etc.
classification/performance.py evaluates the learning process and returns the results of the classification.



