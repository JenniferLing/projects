# Irony and Sarcasm in Twitter

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

# Publication

Jennifer Ling and Roman Klinger. An empirical, quantitative analysis of the differences between sarcasm and irony. In Harald Sack, Giuseppe Rizzo, Nadine Steinmetz, Dunja Mladenić, Sören Auer, and Christoph Lange, editors, The Semantic Web: ESWC 2016 Satellite Events, Heraklion, Crete, Greece, May 29 -- June 2, 2016, Revised Selected Papers, pages 203--216. Springer International Publishing, 2016. Best Paper. 

final article: http://link.springer.com/chapter/10.1007%2F978-3-319-47602-5_39

preprint: http://www.romanklinger.de/publications/ling2016.pdf

# BibTeX citation

@inproceedings{Ling2016,<br/>
&nbsp;&nbsp;author = {Ling, Jennifer and Klinger, Roman},<br/>
&nbsp;&nbsp;editor = {Sack, Harald and Rizzo, Giuseppe and Steinmetz,<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Nadine and Mladeni{\'{c}}, Dunja and Auer, S{\"o}ren <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and Lange, Christoph}, <br/>
&nbsp;&nbsp;title = {An Empirical, Quantitative Analysis of the <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Differences Between Sarcasm and Irony}, <br/>
&nbsp;&nbsp;booktitle = {The Semantic Web: ESWC 2016 Satellite Events, <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Heraklion, Crete, Greece, May 29 -- June 2, 2016, <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Revised Selected Papers}, <br/>
&nbsp;&nbsp;year = {2016}, <br/>
&nbsp;&nbsp;publisher = {Springer International Publishing}, <br/>
&nbsp;&nbsp;pages = {203--216}, <br/>
&nbsp;&nbsp;isbn = {978-3-319-47602-5}, <br/>
&nbsp;&nbsp;doi = {10.1007/978-3-319-47602-5_39}, <br/>
&nbsp;&nbsp;url = {http://dx.doi.org/10.1007/978-3-319-47602-5_39}<br/>
}
