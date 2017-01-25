"""
@copyright: Jennifer Ling
Adapted from Konstantin Buschmeier (code for amazon review sentiment prediction)

First approach to load a saved model and evaluate it on test data.
"""
CORPUS_PATH = "../corpora/"
TEST_PATH = "test/"

import os
from corpus import Corpus
import random
from features import createBagOfWordsDictionary, extractFeatures, createBagOfBigramsDictionary
from performance import showMeanPerformance, showPerformance
from sklearn.externals import joblib
from scipy.sparse import *

# load model.
files = os.listdir(".")
for file in files:
    if file.endswith(".pkl"):
        model = joblib.load(file)

# define some needed variables.       
class1 = "irony"
class2 = "sarcasm"
nr = "x"
testSetFilename = "test_set" + "_" + nr + ".txt"
randomSeed = 42
arff_path = "../arff/"

# load test corpus.
testSet = Corpus(class1, class2, testSetFilename, corpusPath=CORPUS_PATH+TEST_PATH)

testIDs = testSet.tweetIDs
                
random.seed(randomSeed)
random.shuffle(testIDs)

# define which features have to be extracted from test data (same as in loaded model).
featureConfigs = []
top10_iro_sarc_Config = [43,57,52,156,44,56,51,42,50,49]
featureConfigs.append(("all",True,False,False,True,top10_iro_sarc_Config, "features_top10Config"))

# feature extraction.
for mode, feat, regExp, new_stack, bigram, config, createARFF in featureConfigs:
    
    createARFF = createARFF + "_" + class1 + "_vs_" + class2
    
    binary_combination=False
    sentiment = False

    testFeatures, testFeatureVectors = extractFeatures(class1, class2, mode, arff_path, testIDs, testSet.tweets, config, feat, regExp, new_stack, binary_combination, sentiment, bigram, createARFF, bowDictionary, bigramDictionary)


# array of test data:
tsTargets = []
tsData = []

# sparse matrix of test data                   
rdim = len(testFeatureVectors.keys())
cdim = len(testFeatureVectors[testIDs[0]])
testData = lil_matrix((rdim, cdim))

testGold = testSet.goldStandard

testTargets = range(len(testGold))

ID_map_test = {}
j = 0

for ID, g in testGold.items():
                    
    ID_map_test[j] = ID
     
    # array           
    tsTargets.append(g)
    tsData.append(testFeatureVectors[ID])
    
    # matrix
    for i in range(len(testFeatureVectors[ID])):
        if testFeatureVectors[ID][i] != 0:
            testData[j, i] = testFeatureVectors[ID][i]
            testTargets[j] = g
    
    j += 1

# do prediction for test data.
classification = [model.predict(d)[0] for d in testData]

# TODO: Evaluate predicion.
     