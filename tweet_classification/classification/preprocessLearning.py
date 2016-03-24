# -*- coding: utf-8 -*-

"""
@copyright: Jennifer Ling
Adapted from Konstantin Buschmeier (code for amazon review sentiment prediction)
"""

# -------------- configurations ------------------------------------
TRAINING_SET_SIZE = 90
VALIDATION_SET_SIZE = 0
TEST_SET_SIZE = 10
RANDOM_SEED = 1

CORPUS_PATH = "../corpora/"

SET_FILENAMES = ["training_set",
                "test_set", 
                "validation_set",
                "shuffled_set"]

TEST_PATH = "test/"
TRAIN_PATH = "train/"

# -------------- imports -------------------------------------------
import codecs
import random
from corpus import readAllIDs

# ------------- code -----------------------------------------------

def divideData(data, setSizes):
    """
    Returns a list of list from the data accordingly to the given percentages.
    """
    assert not sum(setSizes) > 100
    random.seed(RANDOM_SEED)
    random.shuffle(data)

    result = []
    numberOfSets = len(setSizes)
    offset = 0
    for setNumber in range(numberOfSets):
        if setNumber == numberOfSets-1:
            result.append(data[offset:])
        else:
            end = offset + len(data)*setSizes[setNumber]/100
            result.append(data[offset:end])
            offset = end
    return result

def setToString(IDLabelSet):
    """Returns a list of string representations for every ID, label pair."""
    return ["{0}:\t{1}".format(label.upper(), ID) for ID, label in IDLabelSet]

def createIDLabelSet(data, label):
    """Returns a list of ID, label pairs."""
    return [(ID, label) for ID in data]

def saveSet(set, filename):
    """Saves the given set in a file."""
    with codecs.open(filename, 'w', encoding='utf-8') as setFile:
        setFile.writelines("\n".join(set))

def createSets(class1, class2, nr, setSizes=[TRAINING_SET_SIZE, TEST_SET_SIZE, VALIDATION_SET_SIZE]):
    """Reads IDs, creates and saves randomly shuffled subsets of these."""
    tweetClass1IDs, tweetClass2IDs = readAllIDs(class1, class2)
    
    tweetIDs = setToString(createIDLabelSet(tweetClass1IDs, str(class1))) 
    tweetIDs += setToString(createIDLabelSet(tweetClass2IDs, str(class2)))

    sets = divideData(tweetIDs, setSizes)

    for i in range(len(sets)):
        saveSet(sets[i], CORPUS_PATH + SET_FILENAMES[i] + "_" + nr + ".txt")


def createTrainingSet(class1, class2, nr, path= CORPUS_PATH, randomSeed=RANDOM_SEED):
    """Reads IDs and saves a shuffled version of it."""
    
    tweetClass1IDs, tweetClass2IDs = readAllIDs(class1, class2)
    
    tweetIDs = setToString(createIDLabelSet(tweetClass1IDs, str(class1))) 
    tweetIDs += setToString(createIDLabelSet(tweetClass2IDs, str(class2)))
    
    random.seed(randomSeed)
    random.shuffle(tweetIDs)

    saveSet(tweetIDs, path + "training_set_" + nr + ".txt")

def createTestSet(class1, class2, nr, path= CORPUS_PATH, randomSeed=RANDOM_SEED):
    """Reads IDs and saves a shuffled version of it."""
    tweetClass1IDs, tweetClass2IDs = readAllIDs(class1, class2, path=CORPUS_PATH + "testIDs.txt")

    tweetIDs = setToString(createIDLabelSet(tweetClass1IDs, str(class1))) 
    tweetIDs += setToString(createIDLabelSet(tweetClass2IDs, str(class2)))
   
    random.seed(randomSeed)
    random.shuffle(tweetIDs)

    saveSet(tweetIDs, path + "test_set_" + nr + ".txt")