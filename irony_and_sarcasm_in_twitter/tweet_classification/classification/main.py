# -*- coding: utf-8 -*-

"""
@copyright: Jennifer Ling
Adapted from Konstantin Buschmeier (code for amazon review sentiment prediction)
"""

## ------------------------------- configurations -----------------------------
SET_FILENAMES = ["training_set",
                "test_set", 
                "validation_set",
                "shuffled_set"]
CORPUS_PATH = "../corpora/"
TRAIN_PATH = "train/"
TEST_PATH = "test/"
# ------------------------------ imports -------------------------------------
from timeit import timeit

from corpus import Corpus
from features import extractFeatures, showFeatureOccurrence
from learning_sparse import applyMachineLearning
from create_ID_file import createIDFile
from preprocessLearning import createTestSet, createTrainingSet
import os
import shutil
# ----------------------------- functions ------------------------------------

def createARFF(class1, class2, arff_path, corpus_path=CORPUS_PATH+TRAIN_PATH):
    """
    Define features by their number (see features: feature_names) and create ARFF file.
    """
    corpus = Corpus(class1, class2,corpusPath=corpus_path)
            
    allConfig = range(5)
    
    featureConfigs = []
                        #  mode,feat,regEx,new_stack,bigram,sentiment,configuration
    featureConfigs.append(("specific",True,False,False,False,False,allConfig,"features_all_specific" + "_" + class1 + "_vs_" + class2))
    
    # combine each feature with each feature, e.g. IF stopword = true AND negation = true THEN combi = true.
    binary_combination=False
    # = createARFF -> wird erstellt oder "" -> wird nicht erstellt
    
    for mode, feat, regExp, new_stack, bigram, sentiment, config, createARFF in featureConfigs:
        createARFF_file = createARFF        
        features, featureVectors = extractFeatures(class1, class2, mode, arff_path, corpus.class1IDs + corpus.class2IDs, corpus.tweets, config, feat, regExp, new_stack, binary_combination, sentiment, bigram, createARFF_file)


def runLearning(class1, class2, randomSeed, arff_path, train_modus="cross-validation"):
    """
    Determines which set will be used as the validation set, i.e. the set the
    classifiers will be tested against. Currently are the following values 
    available 
    * 'test' and
    * 'cross-validation'.
    """
    # number of differentiation of different runs.
    nr = "1"
    runSetMode(class1, class2, nr, randomSeed)
    
    if train_modus.lower() == "cross-validation":
        duration = timeit(lambda: applyMachineLearning(class1, class2, randomSeed, arff_path, "training_set" + "_" + nr + ".txt"), 
                        number=1)
    elif train_modus.lower() == "test":
        duration = timeit(lambda: applyMachineLearning(class1, class2, randomSeed, arff_path, "training_set" + "_" + nr + ".txt", "test_set" + "_" + nr + ".txt"),
                        number=1)
    showDuration(duration)

# ------- duration computation -----------------------------------------
def showDuration(duration):
    print "Elapsed time: {duration}s".format(duration=duration)


# ---------- set generation -------------------------------------------
def runSetMode(class1, class2, nr, randomSeed):
    duration = timeit(lambda: generateSets(class1, class2, nr, randomSeed), number=1), 
    showDuration(duration)

def generateSets(class1, class2, nr, randomSeed):
    """
    Generate a shuffled set for cross-validation and training and test sets.
    """
    createTrainingSet(class1, class2, nr, randomSeed=randomSeed)
    createTestSet(class1, class2, nr, randomSeed=randomSeed)

def removeOldFiles(myPath, prefix):
    """ Old files which will be overwritten when running main, will be moved to an folder called old """
    files = os.listdir(myPath)
    
    if not os.path.exists(myPath + "old/"):
        os.makedirs(myPath + "old/")
    else:
        old_files = os.listdir(myPath + "old/")
        for file_element in old_files:
            if os.path.exists(myPath + file_element):
                os.remove(myPath + "old/" + file_element)
            
    for file_name in files:
        for p in prefix:
            if file_name.startswith(p):
                shutil.move(myPath + file_name, myPath + "old/")

def saveResults(myPath, class1, class2):
    """ create folder for all results and moves old files to temp folder called old (loss protection)"""
    
    files = os.listdir(myPath)
    
    if not os.path.exists(myPath + class1 + "_vs_" + class2 + "_results/"):
        os.makedirs(myPath + class1 + "_vs_" + class2 + "_results/")
    else:
        old_files = os.listdir(myPath + class1 + "_vs_" + class2 + "_results/")
        for file_element in old_files:
            if os.path.exists(myPath + file_element):
                os.remove(myPath + class1 + "_vs_" + class2 + "_results/" + file_element)
    
    for file_name in files:
        if file_name.endswith(".txt"):
            shutil.move(myPath + file_name, myPath + class1 + "_vs_" + class2 + "_results/")

def saveModel(path, speicher, class1, class2):
    """ save all model files in same folder """
    files = os.listdir(path)
    
    if not os.path.exists(speicher + class1 + "_vs_" + class2 + "/"):
        os.makedirs(speicher + class1 + "_vs_" + class2 + "/")
    
    for file in files:
        if file.startswith("model_") and (file.endswith(".npy") or file.endswith(".pkl")):
            if os.path.exists(speicher + class1 + "_vs_" + class2 + "/" + file):
                os.remove(speicher + class1 + "_vs_" + class2 + "/" + file)
            shutil.move(path + file, speicher + class1 + "_vs_" + class2 + "/")

def main(randomSeed=42):
    """ main function: controls the general application behaviour. Defines which processing is started. """   
    arff_path = "../arff/"
    
    if not os.path.exists(arff_path):
        os.makedirs(arff_path)
    
    removeOldFiles(arff_path, ["features_"])
    removeOldFiles("./", ["info_", "mispredicted_", "predicted_as_"])
   
    classes = [("irony", "sarcasm"), ("irony", "regular"), ("sarcasm", "regular"),("figurative", "regular")]
  
    for class1, class2 in classes:
     
        createARFF(class1, class2, arff_path,corpus_path=CORPUS_PATH+TEST_PATH)
        
        #runLearning(class1, class2, randomSeed, arff_path, "test")
        #saveResults("./", class1, class2)
        #saveModel("./", "../models/", class1, class2)
        

     

if __name__ == '__main__':
    createIDFile()  
    main()