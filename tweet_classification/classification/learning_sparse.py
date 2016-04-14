"""
@copyright: Jennifer Ling
Adapted from Konstantin Buschmeier (code for amazon review sentiment prediction)

Contains feature configuration and does machine learning (cv and test).
"""

#------ CONFIGURATIONS ------------------
CORPUS_PATH = "../corpora/"

SET_FILENAMES = ["training_set.txt",
                "test_set.txt",
                "validation_set.txt",
                "shuffled_set.txt"]
TEST_PATH = "test/"
TRAIN_PATH = "train/"


# ------------ imports --------------------
from corpus import Corpus
import random
from features import createBagOfWordsDictionary, extractFeatures, createBagOfBigramsDictionary
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import cross_validation
from sklearn.externals.six import StringIO
from performance import showMeanPerformance, showPerformance
from sklearn.externals import joblib
import pickle
from scipy.sparse import *
from timeit import timeit
import numpy as np
from time import *
# ------------ code -----------------------
def showDuration(doc_name, clf_name, duration):
    with open("info_" + doc_name + ".txt", "a") as info:
        info.write("{clf} needed: {duration}s\n\n".format(clf=clf_name, duration=duration))

def applyMachineLearning(class1, class2, randomSeed, arff_path, trainingSetFilename, testSetFilename=None, setPath=CORPUS_PATH):
    """
    Uses machine learning approach to classify sentences.
    """
    
    no_configs = [("irony", "figurative"), ("irony", "irony"), ("sarcasm", "sarcasm"), ("sarcasm", "figurative"), ("regular", "regular"), ("figurative", "irony"), ("figurative", "sarcasm"), ("figurative", "figurative")]
    if (class1, class2) in no_configs:
        print "ERROR! Please use allowed combination of classes!"
        exit()
    
    ## ---------- feature configurations --------------------------------        
    featureConfigs = []
    
    bowConfig = []
    bowBigramConfig = []
    
    allBinaryConfig = range(18,40)
    allConfig = range(60)
    allConfig.extend(range(156,157))
    allWithoutNumbers = range(18,60)
    allWithoutNorm = range(40)
    allWithoutStacks = range(60)
    
    normConfig = range(40,60)
    normConfig.extend(range(156,157))
    numbersConfig = range(18)
    stacksConfig = range(60,156)
    
    allWithoutBinary = range(18)
    allWithoutBinary.extend(range(40,60))
        
    allWithoutBoW = range(60)
    allWithoutBowBigram = range(60)
    allWithoutBigrams = range(60)
    
    # top10 - evaluated with weka chi^2-test
    if class1 == "irony" and class2 == "sarcasm":    
        top10Config = [43,57,52,156,44,56,51,42,50,49]
    elif class1 == "irony" and class2 == "regular":
        top10Config = [35,54,45,56,50,42,52,44]
    elif class1 == "sarcasm" and class2 == "regular":
        top10Config = [35,54,45,50,52,42,57,56]
    else:
        top10Config = []
    
    # ablation study 1
    sentimentConfig = [0,5,42,43,44,18,20,21,22,23,24,25,26,28,31,34,36,70,71,72,73,74,75,76,78,79,80,81,82,83,84,155] # number = [0,5], norm = [42,43,44], binaer = [18,20,21,22,23,24,25,26,28,31,34,36], stack = [70,71,72,73,74,75,76,78,79,80,81,82,83,84,155]
    
    subjConfig = [4,59,150,151,152,153,154] # number = [4], norm = [59], binaer = [], stack = [150,151,152,153,154]
    
    syntaxConfig =  [1,13,16,52,56,156,29,115,116,117,118,119,135,136,137,138,139] # number = [1,13,16], norm = [52,56,156], binaer = [29], stack = [115,116,117,118,119,135,136,137,138,139]
    
    posConfig = [6,7,11,17,45,50,51,57,85,86,87,88,89,105,106,107,108,109,110,111,112,113,114,140,141,142,143,144] # number = [6,7,11,17], norm = [45,50,51,57], binaer = [], stack = [85,86,87,88,89,105,106,107,108,109,110,111,112,113,114,140,141,142,143,144]
    
    emoticonConfig = [2,10,12,14,15,41,46,48,53,55,58,30,32,37,65,66,67,68,69,90,91,92,93,94,95,96,97,98,99,120,121,122,123,124,130,131,132,133,134,145,146,147,148,149]
    
    urlAndUserConfig = [39,35] # number = [], norm = [], binaer = [35,39], stack = []
    
    signalConfig = [3,8,9,40,41,42,43,44,45,46,47,48,49,54,19,27,33,38,60,61,62,63,64,100,101,102,103,104,125,126,127,128,129] # number = [3,8,9], norm = [40,41,42,43,44,45,46,47,48,49,54], binaer = [19,27,33,38], stack = [60,61,62,63,64,100,101,102,103,104,125,126,127,128,129]
    
    signalGroupConfig = signalConfig
    signalGroupConfig.extend(urlAndUserConfig)
    signalGroupConfig.extend(emoticonConfig)
    
    syntaxGroupConfig = syntaxConfig
    syntaxGroupConfig.extend(subjConfig)
    syntaxGroupConfig.extend(posConfig)
    
    syntaxAndSentiment = syntaxGroupConfig
    syntaxAndSentiment.extend(sentimentConfig)
    
    syntaxAndSignal = syntaxGroupConfig
    syntaxAndSignal.extend(signalGroupConfig)
    
    sentimentAndSignal = signalGroupConfig
    sentimentAndSignal.extend(sentimentConfig)
    
    
    # ablation study 2: ablation from ALL
    allWithoutSentiment = list(set(range(157)) - set(sentimentConfig)) 
    
    allWithoutPOS = list(set(range(157)) - set(posConfig))
    
    # signal + emoticons + url&User
    allWithoutSignal = list(set(range(157)) - set(signalConfig) - set(emoticonConfig) - set(urlAndUserConfig))

    # Syntax # Subjectivity
    allWithoutSyntax = list(set(range(157)) - set(syntaxConfig) - set(subjConfig) - set(posConfig))
        
    allWithoutTop10 = list(set(allConfig) - set(top10Config))
    

    # ablation study 2: ablation from BINARY
    binaryWithoutSentiment = list(set(allBinaryConfig) - set(sentimentConfig))
    
    binaryWithoutPOS = list(set(allBinaryConfig) - set(posConfig))
    
    # New Signal: Old Signal + Emoticon and RegExp and URL and User
    binaryWithoutSignal = list(set(allBinaryConfig) - set(signalConfig) - set(emoticonConfig) - set(urlAndUserConfig))
        
    # New Syntax: Old Syntax + Subj:

    binaryWithoutSyntax = list(set(allBinaryConfig) - set(syntaxConfig) - set(subjConfig) - set(posConfig))
       
    binaryWithoutTop10 = list(set(allBinaryConfig) - set(top10Config))
    
    # full configuration of feature list for feature extraction
                        #  mode,feat,regEx,stack_binning,bigram,sentiment,configuration
#     featureConfigs.append(("bow",True,True,True,False,False,bowConfig,"features_bowConfig"))
#     featureConfigs.append(("bow",True,True,True,True,False,bowBigramConfig,"features_bowBigramConfig"))
#     featureConfigs.append(("all",True,True,False,True,True,allBinaryConfig,"features_allBinaryConfig"))
#     featureConfigs.append(("all",True,True,True,True,True,allConfig,"features_allConfig"))
#     featureConfigs.append(("all",True,True,True,True,True,allWithoutNumbers,"features_allWithoutNumbers"))
#     featureConfigs.append(("all",True,True,True,True,True,allWithoutNorm,"features_allWithoutNorm"))
#     featureConfigs.append(("all",True,True,False,True,True,allWithoutStacks,"features_allWithoutStacks"))
#     featureConfigs.append(("all",True,False,True,True,True,allWithoutBinary,"features_allWithoutBinary"))
#     featureConfigs.append(("specific",True,True,True,True,True,allWithoutBoW,"features_allWithoutBoW"))
    featureConfigs.append(("specific",True,True,True,False,True,allWithoutBowBigram,"features_allWithoutBowBigram"))
#     featureConfigs.append(("all",True,True,True,False,True,allWithoutBigrams,"features_allWithoutBigrams"))

#     featureConfigs.append(("all",True,False,False,True,False,numbersConfig, "features_numbersConfig"))
#     featureConfigs.append(("all",True,False,False,True,False,normConfig, "features_normConfig"))
#     featureConfigs.append(("all",True,False,False,True,False,stacksConfig, "features_stacksConfig"))

#     featureConfigs.append(("all",True,True,False,True,True,allWithoutEmoticons,"features_allWithoutEmoticons"))
#     featureConfigs.append(("all",True,True,False,True,True,allWithoutSubj,"features_allWithoutSubj"))
#     featureConfigs.append(("all",True,False,False,True,True,allWithoutRegExpAndURLandUser,"features_allWithoutRegExpAndURLandUser"))

#     featureConfigs.append(("specific",True,True,False,True,True,allBinaryConfig,"features_binaryWithoutBoW"))
#     featureConfigs.append(("specific",True,True,False,False,True,allBinaryConfig,"features_binaryWithoutBoWBi"))
#     featureConfigs.append(("specific",True,True,False,False,True,allBinaryConfig,"features_binaryWithoutBi"))
#     
#     featureConfigs.append(("all",True,True,False,True,False,binaryWithoutSentiment,"features_binaryWithoutSentiment"))
#     featureConfigs.append(("all",True,True,False,True,True,binaryWithoutPOS,"features_binaryWithoutPOS"))
#     featureConfigs.append(("all",True,False,False,True,True,binaryWithoutSignal,"features_binaryWithoutSignal"))
#     featureConfigs.append(("all",True,True,False,True,True,binaryWithoutSyntax,"features_binaryWithoutSyntax"))
#     featureConfigs.append(("all",True,True,False,True,False,binaryWithoutWeka,"features_binaryWithoutWeka"))
#     featureConfigs.append(("all",True,True,False,True,True,binaryWithoutTop10,"features_binaryWithoutTop10"))

#     featureConfigs.append(("all",True,True,False,True,False,allWithoutSentiment,"features_allWithoutSentiment"))
#     featureConfigs.append(("all",True,True,False,True,True,allWithoutPOS,"features_allWithoutPOS"))
#     featureConfigs.append(("all",True,False,False,True,True,allWithoutSignal,"features_allWithoutSignal"))
#     featureConfigs.append(("all",True,True,False,True,True,allWithoutSyntax,"features_allWithoutSyntax"))
#     featureConfigs.append(("all",True,True,False,True,False,allWithoutWeka,"features_allWithoutWeka"))
#     featureConfigs.append(("all",True,True,False,True,True,allWithoutTop10,"features_allWithoutTop10"))

    # feature categories:
#     featureConfigs.append(("all",True,False,False,True,True,sentimentConfig, "features_sentimentConfig"))
#     featureConfigs.append(("all",True,False,False,True,False,posConfig, "features_posConfig"))
#     featureConfigs.append(("all",True,True,False,True,False,signalGroupConfig, "features_signalGroupConfig"))
#     featureConfigs.append(("all",True,False,False,True,False,syntaxGroupConfig, "features_syntaxGroupConfig"))
#      
#     featureConfigs.append(("specific",True,False,False,False,True,sentimentConfig, "features_sentimentConfig_specific"))
#     featureConfigs.append(("specific",True,False,False,False,False,posConfig, "features_posConfig_specific"))
#     featureConfigs.append(("specific",True,True,False,False,False,signalGroupConfig, "features_signalGroupConfig_specific"))
#     featureConfigs.append(("specific",True,False,False,False,False,syntaxGroupConfig, "features_syntaxGroupConfig_specific"))

    # Combinations:
#     featureConfigs.append(("specific",True,True,False,False,True,sentimentAndSignal,"features_sentimentAndSignal"))
#     featureConfigs.append(("specific",True,False,False,False,True,syntaxAndSentiment,"features_syntaxAndSentiment"))
#     featureConfigs.append(("specific",True,True,False,False,False,syntaxAndSignal,"features_syntaxAndSignal"))

#     featureConfigs.append(("all",True,False,False,True,False,top10Config, "features_top10Config"))
   
    print str(len(featureConfigs)) + " different configurations of features"   
    
    # create file which contains status reports.
    with open("info.txt", "a") as info:
        info.write("Start" + "\n")
        print "Start"
        # TODO: Add condition to create corpus, if no file exists.
        info.write("Training the classifiers using the set at '{path}{file}'".format(
                                                        path=setPath,
                                                        file=trainingSetFilename) + "\n")
        print("Training the classifiers using the set at '{path}{file}'".format(
                                                        path=setPath,
                                                        file=trainingSetFilename))
        
        
        lt = localtime()
        info.write("Begin loading Corpus " + class1 + " vs " + class2 + " - " + str(lt[3]) + "h:" + str(lt[4]) + "m:" + str(lt[5]) + "s Uhr am " + str(lt[2]) + "." + str(lt[1]) + "." + str(lt[0]) + "\n")
        print("Begin loading Corpus " + class1 + " vs " + class2 + " - " + str(lt[3]) + "h:" + str(lt[4]) + "m:" + str(lt[5]) + "s Uhr am " + str(lt[2]) + "." + str(lt[1]) + "." + str(lt[0]))
        
        # load training corpus.
        trainingSet = Corpus(class1, class2, trainingSetFilename, corpusPath=CORPUS_PATH+TRAIN_PATH)
        
        
        # Get the ids - which are ordered class1, class2 and shuffle them.
        trainingIDs = trainingSet.tweetIDs          
        random.seed(randomSeed)
        random.shuffle(trainingIDs)
        
        # load test corpus if filename is given; not needed for cross validation.
        if not testSetFilename == None:
            testSet = Corpus(class1, class2, testSetFilename, corpusPath=CORPUS_PATH+TEST_PATH)
            tweets = dict(trainingSet.tweets.items() + testSet.tweets.items())

            mode_list = []
            bigram_list = []
    
            # only create dict for bag-of-words and bigrams if really necessary!
            for mode, feat, regExp, stack_binning, bigram, sentiment, config, createARFF in featureConfigs:
                mode_list.append(mode)
                bigram_list.append(bigram)
    
            if "all" in mode_list:
                bowDictionary = createBagOfWordsDictionary(tweets)
                print "loaded bow"
            elif "bow" in mode_list:
                bowDictionary = createBagOfWordsDictionary(tweets)
                print "loaded bow"
            else:
                bowDictionary = {}
                print "bow not necessary"
    
            if True in bigram_list:
                bigramDictionary = createBagOfBigramsDictionary(tweets)
                print "loaded bigrams"
            else:
                bigramDictionary = {}
                print "bigrams not necessary"

        else:
            bowDictionary = None
            bigramDictionary = None
        
        lt = localtime()
        info.write("Corpus loaded -" + str(lt[3]) + "h:" + str(lt[4]) + "m:" + str(lt[5]) + "s Uhr am " + str(lt[2]) + "." + str(lt[1]) + "." + str(lt[0])+ "\n")
        print("Corpus loaded -" + str(lt[3]) + "h:" + str(lt[4]) + "m:" + str(lt[5]) + "s Uhr am " + str(lt[2]) + "." + str(lt[1]) + "." + str(lt[0]))

        info.write("Extracting features with different configurations \n")
        print("Extracting features with different configurations")
        

    
    t = 0
    
    # feature extraction using above feature configurations.
    for mode, feat, regExp, stack_binning, bigram, sentiment, config, createARFF in featureConfigs:
        
        trainFeatures = []
        trainFeatureVectors = {}
        testFeatures = []
        testFeatureVectors = {}
        
        t += 1
        
        config_name = createARFF + "_" + class1 + "_vs_" + class2
        # if empty string no arff file will be generated
        # else set createARFF_file = createARFF
        createARFF_file = createARFF
        
        with open("info_" + config_name + ".txt", "a") as info:
            print "\n" + str(t) + "th configuration\n-----------------------------------------\n"
            info.write("\n" + str(t) + "th configuration\n-----------------------------------------\n")
        
        # optional: if true, then all binary combinations of all 
        # features are added to feature list.        
        binary_combination=False       
        
        # feature extraction.    
        trainFeatures, trainFeatureVectors = extractFeatures(class1, class2, mode, arff_path, trainingIDs, trainingSet.tweets, config, feat, regExp, stack_binning, binary_combination, sentiment, bigram, createARFF_file, bowDictionary, bigramDictionary)
                
        # array of train data - is not necessary; just used for safeguard.
        tTargets = []
        tData = []
        
        # sparse matrix of train data:
        ID_map_train = {}
        rdim = len(trainFeatureVectors.keys())
        cdim = len(trainFeatureVectors[trainingIDs[0]])
        
        # create sparse matrix with rdim x cdim
        trainData = lil_matrix((rdim, cdim))
                    
        trainGold = trainingSet.goldStandard
        
        trainTargets = range(len(trainGold))
        j = 0
        for ID, g in trainGold.items():
            
            ID_map_train[j] = ID
            
            # array part.      
            tTargets.append(g)
            tData.append(trainFeatureVectors[ID])
            
            # matrix will be filled.
            for i in range(len(trainFeatureVectors[ID])):
                if trainFeatureVectors[ID][i] != 0:
                    trainData[j, i] = trainFeatureVectors[ID][i]
                    trainTargets[j] = g
            
            j += 1
        
        trainFeatureVectors = {}
        trainGold = {} 
                
        classifiers = [(DecisionTreeClassifier(), "Decision_Tree"), (SVC(kernel="linear"), "Linear_SVC"), (SVC(), "SVC"), (LinearSVC(), "LinearSVC"), (LogisticRegression(), "logRegression")]
        
        # classifiers which need matrix
        matrixClassifier = ["Linear_SVC", "SVC", "LinearSVC", "logRegression"]
        
        # Cross validation
        if testSetFilename == None:
            for c, name in classifiers:
                if name in matrixClassifier:
                    if isspmatrix(trainData):
                        duration = timeit(lambda: applyCrossValidation(class1, class2, createARFF, ID_map_train, c, name, trainData, trainTargets, 10), number=1)        
                        showDuration(createARFF, name, duration)

                else:
                    duration = timeit(lambda: applyCrossValidation(class1, class2, createARFF, ID_map_train, c, name, tData, tTargets, 10), number=1)                
                    showDuration(createARFF, name, duration)
                    
        # use test data for evaluation.
        else:
            with open("info.txt", "a") as info:   
                info.write("Testing the classifiers using the set at '{path}{file}'".format(
                                                            path=CORPUS_PATH,
                                                            file=testSetFilename) + "\n")
                print("Testing the classifiers using the set at '{path}{file}'".format(
                                                            path=CORPUS_PATH,
                                                            file=testSetFilename))
        
                info.write("Extracting features... \n")
                testIDs = testSet.tweetIDs
                
                random.seed(randomSeed)
                random.shuffle(testIDs)
                
                # feature extraction for test data.
                testFeatures, testFeatureVectors = extractFeatures(class1, class2, mode, arff_path, testIDs, testSet.tweets, config, feat, regExp, stack_binning, binary_combination, sentiment, bigram, createARFF_file, bowDictionary, bigramDictionary)
                                
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
                
                testFeatureVectors ={}
                testGold = {}
         
                for c, name in classifiers:
                    if name in matrixClassifier:
                        duration = timeit(lambda: applyClassifier(class1, class2, createARFF, ID_map_test, c, name, trainData, trainTargets, testData, testTargets), number=1)
                        showDuration(createARFF, name, duration)
                        
                    else:
                        duration = timeit(lambda: applyClassifier(class1, class2, createARFF, ID_map_test, c, name, tData, tTargets, tsData, tsTargets), number=1)                   
                        showDuration(createARFF, name, duration)
         
def applyCrossValidation(class1, class2, doc_name, id_map, classifier, clf_name, data, targets, k=10):
    """
    Uses k fold cross validation to test classifiers.
    """  
    with open("info_" + doc_name + ".txt", "a") as info:
            info.write("\nUsing {k} fold cross validation with {c}...".format(c=classifier, k=k) + "\n")                                                   
            info.write("\nMispredicted Tweets: ({k}-fold cross validation)\n".format(k=k))
     
    matrix = False

    goldStandards = []
    classifications = []
    cv_map_train = {}
    cv_map_validation = {}
    
    if isspmatrix(data):
        data = data.toarray()
        matrix = True
        
    n = 0
      
    for train, validation in cross_validation.KFold(len(targets), k):
        if matrix:
            rdim = len(train)
            cdim = len(data[0])
            trainData = lil_matrix((rdim, cdim))
            for i in range(len(train)):
                cv_map_train[i] = train[i] 
                for j in range(len(data[i])):
                    if data[train[i]][j] != 0:
                        trainData[i,j] = data[train[i]][j]

            rdim = len(validation)
            cdim = len(data[0])
            validationData = lil_matrix((rdim, cdim))
            for i in range(len(validation)):
                cv_map_validation[i] = validation[i]
                for j in range(len(data[i])):
                    if data[validation[i]][j] != 0:
                        validationData[i,j] = data[validation[i]][j]
        
        else:
            trainData = [d for i, d in enumerate(data) if i in train]
            validationData = [d for i, d in enumerate(data) if i in validation]
      
        trainTargets = [d for i, d in enumerate(targets) if i in train]            
        validationTargets = [d for i, d in enumerate(targets) if i in validation]
         
        model = classifier.fit(trainData, trainTargets)
        trainData = []
        trainTargets = []

        # TODO: possible feature:
        # import pydot
        # -> draw decision tree for visualisation.
#         try:
#             if clf_name == "DecisionTree":
#       
#                 dot_data = StringIO() 
#                 export_graphviz(model, out_file=dot_data) 
#                 graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#                 graph.write_pdf("DT" + str(k) + ".pdf")
#         except:
#             with open("info_" + doc_name + ".txt", "a") as info:
#                 info.write("No dot file \n")
#             print "No dot file"
          
        classification = [model.predict(d)[0] for d in validationData]
        validationData = []
        
        y_pred = np.asarray(classification)           
        y = np.asarray(validationTargets)
        
        classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
        
        with open("info_" + doc_name + ".txt", "a") as info:
                info.write("classifier_rate for " + clf_name + ": " + str(classif_rate) + "\n")
                print("classif_rate for %s : %f " % (clf_name, classif_rate))
    
        goldStandards.append(validationTargets)
        validationTargets = []
        classifications.append(classification)
        
        n += 1
        
        
    showMeanPerformance(doc_name, clf_name, goldStandards, classifications, [class1, class2])
    goldStandards = []
    classifications = []

def applyClassifier(class1, class2, doc_name, id_map, classifier, name, trainData, trainTargets, testData, testTargets):
    """Train and classify using a Support Vector Machine."""
    
    # feed model with data from feature extraction.
    model = classifier.fit(trainData, trainTargets)
    
    trainData = []
    trainTargets = []
    
    # save created model for re-use.
    joblib.dump(model, "model_" + doc_name + '.pkl')
    
    classification = [model.predict(d)[0] for d in testData]
    testData = []
    
    y_pred = np.asarray(classification)          
    y = np.asarray(testTargets)
    classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
    
    # get IDs of 'problematic' tweets in additional file for further analysis.
    with open("info_" + doc_name + ".txt", "a") as info, open("mispredicted_ids_" + doc_name + ".txt", "w") as mis, open("predicted_as_" + class1 + "_" + doc_name + ".txt", "w") as correct, open("predicted_as_" + class2 + "_" + doc_name + ".txt", "w") as wrong:
        info.write("\nUsing {0}".format(classifier) + "\n")
        print("\nUsing {0}".format(classifier))
        info.write("classifier_rate for " + name + ": " + str(classif_rate) + "\n")
        print("classif_rate for %s : %f " % (name, classif_rate))
        
        showPerformance(doc_name, name, testTargets, classification, [class1, class2]) 
        
        
        for i in range(len(classification)):
            # Get IDs of the mispredicted tweets:
            if classification[i] != testTargets[i]:
                mis.write(str(id_map[i]) + "\n")
            
            # Get all tweets predicted as class 1 - does not matter if prediction is correct
            if classification[i] == 1:
                correct.write(str(id_map[i]) + "\n")
                
            # Get all tweets predicted as class 2
            else:
                wrong.write(str(id_map[i]) + "\n")
        
        testTargets = []
        id_map = {}
        classification = []

    



