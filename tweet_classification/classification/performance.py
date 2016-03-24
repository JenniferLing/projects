# -*- coding: utf-8 -*-

"""
@copyright: Jennifer Ling
Adapted from Konstantin Buschmeier (code for amazon review sentiment prediction)
"""

from __future__ import print_function
from numpy import mean, std
from math import sqrt

# ---- Functions to calculate performances ----
def calcPerformance(gold, prediction):
    """
    Calculates performance of the prediction.
    Gold and prediction can be lists or lists of lists.
    """
    assert(len(gold) == len(prediction))
    # If gold and prediction are lists of lists.
    if (all(isinstance(l, list) for l in gold) and
            all(isinstance(l, list) for l in prediction)):
        return [calcPerformance(g, p) 
                for g, p in zip(gold, prediction)]
    else:
        result = zip(gold, prediction)

        tp_class1 = sum([1 if g and p else 0 
                    for g, p in result])
        tn_class1 = sum([1 if not g and not p else 0 
                    for g, p in result])
        fp_class1 = sum([1 if not g and p else 0 
                    for g, p in result])
        fn_class1 = sum([1 if g and not p else 0 
                    for g, p in result])  
        
        tn_class2 = sum([1 if g and p else 0 
                    for g, p in result])
        tp_class2 = sum([1 if not g and not p else 0 
                    for g, p in result])
        fn_class2 = sum([1 if not g and p else 0
                    for g, p in result])
        fp_class2 = sum([1 if g and not p else 0
                    for g, p in result])
               
        precision_class1 = float(tp_class1)/(tp_class1 + fp_class1) if not (tp_class1 + fp_class1) == 0 else 1.0 
        recall_class1 = float(tp_class1)/(tp_class1 + fn_class1) if not (tp_class1 + fn_class1) == 0 else 1.0
        
        precision_class2 = float(tp_class2)/(tp_class2 + fp_class2) if not (tp_class2 + fp_class2) == 0 else 1.0 
        recall_class2 = float(tp_class2)/(tp_class2 + fn_class2) if not (tp_class2 + fn_class2) == 0 else 1.0
        
        accuracy = float(tp_class1 + tn_class1)/float(tp_class1 + fp_class1 + fn_class1 + tn_class1)
        precision = (precision_class1 + precision_class2)/2
        recall = (recall_class1 + recall_class2) / 2

        if precision_class1 is None or recall_class1 is None:
            fScore_class1 = None
        elif precision_class1 == 0 and recall_class1 == 0:
            fScore_class1 = 0.0
        else:
            fScore_class1 = 2*precision_class1*recall_class1/(precision_class1 + recall_class1)
        
        if precision_class2 is None or recall_class2 is None:
            fScore_class2 = None
        elif precision_class2 == 0 and recall_class2 == 0:
            fScore_class2 = 0.0
        else:
            fScore_class2 = 2*precision_class2*recall_class2/(precision_class2 + recall_class2)

        if precision is None or recall is None:
            fScore = None
        elif precision == 0 and recall == 0:
            fScore = 0.0
        else:
            fScore = (fScore_class2 + fScore_class1) / 2
        
        return {"truePositiveClass1": tp_class1, "trueNegativeClass1": tn_class1, 
                "falsePositiveClass1": fp_class1, "falseNegativeClass1": fn_class1, 
                "precisionClass1": precision_class1, "recallClass1": recall_class1,
                "accuracy": accuracy, "F-scoreClass1": fScore_class1, 
                "truePositiveClass2": tp_class2, "trueNegativeClass2": tn_class2, 
                "falsePositiveClass2": fp_class2, "falseNegativeClass2": fn_class2, 
                "precisionClass2": precision_class2, "recallClass2": recall_class2,
                "accuracy": accuracy, "F-scoreClass2": fScore_class2, 
                "F-score": fScore, "recall": recall, "precision": precision}

def calcMeanPerformance(doc_name, clf_name, goldStandards, prediction):
    """Calculates the mean performances."""
    assert(len(goldStandards) == len(prediction))

    measures = {"truePositiveClass1", "trueNegativeClass1", "falsePositiveClass1", 
                "falseNegativeClass1", "precisionClass1", "recallClass1","accuracy", "F-scoreClass1",
                "truePositiveClass2", "trueNegativeClass2", "falsePositiveClass2", 
                "falseNegativeClass2", "precisionClass2", "recallClass2", "F-scoreClass2",
                "F-score", "recall", "precision"}

    performances = calcPerformance(goldStandards, prediction)

    for number, each in enumerate(performances):
        with open("info_" + doc_name + ".txt", "a") as info:
            info.write("{number}. fold:\n".format(number=number) + str(each) + "\n")

    report = {}
    for m in measures:
        total = sum([p[m] if p[m] is not None else 0 for p in performances])
        count = sum([1 if p[m] is not None else 0 for p in performances])
        if not count == 0:
            report[m] = total/float(count)
        else:
            report[m] = 0

    return report

# ---- Functions to show performances ----
def showPerformance(doc_name, clf_name, gold, classification, label):
    """Shows the classification's performance."""
    
    performance = calcPerformance(gold, classification)
    with open("info_" + doc_name + ".txt", "a") as info:
        info.write("# Predictions: " + str(len(classification)) + "\n")
    
    showEvaluation(doc_name, clf_name, performance, label)
    showScores(doc_name, clf_name, performance, label)

def showMeanPerformance(doc_name, clf_name, golds, classifications, label):
    """Shows the classification's mean performance."""
    with open("info_" + doc_name + ".txt", "a") as info:
        info.write("\nBelow shows means: \n")
    
    performances = calcPerformance(golds, classifications)
    showEvaluation(doc_name, clf_name, calcMeanPerformance(doc_name, clf_name, golds, classifications), label)
    showMeanScores(doc_name, clf_name, performances, label)

def showEvaluation(doc_name, clf_name, performance, label):
    """Shows a the result of the classification."""
    
    # Table form of prediction/gold
    with open("info_" + doc_name + ".txt", "a") as info:
        info.write(("\n\n\t\t\t\tGold: {positive}\t\tGold: {negative}".format(positive=label[0], 
                                                            negative=label[1])) + "\n")
        info.write(("Predict: {positive}\t\t{tp} (tp)\t\t{fp} (fp)".format(
                                                positive=label[0],
                                                tp=performance["truePositiveClass1"],
                                                fp=performance["falsePositiveClass1"])) + "\n")
        info.write(("Predict: {negative}\t{fn} (fn)\t\t{tn} (tn)".format(
                                                negative=label[1],
                                                fn=performance["falseNegativeClass1"],
                                                tn=performance["trueNegativeClass1"])) + "\n\n")
        
        info.write(("\t\t\t\tGold: {positive}\tGold: {negative}".format(positive=label[1], 
                                                            negative=label[0])) + "\n")
        info.write(("Predict: {positive}\t{tp} (tp)\t\t{fp} (fp)".format(
                                                positive=label[1],
                                                tp=performance["truePositiveClass2"],
                                                fp=performance["falsePositiveClass2"])) + "\n")
        info.write(("Predict: {negative}\t\t{fn} (fn)\t\t{tn} (tn)\n".format(
                                                negative=label[0],
                                                fn=performance["falseNegativeClass2"],
                                                tn=performance["trueNegativeClass2"])) + "\n")      
        
def showMeanEvaluation(doc_name, clf_name, performances, label):
    """Shows a the result of the classification."""

    meanPerformance = calcMeanPerformance(doc_name, clf_name, performances)
    showEvaluation(doc_name, clf_name, meanPerformance, label)

def showScores(doc_name, clf_name, performance, labels):
    """Shows performance measurements precision, recall, accuracy, f-Score."""
    
    with open("info_" + doc_name + ".txt", "a") as info:
        for i in range(len(labels)):
            info.write(("Precision of {0}:\t{1}".format(labels[i], performance["precisionClass" + str(i+1)])) + "\n")
            info.write(("Recall of {0}:\t{1}".format(labels[i], performance["recallClass" + str(i+1)])) + "\n")
            info.write(("F-Score of {0}:\t{1}".format(labels[i], performance["F-scoreClass" + str(i+1)])) + "\n\n")
        
        info.write("Precision:\t{0}".format(performance["precision"]) + "\n")
        info.write("Recall:\t{0}".format(performance["recall"]) + "\n")
        info.write("F-Score:\t{0}".format(performance["F-score"]) + "\n\n")
            
        info.write("Accuracy:\t\t{0}".format(performance["accuracy"]) + "\n")

def showMeanScores(doc_name, clf_name, performances, label):
    """
    Shows mean performance measurements precision, recall, accuracy, f-Score.
    """
    
    precisionsClass1 = [p["precisionClass1"] for p in performances]
    recallsClass1 = [p["recallClass1"] for p in performances]
    fScoresClass1 = [p["F-scoreClass1"] for p in performances]
    
    precisionsClass2 = [p["precisionClass2"] for p in performances]
    recallsClass2 = [p["recallClass2"] for p in performances]
    fScoresClass2 = [p["F-scoreClass2"] for p in performances]
    
    precisions = [p["precision"] for p in performances]
    recalls = [p["recall"] for p in performances]
    fScores = [p["F-score"] for p in performances]
    
    accuracys = [p["accuracy"] for p in performances]
 
    cleanPrecisionsClass1 = [p["precisionClass1"] for p in performances if p["precisionClass1"] is not None]
    cleanRecallsClass1 = [p["recallClass1"] for p in performances if p["recallClass1"] is not None]
    cleanFScoresClass1 = [p["F-scoreClass1"] for p in performances if p["F-scoreClass1"] is not None]
    
    cleanPrecisionsClass2 = [p["precisionClass2"] for p in performances if p["precisionClass2"] is not None]
    cleanRecallsClass2 = [p["recallClass2"] for p in performances if p["recallClass2"] is not None]
    cleanFScoresClass2 = [p["F-scoreClass2"] for p in performances if p["F-scoreClass2"] is not None]
    
    cleanPrecisions = [p["precision"] for p in performances if p["precision"] is not None]
    cleanRecalls = [p["recall"] for p in performances if p["recall"] is not None]
    cleanFScores = [p["F-score"] for p in performances if p["F-score"] is not None]
        
    cleanAccuracys = [p["accuracy"] for p in performances if p["accuracy"] is not None]
 
    with open("info_" + doc_name + ".txt", "a") as info:
        info.write(("Precision of {0}:\t{1}\tstd: {2}".format(label[0], mean(cleanPrecisionsClass1), std(cleanPrecisionsClass1)) + " " + "\t({0} to {1})".format(min(precisionsClass1), max(precisionsClass1))) + "\n")
        info.write(("Recall of {0}:\t{1}\tstd: {2}".format(label[0], mean(cleanRecallsClass1), std(cleanRecallsClass1)) + " " + "\t({0} to {1})".format(min(recallsClass1), max(recallsClass1))) + "\n")
        info.write(("F-Score of {0}:\t{1}\tstd: {2}".format(label[0], mean(cleanFScoresClass1), std(cleanFScoresClass1)) + " " + "\t({0} to {1})".format(min(fScoresClass1), max(fScoresClass1))) + "\n\n")
        
        info.write(("Precision of {0}:\t{1}\tstd: {2}".format(label[1], mean(cleanPrecisionsClass2), std(cleanPrecisionsClass2)) + " " + "\t({0} to {1})".format(min(precisionsClass2), max(precisionsClass2))) + "\n")
        info.write(("Recall of {0}:\t{1}\tstd: {2}".format(label[1], mean(cleanRecallsClass2), std(cleanRecallsClass2)) + " " + "\t({0} to {1})".format(min(recallsClass2), max(recallsClass2))) + "\n")
        info.write(("F-Score of {0}:\t{1}\tstd: {2}".format(label[1], mean(cleanFScoresClass2), std(cleanFScoresClass2)) + " " + "\t({0} to {1})".format(min(fScoresClass2), max(fScoresClass2))) + "\n\n")
        
        info.write(("Precision:\t{0}\tstd: {1}".format(mean(cleanPrecisions), std(cleanPrecisions)) + " " + "\t({0} to {1})".format(min(precisions), max(precisions))) + "\n")
        info.write(("Recall:\t\t{0}\tstd: {1}".format(mean(cleanRecalls), std(cleanRecalls)) + " " + "\t({0} to {1})".format(min(recalls), max(recalls))) + "\n")
        info.write(("F-Score:\t{0}\tstd: {1}".format(mean(cleanFScores), std(cleanFScores)) + " " + "\t({0} to {1})".format(min(fScores), max(fScores))) + "\n\n")
        
        info.write(("Accuracy:\t\t{0}\tstd: {1}".format(mean(accuracys), std(accuracys)) + " " + "\t({0} to {1})".format(min(cleanAccuracys), max(cleanAccuracys))) + "\n")
