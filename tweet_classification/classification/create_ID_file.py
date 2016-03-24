"""
@copyright: Jennifer Ling
"""

import codecs, re, os, sys, shutil, random


def createIDFile(train=True, test=True, labels=None):
    """
    Creates files with list of all tweet IDs of the corpus.
    @param list of class labels; default: ["irony", "sarcasm", "regular", "figurative"].
    """
    path = "../corpora/"
    
    if labels == None:
        labels = ["irony", "sarcasm", "regular", "figurative"]
    
    length = 0
    test_length = 0
      
    if train:
        output = codecs.open(path + "/IDs.txt", "w", encoding='utf-8')
        for label in labels:
            filename = path + "train/" + label + ".csv"
            with codecs.open(filename, "r", "utf-8") as read:
                for line in read:
                    if line:
                        line = line.strip().split("\t")
                        
                        try:
                            id = line[3].strip()
                        except:
                            print line
                            exit()
                    
                        if len(id) != 18:
                            print line
                            print id
                            print len(id)
                        output.write(label.upper() + ":\t" + id + "\n")
                        length += 1
                
    if test:
        test_output = codecs.open(path + "/testIDs.txt", "w", encoding='utf-8')
        for label in labels:
            filename = path + "test" + "/" + label + ".csv"
            with codecs.open(filename, "r", "utf-8") as read:
                for line in read:
                    if line:
                        line = line.strip().split("\t")
                        
                        try:
                            id = line[3].strip()
                        except:
                            print line
                            exit()
                    
                        if len(id) != 18:
                            print len(id)
                        test_output.write(label.upper() + ":\t" + id + "\n")
                        test_length += 1
    
    if train:
        output.close()
    if test:
        test_output.close()
    
    print str(length) + " training and " + str(test_length) + " test tweet IDs\n"

if __name__ == '__main__':
    createIDFile()