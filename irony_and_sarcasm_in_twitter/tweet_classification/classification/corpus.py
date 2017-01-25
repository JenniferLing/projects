"""
@copyright: Jennifer Ling
Adapted from Konstantin Buschmeier (code for amazon review sentiment prediction)
"""
# ------- configuration ---------

CORPUS_PATH = "../corpora/"
ALL_IDS_PATH = "IDs.txt"

# ------- import -------------

from re import sub
import re, os
import codecs
import pickle
from tweet import Tweet

# --------- code ----------
def loadTagDictionary():
    """ 
    Mapping from tweet-ID to tagged tweet text. 
    All tweets have to be tagged before run classification in own process due to time efficiency.
    Identifying corresponding tagged tweets via tweet ID.
    Format: 
    1    <tweet_id_1>_CD
    2    <tweet_text_1_word_1>_<pos_tag> <tweet_text_1_word_2>_<pos_tag> <tweet_text_1_word_3>_<pos_tag>
    3    <tweet_id_2>_CD
    4    <tweet_text_2_word_1>_<pos_tag> <tweet_text_2_word_2>_<pos_tag> <tweet_text_2_word_3>_<pos_tag>
    etc.
    """
    
    TAGGED_PATH = "../corpora/tagged_tweets/"
    
    tag_dict = {}
    
    file_list = os.listdir(TAGGED_PATH)
    for filename in file_list:
        result = codecs.open(TAGGED_PATH + filename, "r",encoding='utf-8').read().split('\n')
 
        for i in range(0, len(result)-1,2):
            ID = re.findall('[0-9]+', result[i])[0]
            tweet = result[i+1]
            tag_dict[ID] = tweet

    return tag_dict

def readTweetsAsIDFiles(tweetIDs, folder, label):
    """
    Returns a dictionary containing tweets to the given IDs. 
    Reads from folder where each txt file is one tweet and information can be readed line per line.
    """
    tag_dict = loadTagDictionary()
            
    return {tweetID: Tweet(tag_dict[tweetID.strip()],filename="{0}{1}.txt".format(folder, tweetID.strip()), 
                            label=label) for tweetID in tweetIDs}
    
def readTweets(tweetIDs, folder, label):
    """
    Returns a dictionary containing tweets to the given IDs.
    Reads from csv file (name = label.csv) and each line is one tweet; data is tab-separated. 
    """
    tag_dict = loadTagDictionary()
    
    with codecs.open(folder + label + ".csv", 'r',encoding='utf-8') as tweetFile:
        tweets = tweetFile.read().split("\n")[:-1]
    
    tweet_dict = {}
    
    for tweet in tweets:
        tweet = tweet.strip().split("\t")
        tweetID = tweet[3].strip()
        if tweetID in tweetIDs:
            tweet_dict[tweetID] = tweet
    
    return {tweetID: Tweet(tag_dict[tweetID],rawTweet=tweet_dict[tweetID], 
                            label=label) for tweetID in tweet_dict.keys()}
        
                    
    

class Corpus(object):
    
    """Represents a corpus."""

    def __init__(self, class1, class2, IDsFilename="IDs.txt",
                corpusPath="../corpora/"):
        # Save File locations
        self.IDsFilename = IDsFilename
        self.corpusPath = corpusPath
        self.class1 = class1        
        self.class2 = class2
               
        # Load IDs
        self.class1IDs = []
        self.class2IDs = []  
        self.loadIDs()
        
        # Load tweets
        self.tweets = {}
        self.loadTweets()

        self.saveCorpus(filename="allIds.txt")

    def __repr__(self):
        description = """Corpus('{0}', class1='{1}', 
                        class2='{2}')"""
        return description.format(self.IDsFilename,
                            self.class1, 
                            self.class2)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        description = "Corpus with {0} {1} and {2} {3} (total {4}) tweets."
        numberOfClass1 = len(self.class1IDs)
        numberOfClass2 = len(self.class2IDs)
        return description.format(numberOfClass1, self.class1, 
                                    numberOfClass2, self.class2,
                                    numberOfClass1 + numberOfClass2)
    def loadIDs(self):
        
        location = CORPUS_PATH + self.IDsFilename
                
        with open(location, 'r') as idsFile:
            rawTweetIDs = idsFile.readlines()
        for rawID in rawTweetIDs:
            
            IDParts = rawID.split()
            
            if IDParts[0] == (self.class1.upper() + ":"):         
                self.class1IDs.append(IDParts[1])
            elif IDParts[0] == (self.class2.upper() + ":"):
                self.class2IDs.append(IDParts[1]) 
                                   
    def loadTweets(self):
        """Loads all tweets."""
        
        self.tweets.update(readTweets(self.class1IDs,
                                    self.corpusPath, 
                                    self.class1))
        self.tweets.update(readTweets(self.class2IDs, 
                                    self.corpusPath, 
                                    self.class2))

    def saveCorpus(self, path=None, filename=None):
        """Save a Corpus object to disk."""
        if path == None:
            path = self.corpusPath
        if filename == None:
            # Delete file extension
            filename = sub(r"\.[^\.]+$", "", self.IDsFilename)

        with open(path + filename + ".pk", 'wb') as dataFile:
            pickle.dump(self, dataFile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadCorpus(path=CORPUS_PATH, filename="training_set.pk"):
        """Load a Corpus object from disk."""
        with open(path + filename, 'rb') as dataFile:
            return pickle.load(dataFile)

    @property
    def class1Tweets(self):
        """Returns all tweets of class 1."""
        return {ID: tweet for ID, tweet in self.tweets.items() 
                            if ID in self.class1IDs}

    @property
    def class2Tweets(self):
        """Returns all stweets of class 2."""
        return {ID: tweet for ID, tweet in self.tweets.items() 
                            if ID in self.class2IDs}

    @property
    def tweetIDs(self):
        """Returns the IDs for all tweets.""" 
        return self.class1IDs + self.class2IDs
    
    @property
    def class1TweetIDs(self):
        """Returns the IDs for all tweets of class1.""" 
        return self.class1IDs
    
    @property
    def class2TweetIDs(self):
        """Returns the IDs for all tweets of class2."""
        return self.class2IDs
    
    @property
    def goldStandard(self):
        """Returns the IDs and the corresponding category (ironic/sarcasm)."""
        return {ID: 1 if self.tweets[ID].label == self.class1 else 0 for ID in self.tweetIDs}

## ---------------------- corpus function -------------------------------    
def readAllIDs(class1, class2, path=CORPUS_PATH+ALL_IDS_PATH):
    """Returns lists of IDs for pairs, ironic and regular reviews from the 
    given file.
    """
    class1IDs = []
    class2IDs = []

    with open(path, 'r') as idsFile:
        rawTweetIDs = idsFile.readlines()
          
    for rawID in rawTweetIDs:
         
        IDParts = rawID.split()
                    
        if IDParts[0] == (class1.upper() + ":"):         
            class1IDs.append(IDParts[1])
        elif IDParts[0] == (class2.upper() + ":"):
            class2IDs.append(IDParts[1])
    
    return class1IDs, class2IDs