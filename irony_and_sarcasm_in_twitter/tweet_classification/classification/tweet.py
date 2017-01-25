# -*- coding: utf-8 -*-
"""
@copyright: Jennifer Ling
Adapted from Konstantin Buschmeier (code for amazon review sentiment prediction)
"""

############################## CONFIGURATIONS #####################################

###################################################################################

############################## IMPORTS ############################################

import codecs
import re
import os
from datetime import datetime
import nltk.data
from tokenizer import MyTokenizer

###################################################################################

def loadPolarityLexicon(filenames, categories):
    """
    Returns a dictionary containing the words from the given files
    and their corresponding polarity category.
    """
   
    assert len(filenames) == len(categories)
    polarityLexicon = {}
    for filename, category in zip(filenames, categories): 
        with codecs.open(filename, 'r', encoding='latin-1') as wordsFile:
            polarityLexicon.update({w.strip(): category 
                                for w in wordsFile.readlines() 
                                if w.strip() and not w.strip().startswith(";")})
    return polarityLexicon

class Tweet(object):
    """Represents a tweet and its meta data."""
    
    sentenceTokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def __init__(self, tagged_tweet, filename=None, rawTweet=None, label=None):
        self.tagged_tweet = tagged_tweet
        
        if filename is not None and rawTweet is None:
            self.parseFromFile(filename, label)
            
        elif rawTweet is not None and filename is None:
            self.parse(rawTweet, label)
            
        else:
            """created at, name, username, tweet_id, processed_text, geoLocation, place"""
            self.created_at = ""
            self.name = ""
            self.username = ""
            self.tweet_id = 0.0
            self.text = ""
            self.geo = datetime()
            self.place = None
            self.wordSpans = []
            self.wordPolarity = []
            self.sentenceSpans = []
        

    def __str__(self):
        return self.text

    def parseFromFile(self, filename, label):
        
        with codecs.open(filename, 'r',encoding='utf-8') as tweetFile:
            self.parse(tweetFile.read().split("\n")[:-1], label)
            
    def parse(self, rawTweet, label):
        
        self.label = label # string!
        self.created_at = rawTweet[0]
        self.name = rawTweet[1]
        self.username = rawTweet[2]
        self.tweet_id = rawTweet[3].strip()
        self.text = rawTweet[4]
        self.geo = rawTweet[5]
        self.place = rawTweet[6]
        
        self.text = self.preprocess(self.text, self.label)
        
        self.sentences = self.tokenizeSentences(self.text)
        
        self.taggedTweet = self.tagWords(self.tweet_id)
        
    @property
    def words(self):
        return [w for s in self.sentences for w in s.words]

    @property
    def bigrams(self):
        words = self.words
        return zip(words[:-1], words[1:])

    @property
    def positiveWords(self):
        return [p for s in self.sentences for p in s.positiveWords]
    
    @property
    def negativeWords(self):
        return [n for s in self.sentences for n in s.negativeWords]

    @property
    def polarity(self):
        """Returns the review's polarity."""
        if len(self.negativeWords) > len(self.positiveWords):
            return "positive"
        elif len(self.negativeWords) > len(self.positiveWords):
            return "negative"
        else:
            return "neutral"

    def preprocess(self, text, label):
        """Preprocesses the given text by removing class hashtags and some trouble chars."""
        
        if label == "irony":
            remove_hashtags = ["#ironisch", "#ironie", "#irony", "#ironic"]
        elif label == "sarcasm":
            remove_hashtags = ["#sarkastisch", "#sarkasmus", "#sarcasm", "#sarcastic"]
        elif label == "regular":
            remove_hashtags = ["#drugs", "#education", "#gopdebate", "#late", "#news", "#peace", "#politics", "#humour"]
        elif label == "figurative":
            remove_hashtags = ["#ironisch", "#ironie", "#irony", "#ironic", "#sarkastisch", "#sarkasmus", "#sarcasm", "#sarcastic"]
        else:
            remove_hashtags = []

        
        text = text.replace("[par]", "\t".decode())
        text = text.replace(u'\U000000AB', '"'.decode())
        text = text.replace(u'\U000000BB', '"'.decode())
        text = text.split()
    
        for hashtag in remove_hashtags:
            for i in range(len(text)):
                if text[i].lower().startswith(hashtag):
                    text.remove(text[i])
                    break

        return " ".join(text)

    def tokenizeSentences(self, text):
        
        return [Sentence(text)
                for text in self.sentenceTokenizer.tokenize(self.text, realign_boundaries=True)]
        
            
    def tagWords(self, tweet_id):
        
        tagged_tweet = self.tagged_tweet
        tagged_tweet = tagged_tweet.split()
        for i in range(len(tagged_tweet)):
            pair = tagged_tweet[i]
            if "_" not in pair:
                tagged_tweet[i] = pair + "_XX"
        
        return [Token(pair.split("_")[0], pair.split("_")[1]) for pair in tagged_tweet]
   
    def numberOfCharacters(self):
        return len(self.text)
    
    def numberOfWords(self):
        return len(self.words)

    def numberOfSentences(self):
        return len(self.sentences)

    # TODO: delete?
    def analysePolarity(self, words):
        self.positivePolarity = []
        self.negativePolarity = []
        for word in words:
            self.positivePolarity += [p for p in self.positiveWords 
                                        if p == word]
            self.negativePolarity += [n for n in self.negativeWords 
                                        if n == word]

class Sentence(object):
    """Represents a single sentence."""
    tok = MyTokenizer(preserve_case=True)
    
    __slots__ = ['text', 'words']

    def __init__(self, text):
#         start = None
#         end = None
        self.text = text
        self.words = self.tokenizeWords(self.text)

    def __repr__(self):
        return "Sentence({0})".format(self.text).encode('utf-8')

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return self.text

    def tokenizeWords(self, text):
        return [Token(w) for w in self.tok.tokenize(text)]
#         return [Token(w,p) for w,p in self.wordTagger.tag(self.wordTokenizer.tokenize(text))]

    @property
    def positiveWords(self):
        return [w.text for w in self.words if not w.positiveScore == 0]

    @property
    def positiveWordPositions(self):
        return [i for i, w in enumerate(self.words) if not w.positiveScore == 0]

    @property
    def negativeWords(self):
        return [w.text for w in self.words if not w.negativeScore == 0]
    
    @property
    def negativeWordPositions(self):
        return [i for i, w in enumerate(self.words) if not w.negativeScore == 0]

class Token(object):
    """Represents a single token."""
    
    NEGATIVE_WORDS_FILENAME = "../resources/negative-words.txt"
    POSITIVE_WORDS_FILENAME = "../resources/positive-words.txt"

    polarityLexicon = loadPolarityLexicon([POSITIVE_WORDS_FILENAME,
                                        NEGATIVE_WORDS_FILENAME],
                                        ["positive", "negative"])

    __slots__ = ['text', 'pos', 'positiveScore', 'negativeScore']

    def __init__(self, text, pos=None):
#         start = None
#         end = None
        self.text = text
        self.pos = pos
        if (self.text in self.polarityLexicon and 
                self.polarityLexicon[self.text] == "positive"):
            self.positiveScore = 1 
        else:
            self.positiveScore = 0

        if (self.text in self.polarityLexicon and 
                self.polarityLexicon[self.text] == "negative"):
            self.negativeScore = 1  
        else:
            self.negativeScore = 0
            
        

    def __repr__(self):
        print type(self.text)
        print type(self.pos)
        print type(self.polarity)
        return "Token({0}, {1}, {2})".format(self.text, 
                                        self.pos, 
                                        self.polarity)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return self.text

    def __eq__(self, other): 
        return (self.text == other.text and self.pos == other.pos and 
                self.positiveScore == other.positiveScore and 
                self.negativeScore == other.negativeScore)

    def __hash__(self):
        return hash(self.__repr__())

    @property
    def polarity(self):
        
        if self.positiveScore > self.negativeScore:
            return "positive"
        if self.positiveScore < self.negativeScore:
            return "negative"
        else:
            return "neutral"
