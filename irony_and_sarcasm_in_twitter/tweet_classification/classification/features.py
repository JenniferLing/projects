"""
@copyright: Jennifer Ling
Adapted from Konstantin Buschmeier (code for amazon review sentiment prediction)

Feature Extraction (each feature is mapped to a number).
Most of the features are implemented in 4 ways:
- feature occurs in tweet (binary) (<feature_name>_binary)
- number of occurrences of feature in tweet (<feature_name>)
- normalized number of occurrences of feature in tweet (<feature_name>_01)
- stacks for normalized numbers (<feature_name>_stack_<nr>) 
  with stacks >0, >0.2, >0.4, >0.6 and >0.8.
  (stack 1 (>0) == binary version of feature).
"""

#-------------------------- CONFIGURATIONS -----------------------------------------

NEGATIVE_WORDS_FILENAME = "../resources/negative-words.txt"
POSITIVE_WORDS_FILENAME = "../resources/positive-words.txt"

AFINN_FILENAME = "../resources/AFINN-111.txt"
INTERJECTIONS_FILENAME = "../resources/interjection_list.txt"
EMOTICONS_FILENAME = "../resources/emoticons.txt"
STOPWORDS_FILENAME = "../resources/stopwords.txt"

####################################################################################

#--------------------------------- IMPORTS ----------------------------------------
from tweet import Tweet
import re
from itertools import combinations
import codecs
from feature_config import REGEX_FEATURE_CONFIG_ALL, REGEX_FEATURE_CONFIG_SMALL, REGEX_FEATURE_CONFIG_VERY_SMALL, REGEX_FEATURE_CONFIG_ARFF, REGEX_FEATURE_CONFIG_GROUPS, REGEX_FEATURE_CONFIG_SPECIAL
####################################################################################
# -------------feature configuration -------------------------
feature_names = {0: u'Sentiment Score', 1: u'Length of Tweet', 2: u'Sentiment Smileys Neg', 3: u'Capitalized Words', 
                 4: u'Subjective Pronomina', 5: u'Sentiment Score Gap', 6: u'Number of Adv', 7: u'Number of Adj', 
                 8: u'Number of Interjections', 9: u'Other Hashtags', 10: u'Smileys', 11: u'Number of Verbs', 
                 12: u'Following Smileys', 13: u'Number of Stopwords', 14: u'Sentiment Smileys Pos', 
                 15: u'Symbols', 16: u'Length of sentences', 17: u'Number of Nouns',
                 18: u'Neg&Punctuation B', 19: u'Ellipsis and Punctuation B', 20: u'Neg&Ellipsis B', 
                 21: u'Negative Quotes B', 22: u'Positive Hyperbole B', 23: u'Negative Hyperbole B', 
                 24: u'Pos&Punctuation B', 25: u'Positive Quotes B', 26: u'Pos&Ellipsis B',
                 27: u'Other Hashtags Binary', 28: u'More negative Sentiment Scores Binary', 29: u'Negation Binary', 
                 30: u'Sentiment Smileys Neg Binary', 31: u'Valence Shift Binary', 32: u'Sentiment Smileys Pos Binary', 
                 33: u'Interjektion?! Binary', 34: u'More positive Sentiment Scores Binary', 35: u'URL Binary', 
                 36: u'Sentiment Score Binary', 37: u'Symbols Binary', 38: u'Repeated Words Binary', 39: u'@User Binary',
                 40: u'Capitalized Words normalized', 41: u'Symbols normalized', 42: u'Sentiment Score Gap normalized', 
                 43: u'Sentiment Score Pos normalized', 44: u'Sentiment Score Neg normalized', 
                 45: u'Number of Verbs normalized', 46: u'Sentiment Smiley Pos normalized', 
                 47: u'Sentiment Score Neut normalized', 48: u'Smileys normalized', 49: u'Number of Interjections normalized', 
                 50: u'Number of Adv normalized', 51: u'Number of Adj normalized', 52: u'Length of sentences normalized', 
                 53: u'Sentiment Smiley Neg normalized', 54: u'Other Hashtags normalized', 55: u'Following Smileys normalized', 
                 56: u'Number of Stopwords normalized', 57: u'Number of Nouns normalized', 58: u'Sentiment Smileys normalized', 
                 59: u'Subjective Pronomina normalized',
                 60: u'Capitalized Words Stack 1', 61: u'Capitalized Words Stack 2', 62: u'Capitalized Words Stack 3', 63: u'Capitalized Words Stack 4', 
                 64: u'Capitalized Words Stack 5', 65: u'Symbols Stack 1', 66: u'Symbols Stack 2', 67: u'Symbols Stack 3', 68: u'Symbols Stack 4', 
                 69: u'Symbols Stack 5', 70: u'Sentiment Score Gap Stack 1', 71: u'Sentiment Score Gap Stack 2', 72: u'Sentiment Score Gap Stack 3', 
                 73: u'Sentiment Score Gap Stack 4', 74: u'Sentiment Score Gap Stack 5', 75: u'Sentiment Score Pos Stack 1', 
                 76: u'Sentiment Score Pos Stack 2', 77: u'Sentiment Score Pos Stack 3', 78: u'Sentiment Score Pos Stack 4', 
                 79: u'Sentiment Score Pos Stack 5', 80: u'Sentiment Score Neg Stack 1', 81: u'Sentiment Score Neg Stack 2', 
                 82: u'Sentiment Score Neg Stack 3', 83: u'Sentiment Score Neg Stack 4', 84: u'Sentiment Score Neg Stack 5', 
                 85: u'Number of Verbs Stack 1', 86: u'Number of Verbs Stack 2', 87: u'Number of Verbs Stack 3', 88: u'Number of Verbs Stack 4', 
                 89: u'Number of Verbs Stack 5', 90: u'Sentiment Smileys Pos Stack 1', 91: u'Sentiment Smileys Pos Stack 2', 
                 92: u'Sentiment Smileys Pos Stack 3', 93: u'Sentiment Smileys Pos Stack 4', 94: u'Sentiment Smileys Pos Stack 5', 
                 95: u'Smileys Stack 1', 96: u'Smileys Stack 2', 97: u'Smileys Stack 3', 98: u'Smileys Stack 4', 99: u'Smileys Stack 5', 
                 100: u'Number of Interjections Stack 1', 101: u'Number of Interjections Stack 2', 102: u'Number of Interjections Stack 3', 
                 103: u'Number of Interjections Stack 4', 104: u'Number of Interjections Stack 5', 105: u'Number of Adv Stack 1', 
                 106: u'Number of Adv Stack 2', 107: u'Number of Adv Stack 3', 108: u'Number of Adv Stack 4', 109: u'Number of Adv Stack 5', 
                 110: u'Number of Adj Stack 1', 111: u'Number of Adj Stack 2', 112: u'Number of Adj Stack 3', 113: u'Number of Adj Stack 4', 
                 114: u'Number of Adj Stack 5', 115: u'Length of sentences Stack 1', 116: u'Length of sentences Stack 2', 
                 117: u'Length of sentences Stack 3', 118: u'Length of sentences Stack 4', 119: u'Length of sentences Stack 5', 
                 120: u'Sentiment Smileys Neg Stack 1', 121: u'Sentiment Smileys Neg Stack 2', 122: u'Sentiment Smileys Neg Stack 3', 
                 123: u'Sentiment Smileys Neg Stack 4', 124: u'Sentiment Smileys Neg Stack 5', 125: u'Other Hashtags Stack 1', 
                 126: u'Other Hashtags Stack 2', 127: u'Other Hashtags Stack 3', 128: u'Other Hashtags Stack 4', 129: u'Other Hashtags Stack 5', 
                 130: u'Following Smileys Stack 1', 131: u'Following Smileys Stack 2', 132: u'Following Smileys Stack 3', 
                 133: u'Following Smileys Stack 4', 134: u'Following Smileys Stack 5', 135: u'Number of Stopwords Stack 1', 
                 136: u'Number of Stopwords Stack 2', 137: u'Number of Stopwords Stack 3', 138: u'Number of Stopwords Stack 4', 
                 139: u'Number of Stopwords Stack 5', 140: u'Number of Nouns Stack 1', 141: u'Number of Nouns Stack 2', 142: u'Number of Nouns Stack 3', 
                 143: u'Number of Nouns Stack 4', 144: u'Number of Nouns Stack 5', 145: u'Sentiment Smileys Stack 1', 146: u'Sentiment Smileys Stack 2', 
                 147: u'Sentiment Smileys Stack 3', 148: u'Sentiment Smileys Stack 4', 149: u'Sentiment Smileys Stack 5', 
                 150: u'Subjective Pronomina Stack 1', 151: u'Subjective Pronomina Stack 2', 152: u'Subjective Pronomina Stack 3', 
                 153: u'Subjective Pronomina Stack 4', 154: u'Subjective Pronomina Stack 5',
                 155: u'Sentiment Score Neut Stack', 156: u'Tweet Length normalized'}

max_tweet_length = 77 # = 140/2 + 10%
max_sentiment_score_gap = 10

# --------------------------------------------------------------------------------------------------------------------------------

class Feature(object):
    """
    Basic feature class. A feature has a name and some function that extracts
    information from a tweet.
    """
    def __init__(self, name, short=None, function=None):
        self.name = name.encode("utf-8")
        if not function == None:
            self.extract = function
        if not short == None:
            self.short = short[:4].encode("utf-8")
        else: 
            self.short = name[:4].encode("utf-8")

    def __repr__(self):
        return "Feature({name}, {function})".format(name=self.name, 
                                                function=self.extract.__name__)

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        return u"{name}({short})".format(name=self.name, short=self.short)

    def extract(self, tweet):
        """Extract the information from the review."""
        pass
    
class RegularExpressionFeature(Feature):
    """A feature that is based on regular expressions."""
    def __init__(self, name, short=None, regEx=r""):
        Feature.__init__(self, name, short)
        self.regEx = re.compile(regEx, flags=re.UNICODE|re.VERBOSE)
        
    def __repr__(self):
        return "RegExFeature({name})".format(name=self.name)

    def extract(self, tweet):
        """Search for the regular expression in the tweets' text."""
        found = self.regEx.findall(tweet.text)
        for f in found:
            if len(f) > 0:
                return 1
            else:
                continue
            
        return 0

#---------------- FEATURE TEMPLATES----------------------------------
# ---- Woerterbuecher laden ----- 

def loadStopwords():
    """ load stopwords list """
    stopword_list = []
    stopwords = open(STOPWORDS_FILENAME, "r").read().split("\n")
    for word in stopwords:
        if word != "\n":
            stopword_list.append(word)
    return stopword_list

def loadSentimentScore():
    """ sentiment score for words --> AFINN """
    sentiment_scores_dict = {}
     
    scores = codecs.open(AFINN_FILENAME, "r",encoding='utf-8').read().split("\n")
    for score_pair in scores:
        score_pair = score_pair.split("\t")
        
        word = score_pair[0].encode('utf-8')
               
            
        score = int(score_pair[1])
     
        sentiment_scores_dict[word] = score
     
    return sentiment_scores_dict

def loadInterjections():
    """ load interjection list from file """
    interjections = open(INTERJECTIONS_FILENAME, "r").read().split("\n")
     
    interjection_list = []
     
    for interjection in interjections:
        interjection = interjection.strip()
        if len(interjection) > 0:
            interjection = interjection.split("\t")
            interjection_list.append(interjection[0])
             
            alternative = interjection[1].split(",")
            for alt in alternative:
                if len(alt) > 0 and len(alt.split()) < 2:
                    alt = alt.strip("?! ")
                    interjection_list.append(alt)
     
    return interjection_list

def loadSmileys():
    """
    Unicode encoded emoticons from http://apps.timwhitlock.info/emoji/tables/unicode.
    """
    all_smileys = [u"\U0001F601",u"\U0001F602",u"\U0001F603",u"\U0001F604",u"\U0001F605",u"\U0001F606",u"\U0001F609",u"\U0001F60A",u"\U0001F60B",u"\U0001F60C",u"\U0001F60D",u"\U0001F60F",u"\U0001F612",u"\U0001F613",u"\U0001F614",u"\U0001F616",u"\U0001F618",u"\U0001F61A",u"\U0001F61C",u"\U0001F61D",u"\U0001F61E",u"\U0001F620",u"\U0001F621",u"\U0001F622",u"\U0001F623",u"\U0001F624",u"\U0001F625",u"\U0001F628",u"\U0001F629",u"\U0001F62A",u"\U0001F62B",u"\U0001F62D",u"\U0001F630",u"\U0001F631",u"\U0001F632",u"\U0001F633",u"\U0001F635",u"\U0001F637",u"\U0001F638",u"\U0001F639",u"\U0001F63A",u"\U0001F63B",u"\U0001F63C",u"\U0001F63D",u"\U0001F63E",u"\U0001F63F",u"\U0001F640",u"\U0001F645",u"\U0001F646",u"\U0001F647",u"\U0001F648",u"\U0001F649",u"\U0001F64A",u"\U0001F64B",u"\U0001F64C",u"\U0001F64D",u"\U0001F64E",u"\U0001F64F",u"\U00002702",u"\U00002705",u"\U00002708",u"\U00002709",u"\U0000270A",u"\U0000270B",u"\U0000270C",u"\U0000270F",u"\U00002712",u"\U00002714",u"\U00002716",u"\U00002728",u"\U00002733",u"\U00002734",u"\U00002744",u"\U00002747",u"\U0000274C",u"\U0000274E",u"\U00002753",u"\U00002754",u"\U00002755",u"\U00002757",u"\U00002764",u"\U00002795",u"\U00002796",u"\U00002797",u"\U000027A1",u"\U000027B0",u"\U0001F680",u"\U0001F683",u"\U0001F684",u"\U0001F685",u"\U0001F687",u"\U0001F689",u"\U0001F68C",u"\U0001F68F",u"\U0001F691",u"\U0001F692",u"\U0001F693",u"\U0001F695",u"\U0001F697",u"\U0001F699",u"\U0001F69A",u"\U0001F6A2",u"\U0001F6A4",u"\U0001F6A5",u"\U0001F6A7",u"\U0001F6A8",u"\U0001F6A9",u"\U0001F6AA",u"\U0001F6AB",u"\U0001F6AC",u"\U0001F6AD",u"\U0001F6B2",u"\U0001F6B6",u"\U0001F6B9",u"\U0001F6BA",u"\U0001F6BB",u"\U0001F6BC",u"\U0001F6BD",u"\U0001F6BE",u"\U0001F6C0",u"\U000024C2",u"\U0001F170",u"\U0001F171",u"\U0001F17E",u"\U0001F17F",u"\U0001F18E",u"\U0001F191",u"\U0001F192",u"\U0001F193",u"\U0001F194",u"\U0001F195",u"\U0001F196",u"\U0001F197",u"\U0001F198",u"\U0001F199",u"\U0001F19A",u"\U0001F1E9 \U0001F1EA",u"\U0001F1EC \U0001F1E7",u"\U0001F1E8 \U0001F1F3",u"\U0001F1EF \U0001F1F5",u"\U0001F1F0 \U0001F1F7",u"\U0001F1EB \U0001F1F7",u"\U0001F1EA \U0001F1F8",u"\U0001F1EE \U0001F1F9",u"\U0001F1FA \U0001F1F8",u"\U0001F1F7 \U0001F1FA",u"\U0001F201",u"\U0001F202",u"\U0001F21A",u"\U0001F22F",u"\U0001F232",u"\U0001F233",u"\U0001F234",u"\U0001F235",u"\U0001F236",u"\U0001F237",u"\U0001F238",u"\U0001F239",u"\U0001F23A",u"\U0001F250",u"\U0001F251",u"\U000000A9",u"\U000000AE",u"\U0000203C",u"\U00002049",u"\U00000038 \U000020E3",u"\U00000039 \U000020E3",u"\U00000037 \U000020E3",u"\U00000036 \U000020E3",u"\U00000031 \U000020E3",u"\U00000030 \U000020E3",u"\U00000032 \U000020E3",u"\U00000033 \U000020E3",u"\U00000035 \U000020E3",u"\U00000034 \U000020E3",u"\U00000023 \U000020E3",u"\U00002122",u"\U00002139",u"\U00002194",u"\U00002195",u"\U00002196",u"\U00002197",u"\U00002198",u"\U00002199",u"\U000021A9",u"\U000021AA",u"\U0000231A",u"\U0000231B",u"\U000023E9",u"\U000023EA",u"\U000023EB",u"\U000023EC",u"\U000023F0",u"\U000023F3",u"\U000025AA",u"\U000025AB",u"\U000025B6",u"\U000025C0",u"\U000025FB",u"\U000025FC",u"\U000025FD",u"\U000025FE",u"\U00002600",u"\U00002601",u"\U0000260E",u"\U00002611",u"\U00002614",u"\U00002615",u"\U0000261D",u"\U0000263A",u"\U00002648",u"\U00002649",u"\U0000264A",u"\U0000264B",u"\U0000264C",u"\U0000264D",u"\U0000264E",u"\U0000264F",u"\U00002650",u"\U00002651",u"\U00002652",u"\U00002653",u"\U00002660",u"\U00002663",u"\U00002665",u"\U00002666",u"\U00002668",u"\U0000267B",u"\U0000267F",u"\U00002693",u"\U000026A0",u"\U000026A1",u"\U000026AA",u"\U000026AB",u"\U000026BD",u"\U000026BE",u"\U000026C4",u"\U000026C5",u"\U000026CE",u"\U000026D4",u"\U000026EA",u"\U000026F2",u"\U000026F3",u"\U000026F5",u"\U000026FA",u"\U000026FD",u"\U00002934",u"\U00002935",u"\U00002B05",u"\U00002B06",u"\U00002B07",u"\U00002B1B",u"\U00002B1C",u"\U00002B50",u"\U00002B55",u"\U00003030",u"\U0000303D",u"\U00003297",u"\U00003299",u"\U0001F004",u"\U0001F0CF",u"\U0001F300",u"\U0001F301",u"\U0001F302",u"\U0001F303",u"\U0001F304",u"\U0001F305",u"\U0001F306",u"\U0001F307",u"\U0001F308",u"\U0001F309",u"\U0001F30A",u"\U0001F30B",u"\U0001F30C",u"\U0001F30F",u"\U0001F311",u"\U0001F313",u"\U0001F314",u"\U0001F315",u"\U0001F319",u"\U0001F31B",u"\U0001F31F",u"\U0001F320",u"\U0001F330",u"\U0001F331",u"\U0001F334",u"\U0001F335",u"\U0001F337",u"\U0001F338",u"\U0001F339",u"\U0001F33A",u"\U0001F33B",u"\U0001F33C",u"\U0001F33D",u"\U0001F33E",u"\U0001F33F",u"\U0001F340",u"\U0001F341",u"\U0001F342",u"\U0001F343",u"\U0001F344",u"\U0001F345",u"\U0001F346",u"\U0001F347",u"\U0001F348",u"\U0001F349",u"\U0001F34A",u"\U0001F34C",u"\U0001F34D",u"\U0001F34E",u"\U0001F34F",u"\U0001F351",u"\U0001F352",u"\U0001F353",u"\U0001F354",u"\U0001F355",u"\U0001F356",u"\U0001F357",u"\U0001F358",u"\U0001F359",u"\U0001F35A",u"\U0001F35B",u"\U0001F35C",u"\U0001F35D",u"\U0001F35E",u"\U0001F35F",u"\U0001F360",u"\U0001F361",u"\U0001F362",u"\U0001F363",u"\U0001F364",u"\U0001F365",u"\U0001F366",u"\U0001F367",u"\U0001F368",u"\U0001F369",u"\U0001F36A",u"\U0001F36B",u"\U0001F36C",u"\U0001F36D",u"\U0001F36E",u"\U0001F36F",u"\U0001F370",u"\U0001F371",u"\U0001F372",u"\U0001F373",u"\U0001F374",u"\U0001F375",u"\U0001F376",u"\U0001F377",u"\U0001F378",u"\U0001F379",u"\U0001F37A",u"\U0001F37B",u"\U0001F380",u"\U0001F381",u"\U0001F382",u"\U0001F383",u"\U0001F384",u"\U0001F385",u"\U0001F386",u"\U0001F387",u"\U0001F388",u"\U0001F389",u"\U0001F38A",u"\U0001F38B",u"\U0001F38C",u"\U0001F38D",u"\U0001F38E",u"\U0001F38F",u"\U0001F390",u"\U0001F391",u"\U0001F392",u"\U0001F393",u"\U0001F3A0",u"\U0001F3A1",u"\U0001F3A2",u"\U0001F3A3",u"\U0001F3A4",u"\U0001F3A5",u"\U0001F3A6",u"\U0001F3A7",u"\U0001F3A8",u"\U0001F3A9",u"\U0001F3AA",u"\U0001F3AB",u"\U0001F3AC",u"\U0001F3AD",u"\U0001F3AE",u"\U0001F3AF",u"\U0001F3B0",u"\U0001F3B1",u"\U0001F3B2",u"\U0001F3B3",u"\U0001F3B4",u"\U0001F3B5",u"\U0001F3B6",u"\U0001F3B7",u"\U0001F3B8",u"\U0001F3B9",u"\U0001F3BA",u"\U0001F3BB",u"\U0001F3BC",u"\U0001F3BD",u"\U0001F3BE",u"\U0001F3BF",u"\U0001F3C0",u"\U0001F3C1",u"\U0001F3C2",u"\U0001F3C3",u"\U0001F3C4",u"\U0001F3C6",u"\U0001F3C8",u"\U0001F3CA",u"\U0001F3E0",u"\U0001F3E1",u"\U0001F3E2",u"\U0001F3E3",u"\U0001F3E5",u"\U0001F3E6",u"\U0001F3E7",u"\U0001F3E8",u"\U0001F3E9",u"\U0001F3EA",u"\U0001F3EB",u"\U0001F3EC",u"\U0001F3ED",u"\U0001F3EE",u"\U0001F3EF",u"\U0001F3F0",u"\U0001F40C",u"\U0001F40D",u"\U0001F40E",u"\U0001F411",u"\U0001F412",u"\U0001F414",u"\U0001F417",u"\U0001F418",u"\U0001F419",u"\U0001F41A",u"\U0001F41B",u"\U0001F41C",u"\U0001F41D",u"\U0001F41E",u"\U0001F41F",u"\U0001F420",u"\U0001F421",u"\U0001F422",u"\U0001F423",u"\U0001F424",u"\U0001F425",u"\U0001F426",u"\U0001F427",u"\U0001F428",u"\U0001F429",u"\U0001F42B",u"\U0001F42C",u"\U0001F42D",u"\U0001F42E",u"\U0001F42F",u"\U0001F430",u"\U0001F431",u"\U0001F432",u"\U0001F433",u"\U0001F434",u"\U0001F435",u"\U0001F436",u"\U0001F437",u"\U0001F438",u"\U0001F439",u"\U0001F43A",u"\U0001F43B",u"\U0001F43C",u"\U0001F43D",u"\U0001F43E",u"\U0001F440",u"\U0001F442",u"\U0001F443",u"\U0001F444",u"\U0001F445",u"\U0001F446",u"\U0001F447",u"\U0001F448",u"\U0001F449",u"\U0001F44A",u"\U0001F44B",u"\U0001F44C",u"\U0001F44D",u"\U0001F44E",u"\U0001F44F",u"\U0001F450",u"\U0001F451",u"\U0001F452",u"\U0001F453",u"\U0001F454",u"\U0001F455",u"\U0001F456",u"\U0001F457",u"\U0001F458",u"\U0001F459",u"\U0001F45A",u"\U0001F45B",u"\U0001F45C",u"\U0001F45D",u"\U0001F45E",u"\U0001F45F",u"\U0001F460",u"\U0001F461",u"\U0001F462",u"\U0001F463",u"\U0001F464",u"\U0001F466",u"\U0001F467",u"\U0001F468",u"\U0001F469",u"\U0001F46A",u"\U0001F46B",u"\U0001F46E",u"\U0001F46F",u"\U0001F470",u"\U0001F471",u"\U0001F472",u"\U0001F473",u"\U0001F474",u"\U0001F475",u"\U0001F476",u"\U0001F477",u"\U0001F478",u"\U0001F479",u"\U0001F47A",u"\U0001F47B",u"\U0001F47C",u"\U0001F47D",u"\U0001F47E",u"\U0001F47F",u"\U0001F480",u"\U0001F481",u"\U0001F482",u"\U0001F483",u"\U0001F484",u"\U0001F485",u"\U0001F486",u"\U0001F487",u"\U0001F488",u"\U0001F489",u"\U0001F48A",u"\U0001F48B",u"\U0001F48C",u"\U0001F48D",u"\U0001F48E",u"\U0001F48F",u"\U0001F490",u"\U0001F491",u"\U0001F492",u"\U0001F493",u"\U0001F494",u"\U0001F495",u"\U0001F496",u"\U0001F497",u"\U0001F498",u"\U0001F499",u"\U0001F49A",u"\U0001F49B",u"\U0001F49C",u"\U0001F49D",u"\U0001F49E",u"\U0001F49F",u"\U0001F4A0",u"\U0001F4A1",u"\U0001F4A2",u"\U0001F4A3",u"\U0001F4A4",u"\U0001F4A5",u"\U0001F4A6",u"\U0001F4A7",u"\U0001F4A8",u"\U0001F4A9",u"\U0001F4AA",u"\U0001F4AB",u"\U0001F4AC",u"\U0001F4AE",u"\U0001F4AF",u"\U0001F4B0",u"\U0001F4B1",u"\U0001F4B2",u"\U0001F4B3",u"\U0001F4B4",u"\U0001F4B5",u"\U0001F4B8",u"\U0001F4B9",u"\U0001F4BA",u"\U0001F4BB",u"\U0001F4BC",u"\U0001F4BD",u"\U0001F4BE",u"\U0001F4BF",u"\U0001F4C0",u"\U0001F4C1",u"\U0001F4C2",u"\U0001F4C3",u"\U0001F4C4",u"\U0001F4C5",u"\U0001F4C6",u"\U0001F4C7",u"\U0001F4C8",u"\U0001F4C9",u"\U0001F4CA",u"\U0001F4CB",u"\U0001F4CC",u"\U0001F4CD",u"\U0001F4CE",u"\U0001F4CF",u"\U0001F4D0",u"\U0001F4D1",u"\U0001F4D2",u"\U0001F4D3",u"\U0001F4D4",u"\U0001F4D5",u"\U0001F4D6",u"\U0001F4D7",u"\U0001F4D8",u"\U0001F4D9",u"\U0001F4DA",u"\U0001F4DB",u"\U0001F4DC",u"\U0001F4DD",u"\U0001F4DE",u"\U0001F4DF",u"\U0001F4E0",u"\U0001F4E1",u"\U0001F4E2",u"\U0001F4E3",u"\U0001F4E4",u"\U0001F4E5",u"\U0001F4E6",u"\U0001F4E7",u"\U0001F4E8",u"\U0001F4E9",u"\U0001F4EA",u"\U0001F4EB",u"\U0001F4EE",u"\U0001F4F0",u"\U0001F4F1",u"\U0001F4F2",u"\U0001F4F3",u"\U0001F4F4",u"\U0001F4F6",u"\U0001F4F7",u"\U0001F4F9",u"\U0001F4FA",u"\U0001F4FB",u"\U0001F4FC",u"\U0001F503",u"\U0001F50A",u"\U0001F50B",u"\U0001F50C",u"\U0001F50D",u"\U0001F50E",u"\U0001F50F",u"\U0001F510",u"\U0001F511",u"\U0001F512",u"\U0001F513",u"\U0001F514",u"\U0001F516",u"\U0001F517",u"\U0001F518",u"\U0001F519",u"\U0001F51A",u"\U0001F51B",u"\U0001F51C",u"\U0001F51D",u"\U0001F51E",u"\U0001F51F",u"\U0001F520",u"\U0001F521",u"\U0001F522",u"\U0001F523",u"\U0001F524",u"\U0001F525",u"\U0001F526",u"\U0001F527",u"\U0001F528",u"\U0001F529",u"\U0001F52A",u"\U0001F52B",u"\U0001F52E",u"\U0001F52F",u"\U0001F530",u"\U0001F531",u"\U0001F532",u"\U0001F533",u"\U0001F534",u"\U0001F535",u"\U0001F536",u"\U0001F537",u"\U0001F538",u"\U0001F539",u"\U0001F53A",u"\U0001F53B",u"\U0001F53C",u"\U0001F53D",u"\U0001F550",u"\U0001F551",u"\U0001F552",u"\U0001F553",u"\U0001F554",u"\U0001F555",u"\U0001F556",u"\U0001F557",u"\U0001F558",u"\U0001F559",u"\U0001F55A",u"\U0001F55B",u"\U0001F5FB",u"\U0001F5FC",u"\U0001F5FD",u"\U0001F5FE",u"\U0001F5FF",u"\U0001F600",u"\U0001F607",u"\U0001F608",u"\U0001F60E",u"\U0001F610",u"\U0001F611",u"\U0001F615",u"\U0001F617",u"\U0001F619",u"\U0001F61B",u"\U0001F61F",u"\U0001F626",u"\U0001F627",u"\U0001F62C",u"\U0001F62E",u"\U0001F62F",u"\U0001F634",u"\U0001F636",u"\U0001F681",u"\U0001F682",u"\U0001F686",u"\U0001F688",u"\U0001F68A",u"\U0001F68D",u"\U0001F68E",u"\U0001F690",u"\U0001F694",u"\U0001F696",u"\U0001F698",u"\U0001F69B",u"\U0001F69C",u"\U0001F69D",u"\U0001F69E",u"\U0001F69F",u"\U0001F6A0",u"\U0001F6A1",u"\U0001F6A3",u"\U0001F6A6",u"\U0001F6AE",u"\U0001F6AF",u"\U0001F6B0",u"\U0001F6B1",u"\U0001F6B3",u"\U0001F6B4",u"\U0001F6B5",u"\U0001F6B7",u"\U0001F6B8",u"\U0001F6BF",u"\U0001F6C1",u"\U0001F6C2",u"\U0001F6C3",u"\U0001F6C4",u"\U0001F6C5",u"\U0001F30D",u"\U0001F30E",u"\U0001F310",u"\U0001F312",u"\U0001F316",u"\U0001F317",u"\U0001F318",u"\U0001F31A",u"\U0001F31C",u"\U0001F31D",u"\U0001F31E",u"\U0001F332",u"\U0001F333",u"\U0001F34B",u"\U0001F350",u"\U0001F37C",u"\U0001F3C7",u"\U0001F3C9",u"\U0001F3E4",u"\U0001F400",u"\U0001F401",u"\U0001F402",u"\U0001F403",u"\U0001F404",u"\U0001F405",u"\U0001F406",u"\U0001F407",u"\U0001F408",u"\U0001F409",u"\U0001F40A",u"\U0001F40B",u"\U0001F40F",u"\U0001F410",u"\U0001F413",u"\U0001F415",u"\U0001F416",u"\U0001F42A",u"\U0001F465",u"\U0001F46C",u"\U0001F46D",u"\U0001F4AD",u"\U0001F4B6",u"\U0001F4B7",u"\U0001F4EC",u"\U0001F4ED",u"\U0001F4EF",u"\U0001F4F5",u"\U0001F500",u"\U0001F501",u"\U0001F502",u"\U0001F504",u"\U0001F505",u"\U0001F506",u"\U0001F507",u"\U0001F509",u"\U0001F515",u"\U0001F52C",u"\U0001F52D",u"\U0001F55C",u"\U0001F55D",u"\U0001F55E",u"\U0001F55F",u"\U0001F560",u"\U0001F561",u"\U0001F562",u"\U0001F563",u"\U0001F564",u"\U0001F565",u"\U0001F566",u"\U0001F567",]
    all_smileys = set(all_smileys)
    
# ---- smiley label dict ------------
    # + = positive
    # - = negative
    # S = Symbol
    # F = Flag
    # L = Letter
    smileyLabelDict = dict()
    smileyLabelDict[u"\U0001F601"] = "+"
    smileyLabelDict[u"\U0001F602"] = "+"
    smileyLabelDict[u"\U0001F603"] = "+"
    smileyLabelDict[u"\U0001F604"] = "+"
    smileyLabelDict[u"\U0001F605"] = "+"
    smileyLabelDict[u"\U0001F606"] = "+"
    smileyLabelDict[u"\U0001F609"] = "+"
    smileyLabelDict[u"\U0001F60A"] = "+"
    smileyLabelDict[u"\U0001F60B"] = "+"
    smileyLabelDict[u"\U0001F60C"] = "+"
    smileyLabelDict[u"\U0001F60D"] = "+"
    smileyLabelDict[u"\U0001F60F"] = "-"
    smileyLabelDict[u"\U0001F612"] = "-"
    smileyLabelDict[u"\U0001F613"] = "-"
    smileyLabelDict[u"\U0001F614"] = "-"
    smileyLabelDict[u"\U0001F616"] = "-"
    smileyLabelDict[u"\U0001F618"] = "-"
    smileyLabelDict[u"\U0001F61A"] = "+"
    smileyLabelDict[u"\U0001F61C"] = "+"
    smileyLabelDict[u"\U0001F61D"] = "+"
    smileyLabelDict[u"\U0001F61E"] = "-"
    smileyLabelDict[u"\U0001F620"] = "-"
    smileyLabelDict[u"\U0001F621"] = "-"
    smileyLabelDict[u"\U0001F622"] = "-"
    smileyLabelDict[u"\U0001F623"] = "-"
    smileyLabelDict[u"\U0001F624"] = "+"
    smileyLabelDict[u"\U0001F625"] = "-"
    smileyLabelDict[u"\U0001F628"] = "-"
    smileyLabelDict[u"\U0001F629"] = "-"
    smileyLabelDict[u"\U0001F62A"] = "-"
    smileyLabelDict[u"\U0001F62B"] = "-"
    smileyLabelDict[u"\U0001F62D"] = "-"
    smileyLabelDict[u"\U0001F630"] = "-"
    smileyLabelDict[u"\U0001F631"] = "-"
    smileyLabelDict[u"\U0001F632"] = "-"
    smileyLabelDict[u"\U0001F633"] = "-"
    smileyLabelDict[u"\U0001F635"] = "-"
    smileyLabelDict[u"\U0001F637"] = "-"
    smileyLabelDict[u"\U0001F638"] = "+"
    smileyLabelDict[u"\U0001F639"] = "+"
    smileyLabelDict[u"\U0001F63A"] = "+"
    smileyLabelDict[u"\U0001F63B"] = "+"
    smileyLabelDict[u"\U0001F63C"] = "+"
    smileyLabelDict[u"\U0001F63D"] = "+"
    smileyLabelDict[u"\U0001F63E"] = "-"
    smileyLabelDict[u"\U0001F63F"] = "-"
    smileyLabelDict[u"\U0001F640"] = "-"
    smileyLabelDict[u"\U0001F645"] = "-"
    smileyLabelDict[u"\U0001F646"] = "+"
    smileyLabelDict[u"\U0001F647"] = "-"
    smileyLabelDict[u"\U0001F648"] = "-"
    smileyLabelDict[u"\U0001F649"] = "-"
    smileyLabelDict[u"\U0001F64A"] = "-"
    smileyLabelDict[u"\U0001F64B"] = "+"
    smileyLabelDict[u"\U0001F64C"] = "+"
    smileyLabelDict[u"\U0001F64D"] = "-"
    smileyLabelDict[u"\U0001F64E"] = "-"
    smileyLabelDict[u"\U0001F64F"] = "-"
    smileyLabelDict[u"\U00002702"] = "S"
    smileyLabelDict[u"\U00002705"] = "S"
    smileyLabelDict[u"\U00002708"] = "S"
    smileyLabelDict[u"\U00002709"] = "S"
    smileyLabelDict[u"\U0000270A"] = "-"
    smileyLabelDict[u"\U0000270B"] = "-"
    smileyLabelDict[u"\U0000270C"] = "+"
    smileyLabelDict[u"\U0000270F"] = "S"
    smileyLabelDict[u"\U00002712"] = "S"
    smileyLabelDict[u"\U00002714"] = "+"
    smileyLabelDict[u"\U00002716"] = "-"
    smileyLabelDict[u"\U00002728"] = "S"
    smileyLabelDict[u"\U00002733"] = "S"
    smileyLabelDict[u"\U00002734"] = "S"
    smileyLabelDict[u"\U00002744"] = "S"
    smileyLabelDict[u"\U00002747"] = "S"
    smileyLabelDict[u"\U0000274C"] = "-"
    smileyLabelDict[u"\U0000274E"] = "-"
    smileyLabelDict[u"\U00002753"] = "-"
    smileyLabelDict[u"\U00002754"] = "-"
    smileyLabelDict[u"\U00002755"] = "-"
    smileyLabelDict[u"\U00002757"] = "-"
    smileyLabelDict[u"\U00002764"] = "+"
    smileyLabelDict[u"\U00002795"] = "+"
    smileyLabelDict[u"\U00002796"] = "-"
    smileyLabelDict[u"\U00002797"] = "S"
    smileyLabelDict[u"\U000027A1"] = "S"
    smileyLabelDict[u"\U000027B0"] = "S"
    smileyLabelDict[u"\U0001F680"] = "S"
    smileyLabelDict[u"\U0001F683"] = "S"
    smileyLabelDict[u"\U0001F684"] = "S"
    smileyLabelDict[u"\U0001F685"] = "S"
    smileyLabelDict[u"\U0001F687"] = "S"
    smileyLabelDict[u"\U0001F689"] = "S"
    smileyLabelDict[u"\U0001F68C"] = "S"
    smileyLabelDict[u"\U0001F68F"] = "S"
    smileyLabelDict[u"\U0001F691"] = "S"
    smileyLabelDict[u"\U0001F692"] = "S"
    smileyLabelDict[u"\U0001F693"] = "S"
    smileyLabelDict[u"\U0001F695"] = "S"
    smileyLabelDict[u"\U0001F697"] = "S"
    smileyLabelDict[u"\U0001F699"] = "S"
    smileyLabelDict[u"\U0001F69A"] = "S"
    smileyLabelDict[u"\U0001F6A2"] = "S"
    smileyLabelDict[u"\U0001F6A4"] = "S"
    smileyLabelDict[u"\U0001F6A5"] = "S"
    smileyLabelDict[u"\U0001F6A7"] = "-"
    smileyLabelDict[u"\U0001F6A8"] = "-"
    smileyLabelDict[u"\U0001F6A9"] = "S"
    smileyLabelDict[u"\U0001F6AA"] = "S"
    smileyLabelDict[u"\U0001F6AB"] = "S"
    smileyLabelDict[u"\U0001F6AC"] = "S"
    smileyLabelDict[u"\U0001F6AD"] = "S"
    smileyLabelDict[u"\U0001F6B2"] = "S"
    smileyLabelDict[u"\U0001F6B6"] = "S"
    smileyLabelDict[u"\U0001F6B9"] = "S"
    smileyLabelDict[u"\U0001F6BA"] = "S"
    smileyLabelDict[u"\U0001F6BB"] = "S"
    smileyLabelDict[u"\U0001F6BC"] = "S"
    smileyLabelDict[u"\U0001F6BD"] = "S"
    smileyLabelDict[u"\U0001F6BE"] = "S"
    smileyLabelDict[u"\U0001F6C0"] = "S"
    smileyLabelDict[u"\U000024C2"] = "L"
    smileyLabelDict[u"\U0001F170"] = "L"
    smileyLabelDict[u"\U0001F171"] = "L"
    smileyLabelDict[u"\U0001F17E"] = "L"
    smileyLabelDict[u"\U0001F17F"] = "L"
    smileyLabelDict[u"\U0001F18E"] = "L"
    smileyLabelDict[u"\U0001F191"] = "L"
    smileyLabelDict[u"\U0001F192"] = "+"
    smileyLabelDict[u"\U0001F193"] = "+"
    smileyLabelDict[u"\U0001F194"] = "L"
    smileyLabelDict[u"\U0001F195"] = "+"
    smileyLabelDict[u"\U0001F196"] = "L"
    smileyLabelDict[u"\U0001F197"] = "+"
    smileyLabelDict[u"\U0001F198"] = "-"
    smileyLabelDict[u"\U0001F199"] = "L"
    smileyLabelDict[u"\U0001F19A"] = "L"
    smileyLabelDict[u"\U0001F1E9 \U0001F1EA"] = "F"
    smileyLabelDict[u"\U0001F1EC \U0001F1E7"] = "F"
    smileyLabelDict[u"\U0001F1E8 \U0001F1F3"] = "F"
    smileyLabelDict[u"\U0001F1EF \U0001F1F5"] = "F"
    smileyLabelDict[u"\U0001F1F0 \U0001F1F7"] = "F"
    smileyLabelDict[u"\U0001F1EB \U0001F1F7"] = "F"
    smileyLabelDict[u"\U0001F1EA \U0001F1F8"] = "F"
    smileyLabelDict[u"\U0001F1EE \U0001F1F9"] = "F"
    smileyLabelDict[u"\U0001F1FA \U0001F1F8"] = "F"
    smileyLabelDict[u"\U0001F1F7 \U0001F1FA"] = "F"
    smileyLabelDict[u"\U0001F201"] = "L"
    smileyLabelDict[u"\U0001F202"] = "L"
    smileyLabelDict[u"\U0001F21A"] = "L"
    smileyLabelDict[u"\U0001F22F"] = "L"
    smileyLabelDict[u"\U0001F232"] = "L"
    smileyLabelDict[u"\U0001F233"] = "L"
    smileyLabelDict[u"\U0001F234"] = "L"
    smileyLabelDict[u"\U0001F235"] = "L"
    smileyLabelDict[u"\U0001F236"] = "L"
    smileyLabelDict[u"\U0001F237"] = "L"
    smileyLabelDict[u"\U0001F238"] = "L"
    smileyLabelDict[u"\U0001F239"] = "L"
    smileyLabelDict[u"\U0001F23A"] = "L"
    smileyLabelDict[u"\U0001F250"] = "L"
    smileyLabelDict[u"\U0001F251"] = "L"
    smileyLabelDict[u"\U000000A9"] = "L"
    smileyLabelDict[u"\U000000AE"] = "L"
    smileyLabelDict[u"\U0000203C"] = "-"
    smileyLabelDict[u"\U00002049"] = "-"
    smileyLabelDict[u"\U00000038 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000039 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000037 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000036 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000031 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000030 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000032 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000033 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000035 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000034 \U000020E3"] = "L"
    smileyLabelDict[u"\U00000023 \U000020E3"] = "L"
    smileyLabelDict[u"\U00002122"] = "L"
    smileyLabelDict[u"\U00002139"] = "L"
    smileyLabelDict[u"\U00002194"] = "S"
    smileyLabelDict[u"\U00002195"] = "S"
    smileyLabelDict[u"\U00002196"] = "S"
    smileyLabelDict[u"\U00002197"] = "S"
    smileyLabelDict[u"\U00002198"] = "S"
    smileyLabelDict[u"\U00002199"] = "S"
    smileyLabelDict[u"\U000021A9"] = "S"
    smileyLabelDict[u"\U000021AA"] = "S"
    smileyLabelDict[u"\U0000231A"] = "S"
    smileyLabelDict[u"\U0000231B"] = "S"
    smileyLabelDict[u"\U000023E9"] = "S"
    smileyLabelDict[u"\U000023EA"] = "S"
    smileyLabelDict[u"\U000023EB"] = "S"
    smileyLabelDict[u"\U000023EC"] = "S"
    smileyLabelDict[u"\U000023F0"] = "S"
    smileyLabelDict[u"\U000023F3"] = "S"
    smileyLabelDict[u"\U000025AA"] = "S"
    smileyLabelDict[u"\U000025AB"] = "S"
    smileyLabelDict[u"\U000025B6"] = "S"
    smileyLabelDict[u"\U000025C0"] = "S"
    smileyLabelDict[u"\U000025FB"] = "S"
    smileyLabelDict[u"\U000025FC"] = "S"
    smileyLabelDict[u"\U000025FD"] = "S"
    smileyLabelDict[u"\U000025FE"] = "S"
    smileyLabelDict[u"\U00002600"] = "S"
    smileyLabelDict[u"\U00002601"] = "S"
    smileyLabelDict[u"\U0000260E"] = "S"
    smileyLabelDict[u"\U00002611"] = "S"
    smileyLabelDict[u"\U00002614"] = "S"
    smileyLabelDict[u"\U00002615"] = "S"
    smileyLabelDict[u"\U0000261D"] = "S"
    smileyLabelDict[u"\U0000263A"] = "S"
    smileyLabelDict[u"\U00002648"] = "S"
    smileyLabelDict[u"\U00002649"] = "S"
    smileyLabelDict[u"\U0000264A"] = "S"
    smileyLabelDict[u"\U0000264B"] = "S"
    smileyLabelDict[u"\U0000264C"] = "S"
    smileyLabelDict[u"\U0000264D"] = "S"
    smileyLabelDict[u"\U0000264E"] = "S"
    smileyLabelDict[u"\U0000264F"] = "S"
    smileyLabelDict[u"\U00002650"] = "S"
    smileyLabelDict[u"\U00002651"] = "S"
    smileyLabelDict[u"\U00002652"] = "S"
    smileyLabelDict[u"\U00002653"] = "S"
    smileyLabelDict[u"\U00002660"] = "S"
    smileyLabelDict[u"\U00002663"] = "S"
    smileyLabelDict[u"\U00002665"] = "S"
    smileyLabelDict[u"\U00002666"] = "S"
    smileyLabelDict[u"\U00002668"] = "S"
    smileyLabelDict[u"\U0000267B"] = "S"
    smileyLabelDict[u"\U0000267F"] = "S"
    smileyLabelDict[u"\U00002693"] = "S"
    smileyLabelDict[u"\U000026A0"] = "S"
    smileyLabelDict[u"\U000026A1"] = "S"
    smileyLabelDict[u"\U000026AA"] = "S"
    smileyLabelDict[u"\U000026AB"] = "S"
    smileyLabelDict[u"\U000026BD"] = "S"
    smileyLabelDict[u"\U000026BE"] = "S"
    smileyLabelDict[u"\U000026C4"] = "S"
    smileyLabelDict[u"\U000026C5"] = "S"
    smileyLabelDict[u"\U000026CE"] = "S"
    smileyLabelDict[u"\U000026D4"] = "S"
    smileyLabelDict[u"\U000026EA"] = "S"
    smileyLabelDict[u"\U000026F2"] = "S"
    smileyLabelDict[u"\U000026F3"] = "S"
    smileyLabelDict[u"\U000026F5"] = "S"
    smileyLabelDict[u"\U000026FA"] = "S"
    smileyLabelDict[u"\U000026FD"] = "S"
    smileyLabelDict[u"\U00002934"] = "S"
    smileyLabelDict[u"\U00002935"] = "S"
    smileyLabelDict[u"\U00002B05"] = "S"
    smileyLabelDict[u"\U00002B06"] = "S"
    smileyLabelDict[u"\U00002B07"] = "S"
    smileyLabelDict[u"\U00002B1B"] = "S"
    smileyLabelDict[u"\U00002B1C"] = "S"
    smileyLabelDict[u"\U00002B50"] = "S"
    smileyLabelDict[u"\U00002B55"] = "S"
    smileyLabelDict[u"\U00003030"] = "S"
    smileyLabelDict[u"\U0000303D"] = "S"
    smileyLabelDict[u"\U00003297"] = "S"
    smileyLabelDict[u"\U00003299"] = "S"
    smileyLabelDict[u"\U0001F004"] = "S"
    smileyLabelDict[u"\U0001F0CF"] = "S"
    smileyLabelDict[u"\U0001F300"] = "S"
    smileyLabelDict[u"\U0001F301"] = "S"
    smileyLabelDict[u"\U0001F302"] = "S"
    smileyLabelDict[u"\U0001F303"] = "S"
    smileyLabelDict[u"\U0001F304"] = "S"
    smileyLabelDict[u"\U0001F305"] = "S"
    smileyLabelDict[u"\U0001F306"] = "S"
    smileyLabelDict[u"\U0001F307"] = "S"
    smileyLabelDict[u"\U0001F308"] = "S"
    smileyLabelDict[u"\U0001F309"] = "S"
    smileyLabelDict[u"\U0001F30A"] = "S"
    smileyLabelDict[u"\U0001F30B"] = "S"
    smileyLabelDict[u"\U0001F30C"] = "S"
    smileyLabelDict[u"\U0001F30F"] = "S"
    smileyLabelDict[u"\U0001F311"] = "S"
    smileyLabelDict[u"\U0001F313"] = "S"
    smileyLabelDict[u"\U0001F314"] = "S"
    smileyLabelDict[u"\U0001F315"] = "S"
    smileyLabelDict[u"\U0001F319"] = "S"
    smileyLabelDict[u"\U0001F31B"] = "S"
    smileyLabelDict[u"\U0001F31F"] = "S"
    smileyLabelDict[u"\U0001F320"] = "S"
    smileyLabelDict[u"\U0001F330"] = "S"
    smileyLabelDict[u"\U0001F331"] = "S"
    smileyLabelDict[u"\U0001F334"] = "S"
    smileyLabelDict[u"\U0001F335"] = "S"
    smileyLabelDict[u"\U0001F337"] = "S"
    smileyLabelDict[u"\U0001F338"] = "S"
    smileyLabelDict[u"\U0001F339"] = "S"
    smileyLabelDict[u"\U0001F33A"] = "S"
    smileyLabelDict[u"\U0001F33B"] = "S"
    smileyLabelDict[u"\U0001F33C"] = "S"
    smileyLabelDict[u"\U0001F33D"] = "S"
    smileyLabelDict[u"\U0001F33E"] = "S"
    smileyLabelDict[u"\U0001F33F"] = "S"
    smileyLabelDict[u"\U0001F340"] = "S"
    smileyLabelDict[u"\U0001F341"] = "S"
    smileyLabelDict[u"\U0001F342"] = "S"
    smileyLabelDict[u"\U0001F343"] = "S"
    smileyLabelDict[u"\U0001F344"] = "S"
    smileyLabelDict[u"\U0001F345"] = "S"
    smileyLabelDict[u"\U0001F346"] = "S"
    smileyLabelDict[u"\U0001F347"] = "S"
    smileyLabelDict[u"\U0001F348"] = "S"
    smileyLabelDict[u"\U0001F349"] = "S"
    smileyLabelDict[u"\U0001F34A"] = "S"
    smileyLabelDict[u"\U0001F34C"] = "S"
    smileyLabelDict[u"\U0001F34D"] = "S"
    smileyLabelDict[u"\U0001F34E"] = "S"
    smileyLabelDict[u"\U0001F34F"] = "S"
    smileyLabelDict[u"\U0001F351"] = "S"
    smileyLabelDict[u"\U0001F352"] = "S"
    smileyLabelDict[u"\U0001F353"] = "S"
    smileyLabelDict[u"\U0001F354"] = "S"
    smileyLabelDict[u"\U0001F355"] = "S"
    smileyLabelDict[u"\U0001F356"] = "S"
    smileyLabelDict[u"\U0001F357"] = "S"
    smileyLabelDict[u"\U0001F358"] = "S"
    smileyLabelDict[u"\U0001F359"] = "S"
    smileyLabelDict[u"\U0001F35A"] = "S"
    smileyLabelDict[u"\U0001F35B"] = "S"
    smileyLabelDict[u"\U0001F35C"] = "S"
    smileyLabelDict[u"\U0001F35D"] = "S"
    smileyLabelDict[u"\U0001F35E"] = "S"
    smileyLabelDict[u"\U0001F35F"] = "S"
    smileyLabelDict[u"\U0001F360"] = "S"
    smileyLabelDict[u"\U0001F361"] = "S"
    smileyLabelDict[u"\U0001F362"] = "S"
    smileyLabelDict[u"\U0001F363"] = "S"
    smileyLabelDict[u"\U0001F364"] = "S"
    smileyLabelDict[u"\U0001F365"] = "S"
    smileyLabelDict[u"\U0001F366"] = "S"
    smileyLabelDict[u"\U0001F367"] = "S"
    smileyLabelDict[u"\U0001F368"] = "S"
    smileyLabelDict[u"\U0001F369"] = "S"
    smileyLabelDict[u"\U0001F36A"] = "S"
    smileyLabelDict[u"\U0001F36B"] = "S"
    smileyLabelDict[u"\U0001F36C"] = "S"
    smileyLabelDict[u"\U0001F36D"] = "S"
    smileyLabelDict[u"\U0001F36E"] = "S"
    smileyLabelDict[u"\U0001F36F"] = "S"
    smileyLabelDict[u"\U0001F370"] = "S"
    smileyLabelDict[u"\U0001F371"] = "S"
    smileyLabelDict[u"\U0001F372"] = "S"
    smileyLabelDict[u"\U0001F373"] = "S"
    smileyLabelDict[u"\U0001F374"] = "S"
    smileyLabelDict[u"\U0001F375"] = "S"
    smileyLabelDict[u"\U0001F376"] = "S"
    smileyLabelDict[u"\U0001F377"] = "S"
    smileyLabelDict[u"\U0001F378"] = "S"
    smileyLabelDict[u"\U0001F379"] = "S"
    smileyLabelDict[u"\U0001F37A"] = "S"
    smileyLabelDict[u"\U0001F37B"] = "S"
    smileyLabelDict[u"\U0001F380"] = "S"
    smileyLabelDict[u"\U0001F381"] = "S"
    smileyLabelDict[u"\U0001F382"] = "S"
    smileyLabelDict[u"\U0001F383"] = "S"
    smileyLabelDict[u"\U0001F384"] = "S"
    smileyLabelDict[u"\U0001F385"] = "S"
    smileyLabelDict[u"\U0001F386"] = "S"
    smileyLabelDict[u"\U0001F387"] = "S"
    smileyLabelDict[u"\U0001F388"] = "S"
    smileyLabelDict[u"\U0001F389"] = "S"
    smileyLabelDict[u"\U0001F38A"] = "S"
    smileyLabelDict[u"\U0001F38B"] = "S"
    smileyLabelDict[u"\U0001F38C"] = "S"
    smileyLabelDict[u"\U0001F38D"] = "S"
    smileyLabelDict[u"\U0001F38E"] = "S"
    smileyLabelDict[u"\U0001F38F"] = "S"
    smileyLabelDict[u"\U0001F390"] = "S"
    smileyLabelDict[u"\U0001F391"] = "S"
    smileyLabelDict[u"\U0001F392"] = "S"
    smileyLabelDict[u"\U0001F393"] = "S"
    smileyLabelDict[u"\U0001F3A0"] = "S"
    smileyLabelDict[u"\U0001F3A1"] = "S"
    smileyLabelDict[u"\U0001F3A2"] = "S"
    smileyLabelDict[u"\U0001F3A3"] = "S"
    smileyLabelDict[u"\U0001F3A4"] = "S"
    smileyLabelDict[u"\U0001F3A5"] = "S"
    smileyLabelDict[u"\U0001F3A6"] = "S"
    smileyLabelDict[u"\U0001F3A7"] = "S"
    smileyLabelDict[u"\U0001F3A8"] = "S"
    smileyLabelDict[u"\U0001F3A9"] = "S"
    smileyLabelDict[u"\U0001F3AA"] = "S"
    smileyLabelDict[u"\U0001F3AB"] = "S"
    smileyLabelDict[u"\U0001F3AC"] = "S"
    smileyLabelDict[u"\U0001F3AD"] = "S"
    smileyLabelDict[u"\U0001F3AE"] = "S"
    smileyLabelDict[u"\U0001F3AF"] = "S"
    smileyLabelDict[u"\U0001F3B0"] = "S"
    smileyLabelDict[u"\U0001F3B1"] = "S"
    smileyLabelDict[u"\U0001F3B2"] = "S"
    smileyLabelDict[u"\U0001F3B3"] = "S"
    smileyLabelDict[u"\U0001F3B4"] = "S"
    smileyLabelDict[u"\U0001F3B5"] = "S"
    smileyLabelDict[u"\U0001F3B6"] = "S"
    smileyLabelDict[u"\U0001F3B7"] = "S"
    smileyLabelDict[u"\U0001F3B8"] = "S"
    smileyLabelDict[u"\U0001F3B9"] = "S"
    smileyLabelDict[u"\U0001F3BA"] = "S"
    smileyLabelDict[u"\U0001F3BB"] = "S"
    smileyLabelDict[u"\U0001F3BC"] = "S"
    smileyLabelDict[u"\U0001F3BD"] = "S"
    smileyLabelDict[u"\U0001F3BE"] = "S"
    smileyLabelDict[u"\U0001F3BF"] = "S"
    smileyLabelDict[u"\U0001F3C0"] = "S"
    smileyLabelDict[u"\U0001F3C1"] = "S"
    smileyLabelDict[u"\U0001F3C2"] = "S"
    smileyLabelDict[u"\U0001F3C3"] = "S"
    smileyLabelDict[u"\U0001F3C4"] = "S"
    smileyLabelDict[u"\U0001F3C6"] = "S"
    smileyLabelDict[u"\U0001F3C8"] = "S"
    smileyLabelDict[u"\U0001F3CA"] = "S"
    smileyLabelDict[u"\U0001F3E0"] = "S"
    smileyLabelDict[u"\U0001F3E1"] = "S"
    smileyLabelDict[u"\U0001F3E2"] = "S"
    smileyLabelDict[u"\U0001F3E3"] = "S"
    smileyLabelDict[u"\U0001F3E5"] = "S"
    smileyLabelDict[u"\U0001F3E6"] = "S"
    smileyLabelDict[u"\U0001F3E7"] = "S"
    smileyLabelDict[u"\U0001F3E8"] = "S"
    smileyLabelDict[u"\U0001F3E9"] = "S"
    smileyLabelDict[u"\U0001F3EA"] = "S"
    smileyLabelDict[u"\U0001F3EB"] = "S"
    smileyLabelDict[u"\U0001F3EC"] = "S"
    smileyLabelDict[u"\U0001F3ED"] = "S"
    smileyLabelDict[u"\U0001F3EE"] = "S"
    smileyLabelDict[u"\U0001F3EF"] = "S"
    smileyLabelDict[u"\U0001F3F0"] = "S"
    smileyLabelDict[u"\U0001F40C"] = "S"
    smileyLabelDict[u"\U0001F40D"] = "S"
    smileyLabelDict[u"\U0001F40E"] = "S"
    smileyLabelDict[u"\U0001F411"] = "S"
    smileyLabelDict[u"\U0001F412"] = "S"
    smileyLabelDict[u"\U0001F414"] = "S"
    smileyLabelDict[u"\U0001F417"] = "S"
    smileyLabelDict[u"\U0001F418"] = "S"
    smileyLabelDict[u"\U0001F419"] = "S"
    smileyLabelDict[u"\U0001F41A"] = "S"
    smileyLabelDict[u"\U0001F41B"] = "S"
    smileyLabelDict[u"\U0001F41C"] = "S"
    smileyLabelDict[u"\U0001F41D"] = "S"
    smileyLabelDict[u"\U0001F41E"] = "S"
    smileyLabelDict[u"\U0001F41F"] = "S"
    smileyLabelDict[u"\U0001F420"] = "S"
    smileyLabelDict[u"\U0001F421"] = "S"
    smileyLabelDict[u"\U0001F422"] = "S"
    smileyLabelDict[u"\U0001F423"] = "S"
    smileyLabelDict[u"\U0001F424"] = "S"
    smileyLabelDict[u"\U0001F425"] = "S"
    smileyLabelDict[u"\U0001F426"] = "S"
    smileyLabelDict[u"\U0001F427"] = "S"
    smileyLabelDict[u"\U0001F428"] = "S"
    smileyLabelDict[u"\U0001F429"] = "S"
    smileyLabelDict[u"\U0001F42B"] = "S"
    smileyLabelDict[u"\U0001F42C"] = "S"
    smileyLabelDict[u"\U0001F42D"] = "S"
    smileyLabelDict[u"\U0001F42E"] = "S"
    smileyLabelDict[u"\U0001F42F"] = "S"
    smileyLabelDict[u"\U0001F430"] = "S"
    smileyLabelDict[u"\U0001F431"] = "S"
    smileyLabelDict[u"\U0001F432"] = "S"
    smileyLabelDict[u"\U0001F433"] = "S"
    smileyLabelDict[u"\U0001F434"] = "S"
    smileyLabelDict[u"\U0001F435"] = "S"
    smileyLabelDict[u"\U0001F436"] = "S"
    smileyLabelDict[u"\U0001F437"] = "S"
    smileyLabelDict[u"\U0001F438"] = "S"
    smileyLabelDict[u"\U0001F439"] = "S"
    smileyLabelDict[u"\U0001F43A"] = "S"
    smileyLabelDict[u"\U0001F43B"] = "S"
    smileyLabelDict[u"\U0001F43C"] = "S"
    smileyLabelDict[u"\U0001F43D"] = "S"
    smileyLabelDict[u"\U0001F43E"] = "S"
    smileyLabelDict[u"\U0001F440"] = "S"
    smileyLabelDict[u"\U0001F442"] = "S"
    smileyLabelDict[u"\U0001F443"] = "S"
    smileyLabelDict[u"\U0001F444"] = "S"
    smileyLabelDict[u"\U0001F445"] = "S"
    smileyLabelDict[u"\U0001F446"] = "S"
    smileyLabelDict[u"\U0001F447"] = "S"
    smileyLabelDict[u"\U0001F448"] = "S"
    smileyLabelDict[u"\U0001F449"] = "S"
    smileyLabelDict[u"\U0001F44A"] = "S"
    smileyLabelDict[u"\U0001F44B"] = "S"
    smileyLabelDict[u"\U0001F44C"] = "S"
    smileyLabelDict[u"\U0001F44D"] = "S"
    smileyLabelDict[u"\U0001F44E"] = "S"
    smileyLabelDict[u"\U0001F44F"] = "S"
    smileyLabelDict[u"\U0001F450"] = "S"
    smileyLabelDict[u"\U0001F451"] = "S"
    smileyLabelDict[u"\U0001F452"] = "S"
    smileyLabelDict[u"\U0001F453"] = "S"
    smileyLabelDict[u"\U0001F454"] = "S"
    smileyLabelDict[u"\U0001F455"] = "S"
    smileyLabelDict[u"\U0001F456"] = "S"
    smileyLabelDict[u"\U0001F457"] = "S"
    smileyLabelDict[u"\U0001F458"] = "S"
    smileyLabelDict[u"\U0001F459"] = "S"
    smileyLabelDict[u"\U0001F45A"] = "S"
    smileyLabelDict[u"\U0001F45B"] = "S"
    smileyLabelDict[u"\U0001F45C"] = "S"
    smileyLabelDict[u"\U0001F45D"] = "S"
    smileyLabelDict[u"\U0001F45E"] = "S"
    smileyLabelDict[u"\U0001F45F"] = "S"
    smileyLabelDict[u"\U0001F460"] = "S"
    smileyLabelDict[u"\U0001F461"] = "S"
    smileyLabelDict[u"\U0001F462"] = "S"
    smileyLabelDict[u"\U0001F463"] = "S"
    smileyLabelDict[u"\U0001F464"] = "S"
    smileyLabelDict[u"\U0001F466"] = "S"
    smileyLabelDict[u"\U0001F467"] = "S"
    smileyLabelDict[u"\U0001F468"] = "S"
    smileyLabelDict[u"\U0001F469"] = "S"
    smileyLabelDict[u"\U0001F46A"] = "S"
    smileyLabelDict[u"\U0001F46B"] = "S"
    smileyLabelDict[u"\U0001F46E"] = "S"
    smileyLabelDict[u"\U0001F46F"] = "S"
    smileyLabelDict[u"\U0001F470"] = "S"
    smileyLabelDict[u"\U0001F471"] = "S"
    smileyLabelDict[u"\U0001F472"] = "S"
    smileyLabelDict[u"\U0001F473"] = "S"
    smileyLabelDict[u"\U0001F474"] = "S"
    smileyLabelDict[u"\U0001F475"] = "S"
    smileyLabelDict[u"\U0001F476"] = "S"
    smileyLabelDict[u"\U0001F477"] = "S"
    smileyLabelDict[u"\U0001F478"] = "S"
    smileyLabelDict[u"\U0001F479"] = "S"
    smileyLabelDict[u"\U0001F47A"] = "S"
    smileyLabelDict[u"\U0001F47B"] = "S"
    smileyLabelDict[u"\U0001F47C"] = "S"
    smileyLabelDict[u"\U0001F47D"] = "S"
    smileyLabelDict[u"\U0001F47E"] = "S"
    smileyLabelDict[u"\U0001F47F"] = "S"
    smileyLabelDict[u"\U0001F480"] = "S"
    smileyLabelDict[u"\U0001F481"] = "S"
    smileyLabelDict[u"\U0001F482"] = "S"
    smileyLabelDict[u"\U0001F483"] = "S"
    smileyLabelDict[u"\U0001F484"] = "S"
    smileyLabelDict[u"\U0001F485"] = "S"
    smileyLabelDict[u"\U0001F486"] = "S"
    smileyLabelDict[u"\U0001F487"] = "S"
    smileyLabelDict[u"\U0001F488"] = "S"
    smileyLabelDict[u"\U0001F489"] = "S"
    smileyLabelDict[u"\U0001F48A"] = "S"
    smileyLabelDict[u"\U0001F48B"] = "+"
    smileyLabelDict[u"\U0001F48C"] = "+"
    smileyLabelDict[u"\U0001F48D"] = "+"
    smileyLabelDict[u"\U0001F48E"] = "+"
    smileyLabelDict[u"\U0001F48F"] = "+"
    smileyLabelDict[u"\U0001F490"] = "+"
    smileyLabelDict[u"\U0001F491"] = "+"
    smileyLabelDict[u"\U0001F492"] = "+"
    smileyLabelDict[u"\U0001F493"] = "+"
    smileyLabelDict[u"\U0001F494"] = "-"
    smileyLabelDict[u"\U0001F495"] = "+"
    smileyLabelDict[u"\U0001F496"] = "+"
    smileyLabelDict[u"\U0001F497"] = "+"
    smileyLabelDict[u"\U0001F498"] = "+"
    smileyLabelDict[u"\U0001F499"] = "+"
    smileyLabelDict[u"\U0001F49A"] = "+"
    smileyLabelDict[u"\U0001F49B"] = "+"
    smileyLabelDict[u"\U0001F49C"] = "+"
    smileyLabelDict[u"\U0001F49D"] = "+"
    smileyLabelDict[u"\U0001F49E"] = "+"
    smileyLabelDict[u"\U0001F49F"] = "+"
    smileyLabelDict[u"\U0001F4A0"] = "S"
    smileyLabelDict[u"\U0001F4A1"] = "S"
    smileyLabelDict[u"\U0001F4A2"] = "-"
    smileyLabelDict[u"\U0001F4A3"] = "-"
    smileyLabelDict[u"\U0001F4A4"] = "-"
    smileyLabelDict[u"\U0001F4A5"] = "-"
    smileyLabelDict[u"\U0001F4A6"] = "S"
    smileyLabelDict[u"\U0001F4A7"] = "S"
    smileyLabelDict[u"\U0001F4A8"] = "S"
    smileyLabelDict[u"\U0001F4A9"] = "S"
    smileyLabelDict[u"\U0001F4AA"] = "S"
    smileyLabelDict[u"\U0001F4AB"] = "S"
    smileyLabelDict[u"\U0001F4AC"] = "S"
    smileyLabelDict[u"\U0001F4AE"] = "S"
    smileyLabelDict[u"\U0001F4AF"] = "S"
    smileyLabelDict[u"\U0001F4B0"] = "S"
    smileyLabelDict[u"\U0001F4B1"] = "S"
    smileyLabelDict[u"\U0001F4B2"] = "S"
    smileyLabelDict[u"\U0001F4B3"] = "S"
    smileyLabelDict[u"\U0001F4B4"] = "S"
    smileyLabelDict[u"\U0001F4B5"] = "S"
    smileyLabelDict[u"\U0001F4B8"] = "S"
    smileyLabelDict[u"\U0001F4B9"] = "S"
    smileyLabelDict[u"\U0001F4BA"] = "S"
    smileyLabelDict[u"\U0001F4BB"] = "S"
    smileyLabelDict[u"\U0001F4BC"] = "S"
    smileyLabelDict[u"\U0001F4BD"] = "S"
    smileyLabelDict[u"\U0001F4BE"] = "S"
    smileyLabelDict[u"\U0001F4BF"] = "S"
    smileyLabelDict[u"\U0001F4C0"] = "S"
    smileyLabelDict[u"\U0001F4C1"] = "S"
    smileyLabelDict[u"\U0001F4C2"] = "S"
    smileyLabelDict[u"\U0001F4C3"] = "S"
    smileyLabelDict[u"\U0001F4C4"] = "S"
    smileyLabelDict[u"\U0001F4C5"] = "S"
    smileyLabelDict[u"\U0001F4C6"] = "S"
    smileyLabelDict[u"\U0001F4C7"] = "S"
    smileyLabelDict[u"\U0001F4C8"] = "S"
    smileyLabelDict[u"\U0001F4C9"] = "S"
    smileyLabelDict[u"\U0001F4CA"] = "S"
    smileyLabelDict[u"\U0001F4CB"] = "S"
    smileyLabelDict[u"\U0001F4CC"] = "S"
    smileyLabelDict[u"\U0001F4CD"] = "S"
    smileyLabelDict[u"\U0001F4CE"] = "S"
    smileyLabelDict[u"\U0001F4CF"] = "S"
    smileyLabelDict[u"\U0001F4D0"] = "S"
    smileyLabelDict[u"\U0001F4D1"] = "S"
    smileyLabelDict[u"\U0001F4D2"] = "S"
    smileyLabelDict[u"\U0001F4D3"] = "S"
    smileyLabelDict[u"\U0001F4D4"] = "S"
    smileyLabelDict[u"\U0001F4D5"] = "S"
    smileyLabelDict[u"\U0001F4D6"] = "S"
    smileyLabelDict[u"\U0001F4D7"] = "S"
    smileyLabelDict[u"\U0001F4D8"] = "S"
    smileyLabelDict[u"\U0001F4D9"] = "S"
    smileyLabelDict[u"\U0001F4DA"] = "S"
    smileyLabelDict[u"\U0001F4DB"] = "S"
    smileyLabelDict[u"\U0001F4DC"] = "S"
    smileyLabelDict[u"\U0001F4DD"] = "S"
    smileyLabelDict[u"\U0001F4DE"] = "S"
    smileyLabelDict[u"\U0001F4DF"] = "S"
    smileyLabelDict[u"\U0001F4E0"] = "S"
    smileyLabelDict[u"\U0001F4E1"] = "S"
    smileyLabelDict[u"\U0001F4E2"] = "S"
    smileyLabelDict[u"\U0001F4E3"] = "S"
    smileyLabelDict[u"\U0001F4E4"] = "S"
    smileyLabelDict[u"\U0001F4E5"] = "S"
    smileyLabelDict[u"\U0001F4E6"] = "S"
    smileyLabelDict[u"\U0001F4E7"] = "S"
    smileyLabelDict[u"\U0001F4E8"] = "S"
    smileyLabelDict[u"\U0001F4E9"] = "S"
    smileyLabelDict[u"\U0001F4EA"] = "S"
    smileyLabelDict[u"\U0001F4EB"] = "S"
    smileyLabelDict[u"\U0001F4EE"] = "S"
    smileyLabelDict[u"\U0001F4F0"] = "S"
    smileyLabelDict[u"\U0001F4F1"] = "S"
    smileyLabelDict[u"\U0001F4F2"] = "S"
    smileyLabelDict[u"\U0001F4F3"] = "S"
    smileyLabelDict[u"\U0001F4F4"] = "S"
    smileyLabelDict[u"\U0001F4F6"] = "S"
    smileyLabelDict[u"\U0001F4F7"] = "S"
    smileyLabelDict[u"\U0001F4F9"] = "S"
    smileyLabelDict[u"\U0001F4FA"] = "S"
    smileyLabelDict[u"\U0001F4FB"] = "S"
    smileyLabelDict[u"\U0001F4FC"] = "S"
    smileyLabelDict[u"\U0001F503"] = "S"
    smileyLabelDict[u"\U0001F50A"] = "S"
    smileyLabelDict[u"\U0001F50B"] = "S"
    smileyLabelDict[u"\U0001F50C"] = "S"
    smileyLabelDict[u"\U0001F50D"] = "S"
    smileyLabelDict[u"\U0001F50E"] = "S"
    smileyLabelDict[u"\U0001F50F"] = "S"
    smileyLabelDict[u"\U0001F510"] = "S"
    smileyLabelDict[u"\U0001F511"] = "S"
    smileyLabelDict[u"\U0001F512"] = "S"
    smileyLabelDict[u"\U0001F513"] = "S"
    smileyLabelDict[u"\U0001F514"] = "S"
    smileyLabelDict[u"\U0001F516"] = "S"
    smileyLabelDict[u"\U0001F517"] = "S"
    smileyLabelDict[u"\U0001F518"] = "S"
    smileyLabelDict[u"\U0001F519"] = "S"
    smileyLabelDict[u"\U0001F51A"] = "S"
    smileyLabelDict[u"\U0001F51B"] = "S"
    smileyLabelDict[u"\U0001F51C"] = "S"
    smileyLabelDict[u"\U0001F51D"] = "S"
    smileyLabelDict[u"\U0001F51E"] = "S"
    smileyLabelDict[u"\U0001F51F"] = "S"
    smileyLabelDict[u"\U0001F520"] = "S"
    smileyLabelDict[u"\U0001F521"] = "S"
    smileyLabelDict[u"\U0001F522"] = "S"
    smileyLabelDict[u"\U0001F523"] = "S"
    smileyLabelDict[u"\U0001F524"] = "S"
    smileyLabelDict[u"\U0001F525"] = "S"
    smileyLabelDict[u"\U0001F526"] = "S"
    smileyLabelDict[u"\U0001F527"] = "S"
    smileyLabelDict[u"\U0001F528"] = "S"
    smileyLabelDict[u"\U0001F529"] = "S"
    smileyLabelDict[u"\U0001F52A"] = "S"
    smileyLabelDict[u"\U0001F52B"] = "S"
    smileyLabelDict[u"\U0001F52E"] = "S"
    smileyLabelDict[u"\U0001F52F"] = "S"
    smileyLabelDict[u"\U0001F530"] = "S"
    smileyLabelDict[u"\U0001F531"] = "S"
    smileyLabelDict[u"\U0001F532"] = "S"
    smileyLabelDict[u"\U0001F533"] = "S"
    smileyLabelDict[u"\U0001F534"] = "S"
    smileyLabelDict[u"\U0001F535"] = "S"
    smileyLabelDict[u"\U0001F536"] = "S"
    smileyLabelDict[u"\U0001F537"] = "S"
    smileyLabelDict[u"\U0001F538"] = "S"
    smileyLabelDict[u"\U0001F539"] = "S"
    smileyLabelDict[u"\U0001F53A"] = "S"
    smileyLabelDict[u"\U0001F53B"] = "S"
    smileyLabelDict[u"\U0001F53C"] = "S"
    smileyLabelDict[u"\U0001F53D"] = "S"
    smileyLabelDict[u"\U0001F550"] = "S"
    smileyLabelDict[u"\U0001F551"] = "S"
    smileyLabelDict[u"\U0001F552"] = "S"
    smileyLabelDict[u"\U0001F553"] = "S"
    smileyLabelDict[u"\U0001F554"] = "S"
    smileyLabelDict[u"\U0001F555"] = "S"
    smileyLabelDict[u"\U0001F556"] = "S"
    smileyLabelDict[u"\U0001F557"] = "S"
    smileyLabelDict[u"\U0001F558"] = "S"
    smileyLabelDict[u"\U0001F559"] = "S"
    smileyLabelDict[u"\U0001F55A"] = "S"
    smileyLabelDict[u"\U0001F55B"] = "S"
    smileyLabelDict[u"\U0001F5FB"] = "S"
    smileyLabelDict[u"\U0001F5FC"] = "S"
    smileyLabelDict[u"\U0001F5FD"] = "S"
    smileyLabelDict[u"\U0001F5FE"] = "S"
    smileyLabelDict[u"\U0001F5FF"] = "S"
    smileyLabelDict[u"\U0001F600"] = "+"
    smileyLabelDict[u"\U0001F607"] = "+"
    smileyLabelDict[u"\U0001F608"] = "+"
    smileyLabelDict[u"\U0001F60E"] = "+"
    smileyLabelDict[u"\U0001F610"] = "-"
    smileyLabelDict[u"\U0001F611"] = "-"
    smileyLabelDict[u"\U0001F615"] = "-"
    smileyLabelDict[u"\U0001F617"] = "+"
    smileyLabelDict[u"\U0001F619"] = "+"
    smileyLabelDict[u"\U0001F61B"] = "+"
    smileyLabelDict[u"\U0001F61F"] = "-"
    smileyLabelDict[u"\U0001F626"] = "-"
    smileyLabelDict[u"\U0001F627"] = "-"
    smileyLabelDict[u"\U0001F62C"] = "-"
    smileyLabelDict[u"\U0001F62E"] = "-"
    smileyLabelDict[u"\U0001F62F"] = "."
    smileyLabelDict[u"\U0001F634"] = "-"
    smileyLabelDict[u"\U0001F636"] = "-"
    smileyLabelDict[u"\U0001F681"] = "S"
    smileyLabelDict[u"\U0001F682"] = "S"
    smileyLabelDict[u"\U0001F686"] = "S"
    smileyLabelDict[u"\U0001F688"] = "S"
    smileyLabelDict[u"\U0001F68A"] = "S"
    smileyLabelDict[u"\U0001F68D"] = "S"
    smileyLabelDict[u"\U0001F68E"] = "S"
    smileyLabelDict[u"\U0001F690"] = "S"
    smileyLabelDict[u"\U0001F694"] = "S"
    smileyLabelDict[u"\U0001F696"] = "S"
    smileyLabelDict[u"\U0001F698"] = "S"
    smileyLabelDict[u"\U0001F69B"] = "S"
    smileyLabelDict[u"\U0001F69C"] = "S"
    smileyLabelDict[u"\U0001F69D"] = "S"
    smileyLabelDict[u"\U0001F69E"] = "S"
    smileyLabelDict[u"\U0001F69F"] = "S"
    smileyLabelDict[u"\U0001F6A0"] = "S"
    smileyLabelDict[u"\U0001F6A1"] = "S"
    smileyLabelDict[u"\U0001F6A3"] = "S"
    smileyLabelDict[u"\U0001F6A6"] = "S"
    smileyLabelDict[u"\U0001F6AE"] = "S"
    smileyLabelDict[u"\U0001F6AF"] = "S"
    smileyLabelDict[u"\U0001F6B0"] = "S"
    smileyLabelDict[u"\U0001F6B1"] = "S"
    smileyLabelDict[u"\U0001F6B3"] = "S"
    smileyLabelDict[u"\U0001F6B4"] = "S"
    smileyLabelDict[u"\U0001F6B5"] = "S"
    smileyLabelDict[u"\U0001F6B7"] = "S"
    smileyLabelDict[u"\U0001F6B8"] = "S"
    smileyLabelDict[u"\U0001F6BF"] = "S"
    smileyLabelDict[u"\U0001F6C1"] = "S"
    smileyLabelDict[u"\U0001F6C2"] = "S"
    smileyLabelDict[u"\U0001F6C3"] = "S"
    smileyLabelDict[u"\U0001F6C4"] = "S"
    smileyLabelDict[u"\U0001F6C5"] = "S"
    smileyLabelDict[u"\U0001F30D"] = "S"
    smileyLabelDict[u"\U0001F30E"] = "S"
    smileyLabelDict[u"\U0001F310"] = "S"
    smileyLabelDict[u"\U0001F312"] = "S"
    smileyLabelDict[u"\U0001F316"] = "S"
    smileyLabelDict[u"\U0001F317"] = "S"
    smileyLabelDict[u"\U0001F318"] = "S"
    smileyLabelDict[u"\U0001F31A"] = "S"
    smileyLabelDict[u"\U0001F31C"] = "S"
    smileyLabelDict[u"\U0001F31D"] = "S"
    smileyLabelDict[u"\U0001F31E"] = "S"
    smileyLabelDict[u"\U0001F332"] = "S"
    smileyLabelDict[u"\U0001F333"] = "S"
    smileyLabelDict[u"\U0001F34B"] = "S"
    smileyLabelDict[u"\U0001F350"] = "S"
    smileyLabelDict[u"\U0001F37C"] = "S"
    smileyLabelDict[u"\U0001F3C7"] = "S"
    smileyLabelDict[u"\U0001F3C9"] = "S"
    smileyLabelDict[u"\U0001F3E4"] = "S"
    smileyLabelDict[u"\U0001F400"] = "S"
    smileyLabelDict[u"\U0001F401"] = "S"
    smileyLabelDict[u"\U0001F402"] = "S"
    smileyLabelDict[u"\U0001F403"] = "S"
    smileyLabelDict[u"\U0001F404"] = "S"
    smileyLabelDict[u"\U0001F405"] = "S"
    smileyLabelDict[u"\U0001F406"] = "S"
    smileyLabelDict[u"\U0001F407"] = "S"
    smileyLabelDict[u"\U0001F408"] = "S"
    smileyLabelDict[u"\U0001F409"] = "S"
    smileyLabelDict[u"\U0001F40A"] = "S"
    smileyLabelDict[u"\U0001F40B"] = "S"
    smileyLabelDict[u"\U0001F40F"] = "S"
    smileyLabelDict[u"\U0001F410"] = "S"
    smileyLabelDict[u"\U0001F413"] = "S"
    smileyLabelDict[u"\U0001F415"] = "S"
    smileyLabelDict[u"\U0001F416"] = "S"
    smileyLabelDict[u"\U0001F42A"] = "S"
    smileyLabelDict[u"\U0001F465"] = "S"
    smileyLabelDict[u"\U0001F46C"] = "S"
    smileyLabelDict[u"\U0001F46D"] = "S"
    smileyLabelDict[u"\U0001F4AD"] = "S"
    smileyLabelDict[u"\U0001F4B6"] = "S"
    smileyLabelDict[u"\U0001F4B7"] = "S"
    smileyLabelDict[u"\U0001F4EC"] = "S"
    smileyLabelDict[u"\U0001F4ED"] = "S"
    smileyLabelDict[u"\U0001F4EF"] = "S"
    smileyLabelDict[u"\U0001F4F5"] = "S"
    smileyLabelDict[u"\U0001F500"] = "S"
    smileyLabelDict[u"\U0001F501"] = "S"
    smileyLabelDict[u"\U0001F502"] = "S"
    smileyLabelDict[u"\U0001F504"] = "S"
    smileyLabelDict[u"\U0001F505"] = "S"
    smileyLabelDict[u"\U0001F506"] = "S"
    smileyLabelDict[u"\U0001F507"] = "S"
    smileyLabelDict[u"\U0001F509"] = "S"
    smileyLabelDict[u"\U0001F515"] = "S"
    smileyLabelDict[u"\U0001F52C"] = "S"
    smileyLabelDict[u"\U0001F52D"] = "S"
    smileyLabelDict[u"\U0001F55C"] = "S"
    smileyLabelDict[u"\U0001F55D"] = "S"
    smileyLabelDict[u"\U0001F55E"] = "S"
    smileyLabelDict[u"\U0001F55F"] = "S"
    smileyLabelDict[u"\U0001F560"] = "S"
    smileyLabelDict[u"\U0001F561"] = "S"
    smileyLabelDict[u"\U0001F562"] = "S"
    smileyLabelDict[u"\U0001F563"] = "S"
    smileyLabelDict[u"\U0001F564"] = "S"
    smileyLabelDict[u"\U0001F565"] = "S"
    smileyLabelDict[u"\U0001F566"] = "S"
    smileyLabelDict[u"\U0001F567"] = "S"
    
# ---- ANFANG: sentiment smiley dict ------------
    sentiment_smiley_dict = dict()
    sentiment_smiley_dict[u"\U0001F601"] = "+"
    sentiment_smiley_dict[u"\U0001F602"] = "+"
    sentiment_smiley_dict[u"\U0001F603"] = "+"
    sentiment_smiley_dict[u"\U0001F604"] = "+"
    sentiment_smiley_dict[u"\U0001F605"] = "+"
    sentiment_smiley_dict[u"\U0001F606"] = "+"
    sentiment_smiley_dict[u"\U0001F609"] = "+"
    sentiment_smiley_dict[u"\U0001F60A"] = "+"
    sentiment_smiley_dict[u"\U0001F60B"] = "+"
    sentiment_smiley_dict[u"\U0001F60C"] = "+"
    sentiment_smiley_dict[u"\U0001F60D"] = "+"
    sentiment_smiley_dict[u"\U0001F60F"] = "-"
    sentiment_smiley_dict[u"\U0001F612"] = "-"
    sentiment_smiley_dict[u"\U0001F613"] = "-"
    sentiment_smiley_dict[u"\U0001F614"] = "-"
    sentiment_smiley_dict[u"\U0001F616"] = "-"
    sentiment_smiley_dict[u"\U0001F618"] = "-"
    sentiment_smiley_dict[u"\U0001F61A"] = "+"
    sentiment_smiley_dict[u"\U0001F61C"] = "+"
    sentiment_smiley_dict[u"\U0001F61D"] = "+"
    sentiment_smiley_dict[u"\U0001F61E"] = "-"
    sentiment_smiley_dict[u"\U0001F620"] = "-"
    sentiment_smiley_dict[u"\U0001F621"] = "-"
    sentiment_smiley_dict[u"\U0001F622"] = "-"
    sentiment_smiley_dict[u"\U0001F623"] = "-"
    sentiment_smiley_dict[u"\U0001F624"] = "+"
    sentiment_smiley_dict[u"\U0001F625"] = "-"
    sentiment_smiley_dict[u"\U0001F628"] = "-"
    sentiment_smiley_dict[u"\U0001F629"] = "-"
    sentiment_smiley_dict[u"\U0001F62A"] = "-"
    sentiment_smiley_dict[u"\U0001F62B"] = "-"
    sentiment_smiley_dict[u"\U0001F62D"] = "-"
    sentiment_smiley_dict[u"\U0001F630"] = "-"
    sentiment_smiley_dict[u"\U0001F631"] = "-"
    sentiment_smiley_dict[u"\U0001F632"] = "-"
    sentiment_smiley_dict[u"\U0001F633"] = "-"
    sentiment_smiley_dict[u"\U0001F635"] = "-"
    sentiment_smiley_dict[u"\U0001F637"] = "-"
    sentiment_smiley_dict[u"\U0001F638"] = "+"
    sentiment_smiley_dict[u"\U0001F639"] = "+"
    sentiment_smiley_dict[u"\U0001F63A"] = "+"
    sentiment_smiley_dict[u"\U0001F63B"] = "+"
    sentiment_smiley_dict[u"\U0001F63C"] = "+"
    sentiment_smiley_dict[u"\U0001F63D"] = "+"
    sentiment_smiley_dict[u"\U0001F63E"] = "-"
    sentiment_smiley_dict[u"\U0001F63F"] = "-"
    sentiment_smiley_dict[u"\U0001F640"] = "-"
    sentiment_smiley_dict[u"\U0001F645"] = "-"
    sentiment_smiley_dict[u"\U0001F646"] = "+"
    sentiment_smiley_dict[u"\U0001F647"] = "-"
    sentiment_smiley_dict[u"\U0001F648"] = "-"
    sentiment_smiley_dict[u"\U0001F649"] = "-"
    sentiment_smiley_dict[u"\U0001F64A"] = "-"
    sentiment_smiley_dict[u"\U0001F64B"] = "+"
    sentiment_smiley_dict[u"\U0001F64C"] = "+"
    sentiment_smiley_dict[u"\U0001F64D"] = "-"
    sentiment_smiley_dict[u"\U0001F64E"] = "-"
    sentiment_smiley_dict[u"\U0001F64F"] = "-"
    sentiment_smiley_dict[u"\U0000270A"] = "-"
    sentiment_smiley_dict[u"\U0000270B"] = "-"
    sentiment_smiley_dict[u"\U0000270C"] = "+"
    sentiment_smiley_dict[u"\U00002714"] = "+"
    sentiment_smiley_dict[u"\U00002716"] = "-"
    sentiment_smiley_dict[u"\U0000274C"] = "-"
    sentiment_smiley_dict[u"\U0000274E"] = "-"
    sentiment_smiley_dict[u"\U00002753"] = "-"
    sentiment_smiley_dict[u"\U00002754"] = "-"
    sentiment_smiley_dict[u"\U00002755"] = "-"
    sentiment_smiley_dict[u"\U00002757"] = "-"
    sentiment_smiley_dict[u"\U00002764"] = "+"
    sentiment_smiley_dict[u"\U00002795"] = "+"
    sentiment_smiley_dict[u"\U00002796"] = "-"
    sentiment_smiley_dict[u"\U0001F6A7"] = "-"
    sentiment_smiley_dict[u"\U0001F6A8"] = "-"
    sentiment_smiley_dict[u"\U0001F192"] = "+"
    sentiment_smiley_dict[u"\U0001F193"] = "+"
    sentiment_smiley_dict[u"\U0001F195"] = "+"
    sentiment_smiley_dict[u"\U0001F197"] = "+"
    sentiment_smiley_dict[u"\U0001F198"] = "-"
    sentiment_smiley_dict[u"\U0000203C"] = "-"
    sentiment_smiley_dict[u"\U00002049"] = "-"
    sentiment_smiley_dict[u"\U0001F48B"] = "+"
    sentiment_smiley_dict[u"\U0001F48C"] = "+"
    sentiment_smiley_dict[u"\U0001F48D"] = "+"
    sentiment_smiley_dict[u"\U0001F48E"] = "+"
    sentiment_smiley_dict[u"\U0001F48F"] = "+"
    sentiment_smiley_dict[u"\U0001F490"] = "+"
    sentiment_smiley_dict[u"\U0001F491"] = "+"
    sentiment_smiley_dict[u"\U0001F492"] = "+"
    sentiment_smiley_dict[u"\U0001F493"] = "+"
    sentiment_smiley_dict[u"\U0001F494"] = "-"
    sentiment_smiley_dict[u"\U0001F495"] = "+"
    sentiment_smiley_dict[u"\U0001F496"] = "+"
    sentiment_smiley_dict[u"\U0001F497"] = "+"
    sentiment_smiley_dict[u"\U0001F498"] = "+"
    sentiment_smiley_dict[u"\U0001F499"] = "+"
    sentiment_smiley_dict[u"\U0001F49A"] = "+"
    sentiment_smiley_dict[u"\U0001F49B"] = "+"
    sentiment_smiley_dict[u"\U0001F49C"] = "+"
    sentiment_smiley_dict[u"\U0001F49D"] = "+"
    sentiment_smiley_dict[u"\U0001F49E"] = "+"
    sentiment_smiley_dict[u"\U0001F49F"] = "+"
    sentiment_smiley_dict[u"\U0001F4A2"] = "-"
    sentiment_smiley_dict[u"\U0001F4A3"] = "-"
    sentiment_smiley_dict[u"\U0001F4A4"] = "-"
    sentiment_smiley_dict[u"\U0001F4A5"] = "-"
    sentiment_smiley_dict[u"\U0001F600"] = "+"
    sentiment_smiley_dict[u"\U0001F607"] = "+"
    sentiment_smiley_dict[u"\U0001F608"] = "+"
    sentiment_smiley_dict[u"\U0001F60E"] = "+"
    sentiment_smiley_dict[u"\U0001F610"] = "-"
    sentiment_smiley_dict[u"\U0001F611"] = "-"
    sentiment_smiley_dict[u"\U0001F615"] = "-"
    sentiment_smiley_dict[u"\U0001F617"] = "+"
    sentiment_smiley_dict[u"\U0001F619"] = "+"
    sentiment_smiley_dict[u"\U0001F61B"] = "+"
    sentiment_smiley_dict[u"\U0001F61F"] = "-"
    sentiment_smiley_dict[u"\U0001F626"] = "-"
    sentiment_smiley_dict[u"\U0001F627"] = "-"
    sentiment_smiley_dict[u"\U0001F62C"] = "-"
    sentiment_smiley_dict[u"\U0001F62E"] = "-"
    sentiment_smiley_dict[u"\U0001F634"] = "-"
    sentiment_smiley_dict[u"\U0001F636"] = "-"
    
    return all_smileys, smileyLabelDict, sentiment_smiley_dict

# -------------------- Features ----------------------------------------------
def numberOfSmileys(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text in allSmileys:
            numberOfSmiley += 1
    
    
    return numberOfSmiley

def numberOfSmileys_01(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text in allSmileys:
            numberOfSmiley += 1
    
    return numberOfSmiley/float(numberOfWords)


def numberOfSmileys_stack1(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text in allSmileys:
            numberOfSmiley += 1
            
    value = numberOfSmiley/float(numberOfWords)
    if value > 0:
        return 1
    else:
        return 0

def numberOfSmileys_stack2(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text in allSmileys:
            numberOfSmiley += 1
            
    value = numberOfSmiley/float(numberOfWords)
    if value > 0.2:
        return 1
    else:
        return 0

def numberOfSmileys_stack3(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text in allSmileys:
            numberOfSmiley += 1
            
    value = numberOfSmiley/float(numberOfWords)
    if value > 0.4:
        return 1
    else:
        return 0

def numberOfSmileys_stack4(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text in allSmileys:
            numberOfSmiley += 1
            
    value = numberOfSmiley/float(numberOfWords)
    if value > 0.6:
        return 1
    else:
        return 0

def numberOfSmileys_stack5(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text in allSmileys:
            numberOfSmiley += 1
            
    value = numberOfSmiley/float(numberOfWords)
    if value > 0.8:
        return 1
    else:
        return 0

def numberOfFollowingSmileys(tweet):
    """ searches for longest sequence of smileys in tweet and returns its length. """
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    max = 0
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text not in allSmileys and i+1 == numberOfWords:
            break
        elif tweet.words[i].text in allSmileys and i+1 == numberOfWords:
            numberOfSmiley += 1   
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text in allSmileys:
            numberOfSmiley += 1
            continue
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text not in allSmileys:
            if numberOfSmiley + 1 > max:
                max = numberOfSmiley + 1
            numberOfSmiley = 0
        else:
            continue
            
    return max

def numberOfFollowingSmileys_01(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    max = 0
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text not in allSmileys and i+1 == numberOfWords:
            break
        elif tweet.words[i].text in allSmileys and i+1 == numberOfWords:
            numberOfSmiley += 1   
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text in allSmileys:
            numberOfSmiley += 1
            continue
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text not in allSmileys:
            if numberOfSmiley + 1 > max:
                max = numberOfSmiley + 1
            numberOfSmiley = 0
        else:
            continue
            
    return max/float(numberOfWords)

def numberOfFollowingSmileys_stack1(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    max = 0
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text not in allSmileys and i+1 == numberOfWords:
            break
        elif tweet.words[i].text in allSmileys and i+1 == numberOfWords:
            numberOfSmiley += 1   
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text in allSmileys:
            numberOfSmiley += 1
            continue
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text not in allSmileys:
            if numberOfSmiley + 1 > max:
                max = numberOfSmiley + 1
            numberOfSmiley = 0
        else:
            continue
    
    value = max/float(numberOfWords)    
    if value > 0:
        return 1
    else:  
        return 0

def numberOfFollowingSmileys_stack2(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    max = 0
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text not in allSmileys and i+1 == numberOfWords:
            break
        elif tweet.words[i].text in allSmileys and i+1 == numberOfWords:
            numberOfSmiley += 1   
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text in allSmileys:
            numberOfSmiley += 1
            continue
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text not in allSmileys:
            if numberOfSmiley + 1 > max:
                max = numberOfSmiley + 1
            numberOfSmiley = 0
        else:
            continue
            
    value = max/float(numberOfWords)    
    if value > 0.2:
        return 1
    else:  
        return 0

def numberOfFollowingSmileys_stack3(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    max = 0
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text not in allSmileys and i+1 == numberOfWords:
            break
        elif tweet.words[i].text in allSmileys and i+1 == numberOfWords:
            numberOfSmiley += 1   
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text in allSmileys:
            numberOfSmiley += 1
            continue
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text not in allSmileys:
            if numberOfSmiley + 1 > max:
                max = numberOfSmiley + 1
            numberOfSmiley = 0
        else:
            continue
            
    value = max/float(numberOfWords)    
    if value > 0.4:
        return 1
    else:  
        return 0

def numberOfFollowingSmileys_stack4(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    max = 0
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text not in allSmileys and i+1 == numberOfWords:
            break
        elif tweet.words[i].text in allSmileys and i+1 == numberOfWords:
            numberOfSmiley += 1   
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text in allSmileys:
            numberOfSmiley += 1
            continue
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text not in allSmileys:
            if numberOfSmiley + 1 > max:
                max = numberOfSmiley + 1
            numberOfSmiley = 0
        else:
            continue
            
    value = max/float(numberOfWords)    
    if value > 0.6:
        return 1
    else:  
        return 0

def numberOfFollowingSmileys_stack5(tweet):
    allSmileys,_ ,_ = loadSmileys()
    
    numberOfSmiley = 0
    max = 0
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text not in allSmileys and i+1 == numberOfWords:
            break
        elif tweet.words[i].text in allSmileys and i+1 == numberOfWords:
            numberOfSmiley += 1   
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text in allSmileys:
            numberOfSmiley += 1
            continue
        elif tweet.words[i].text in allSmileys and tweet.words[i+1].text not in allSmileys:
            if numberOfSmiley + 1 > max:
                max = numberOfSmiley + 1
            numberOfSmiley = 0
        else:
            continue
            
    value = max/float(numberOfWords)    
    if value > 0.8:
        return 1
    else:  
        return 0

def sentiment_smileys_pos(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    pos = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+":
                pos += 1
    
    return pos

def sentiment_smileys_pos_01(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    pos = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+":
                pos += 1
    
    return pos/float(numberOfWords)

def sentiment_smileys_pos_stack1(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    pos = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+":
                pos += 1
    
    value = pos/float(numberOfWords)
    if value > 0:
        return 1
    else:
        return 0

def sentiment_smileys_pos_stack2(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    pos = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+":
                pos += 1
    
    value = pos/float(numberOfWords)
    if value > 0.2:
        return 1
    else:
        return 0

def sentiment_smileys_pos_stack3(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    pos = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+":
                pos += 1
    
    value = pos/float(numberOfWords)
    if value > 0.4:
        return 1
    else:
        return 0

def sentiment_smileys_pos_stack4(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    pos = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+":
                pos += 1
    
    value = pos/float(numberOfWords)
    if value > 0.6:
        return 1
    else:
        return 0

def sentiment_smileys_pos_stack5(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    pos = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+":
                pos += 1
    
    value = pos/float(numberOfWords)
    if value > 0.8:
        return 1
    else:
        return 0   

def sentiment_smileys_neg(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    neg = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "-":
                neg += 1
    
    return neg

def sentiment_smileys_neg_01(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    neg = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "-":
                neg += 1
    
    return neg/float(numberOfWords)

def sentiment_smileys_neg_stack1(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    neg = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "-":
                neg += 1
    
    value = neg/float(numberOfWords)
    if value > 0:
        return 1
    else:
        return 0

def sentiment_smileys_neg_stack2(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    neg = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "-":
                neg += 1
    
    value = neg/float(numberOfWords)
    if value > 0.2:
        return 1
    else:
        return 0 

def sentiment_smileys_neg_stack3(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    neg = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "-":
                neg += 1
    
    value = neg/float(numberOfWords)
    if value > 0.4:
        return 1
    else:
        return 0 
    
def sentiment_smileys_neg_stack4(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    neg = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "-":
                neg += 1
    
    value = neg/float(numberOfWords)
    if value > 0.6:
        return 1
    else:
        return 0

def sentiment_smileys_neg_stack5(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    neg = 0
    
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "-":
                neg += 1
    
    value = neg/float(numberOfWords)
    if value > 0.8:
        return 1
    else:
        return 0  


def sentiment_smileys_01(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
             
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+":
                return 1
            elif label == "0":
                return 0.5
            elif label == "-":
                return 0
    
    return 0

def sentiment_smileys_stack1(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    count = 0       
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+" or label == "-" or label == "0":
                count += 1
    
    value = count/float(numberOfWords)
    if value > 0:
        return 1
    else:
        return 0

def sentiment_smileys_stack2(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    count = 0       
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+" or label == "-" or label == "0":
                count += 1
    
    value = count/float(numberOfWords)
    if value > 0.2:
        return 1
    else:
        return 0
    
def sentiment_smileys_stack3(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    count = 0       
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+" or label == "-" or label == "0":
                count += 1
    
    value = count/float(numberOfWords)
    if value > 0.4:
        return 1
    else:
        return 0
    
def sentiment_smileys_stack4(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    count = 0       
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+" or label == "-" or label == "0":
                count += 1
    
    value = count/float(numberOfWords)
    if value > 0.6:
        return 1
    else:
        return 0
    
def sentiment_smileys_stack5(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
    count = 0       
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+" or label == "-" or label == "0":
                count += 1
    
    value = count/float(numberOfWords)
    if value > 0.8:
        return 1
    else:
        return 0


def sentiment_smileys_pos_binary(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
             
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "+":
                return 1
            else:
                return 0
    
    return 0

def sentiment_smileys_neg_binary(tweet):
    _, _, smileyDict = loadSmileys()
        
    numberOfWords = len(tweet.words)
             
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label == "-":
                return 1
            else:
                return 0
    
    return 0
    
def sentence_length_gap(tweet):
    """ absolute gap between longest and shortest sentence of tweet """
    if len(tweet.sentences) <= 1:
        return 0
    else:
        max_length = 0
        min_length = len(tweet.sentences[0].words)
                
        for sent in tweet.sentences:
            
            if len(sent.words) > max_length:
                max_length = len(sent.words)
            if len(sent.words) < min_length:
                min_length = len(sent.words)

        gap = max_length - min_length
        
        return gap

def sentence_length_gap_01(tweet):
    numberOfWords = len(tweet.words)
    
    if len(tweet.sentences) <= 1:
        return 0
    else:
        max_length = 0
        min_length = len(tweet.sentences[0].words)
                
        for sent in tweet.sentences:
            
            if len(sent.words) > max_length:
                max_length = len(sent.words)
            if len(sent.words) < min_length:
                min_length = len(sent.words)

        gap = max_length - min_length
        
        return gap/float(numberOfWords)

def sentence_length_gap_stack1(tweet):
    
    numberOfWords = len(tweet.words)

    if len(tweet.sentences) <= 1:
        return 0
    else:
        max_length = 0
        min_length = len(tweet.sentences[0].words)
                
        for sent in tweet.sentences:
            
            if len(sent.words) > max_length:
                max_length = len(sent.words)
            if len(sent.words) < min_length:
                min_length = len(sent.words)

        gap = max_length - min_length
        
        value = gap/float(numberOfWords) 
        if value > 0:
            return 1
        else:
            return 0

def sentence_length_gap_stack2(tweet):
    
    numberOfWords = len(tweet.words)
    
    if len(tweet.sentences) <= 1:
        return 0
    else:
        max_length = 0
        min_length = len(tweet.sentences[0].words)
                
        for sent in tweet.sentences:
            
            if len(sent.words) > max_length:
                max_length = len(sent.words)
            if len(sent.words) < min_length:
                min_length = len(sent.words)

        gap = max_length - min_length
        
        value = gap/float(numberOfWords)  
        if value > 0.2:
            return 1
        else:
            return 0

def sentence_length_gap_stack3(tweet):
    
    numberOfWords = len(tweet.words)
    
    if len(tweet.sentences) <= 1:
        return 0
    else:
        max_length = 0
        min_length = len(tweet.sentences[0].words)
                
        for sent in tweet.sentences:
            
            if len(sent.words) > max_length:
                max_length = len(sent.words)
            if len(sent.words) < min_length:
                min_length = len(sent.words)

        gap = max_length - min_length
        
        value = gap/float(numberOfWords)  
        if value > 0.4:
            return 1
        else:
            return 0

def sentence_length_gap_stack4(tweet):
    
    numberOfWords = len(tweet.words)
    
    if len(tweet.sentences) <= 1:
        return 0
    else:
        max_length = 0
        min_length = len(tweet.sentences[0].words)
                
        for sent in tweet.sentences:
            
            if len(sent.words) > max_length:
                max_length = len(sent.words)
            if len(sent.words) < min_length:
                min_length = len(sent.words)

        gap = max_length - min_length
        
        value = gap/float(numberOfWords)  
        if value > 0.6:
            return 1
        else:
            return 0

def sentence_length_gap_stack5(tweet):
    
    numberOfWords = len(tweet.words)
    
    if len(tweet.sentences) <= 1:
        return 0
    else:
        max_length = 0
        min_length = len(tweet.sentences[0].words)
                
        for sent in tweet.sentences:
            
            if len(sent.words) > max_length:
                max_length = len(sent.words)
            if len(sent.words) < min_length:
                min_length = len(sent.words)

        gap = max_length - min_length
        
        value = gap/float(numberOfWords)  
        if value > 0.8:
            return 1
        else:
            return 0

def subjective_pronomina(tweet):
    pronomina = ["I", "we", "my", "our", "me", "mine", "us", "ours"] #Ich/Wir, mein/unser, mir, mich"
    
    numberOfPronomina = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text.lower() in pronomina:
            numberOfPronomina += 1
            
    return numberOfPronomina

def subjective_pronomina_01(tweet):
    pronomina = ["I", "we", "my", "our", "me", "mine", "us", "ours"] #Ich/Wir, mein/unser, mir, mich"
    
    numberOfPronomina = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text.lower() in pronomina:
            numberOfPronomina += 1
            
    return numberOfPronomina/float(numberOfWords)

def subjective_pronomina_stack1(tweet):
    pronomina = ["I", "we", "my", "our", "me", "mine", "us", "ours"] #Ich/Wir, mein/unser, mir, mich"
    
    numberOfPronomina = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text.lower() in pronomina:
            numberOfPronomina += 1
    
    value = numberOfPronomina/float(numberOfWords)
    if value > 0:
        return 1
    else:
        return 0

def subjective_pronomina_stack2(tweet):
    pronomina = ["I", "we", "my", "our", "me", "mine", "us", "ours"] #Ich/Wir, mein/unser, mir, mich"
    
    numberOfPronomina = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text.lower() in pronomina:
            numberOfPronomina += 1
    
    value = numberOfPronomina/float(numberOfWords)
    if value > 0.2:
        return 1
    else:
        return 0

def subjective_pronomina_stack3(tweet):
    pronomina = ["I", "we", "my", "our", "me", "mine", "us", "ours"] #Ich/Wir, mein/unser, mir, mich"
    
    numberOfPronomina = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text.lower() in pronomina:
            numberOfPronomina += 1
    
    value = numberOfPronomina/float(numberOfWords)
    if value > 0.4:
        return 1
    else:
        return 0  

def subjective_pronomina_stack4(tweet):
    pronomina = ["I", "we", "my", "our", "me", "mine", "us", "ours"] #Ich/Wir, mein/unser, mir, mich"
    
    numberOfPronomina = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text.lower() in pronomina:
            numberOfPronomina += 1
    
    value = numberOfPronomina/float(numberOfWords)
    if value > 0.6:
        return 1
    else:
        return 0

def subjective_pronomina_stack5(tweet):
    pronomina = ["I", "we", "my", "our", "me", "mine", "us", "ours"] #Ich/Wir, mein/unser, mir, mich"
    
    numberOfPronomina = 0
    
    numberOfWords = len(tweet.words)
         
    for i in range(numberOfWords):
        if tweet.words[i].text.lower() in pronomina:
            numberOfPronomina += 1
    
    value = numberOfPronomina/float(numberOfWords)
    if value > 0.8:
        return 1
    else:
        return 0        

def noun_pos_tag_01(tweet):
        
    noun_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("NN"):
            noun_count += 1
            
    return noun_count/float(numberOfWords)

def noun_pos_tag(tweet):
        
    noun_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("NN"):
            noun_count += 1
            
    
    return noun_count

def noun_pos_tag_stack1(tweet):
        
    noun_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("NN"):
            noun_count += 1
    
    value = noun_count/float(numberOfWords)
    if value > 0:
        return 1
    else:
        return 0

def noun_pos_tag_stack2(tweet):
        
    noun_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("NN"):
            noun_count += 1
            
    value = noun_count/float(numberOfWords)
    if value > 0.2:
        return 1
    else:
        return 0

def noun_pos_tag_stack3(tweet):
        
    noun_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("NN"):
            noun_count += 1
            
    value = noun_count/float(numberOfWords)
    if value > 0.4:
        return 1
    else:
        return 0

def noun_pos_tag_stack4(tweet):
        
    noun_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("NN"):
            noun_count += 1
            
    value = noun_count/float(numberOfWords)
    if value > 0.6:
        return 1
    else:
        return 0

def noun_pos_tag_stack5(tweet):
        
    noun_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("NN"):
            noun_count += 1
            
    value = noun_count/float(numberOfWords)
    if value > 0.8:
        return 1
    else:
        return 0
 
def verb_pos_tag_01(tweet):
        
    verb_count = 0
     
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("V"):
            verb_count += 1
     
    
    return verb_count/float(numberOfWords)

def verb_pos_tag(tweet):
        
    verb_count = 0
     
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("V"):
            verb_count += 1
     
    return verb_count

def verb_pos_tag_stack1(tweet):
        
    verb_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("V"):
            verb_count += 1
    
    value = verb_count/float(numberOfWords)        
    if value > 0:
        return 1
    else:
        return 0

def verb_pos_tag_stack2(tweet):
        
    verb_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("V"):
            verb_count += 1
            
    value = verb_count/float(numberOfWords)        
    if value > 0.2:
        return 1
    else:
        return 0

def verb_pos_tag_stack3(tweet):
        
    verb_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("V"):
            verb_count += 1
            
    value = verb_count/float(numberOfWords)        
    if value > 0.4:
        return 1
    else:
        return 0

def verb_pos_tag_stack4(tweet):
        
    verb_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("V"):
            verb_count += 1
            
    value = verb_count/float(numberOfWords)        
    if value > 0.6:
        return 1
    else:
        return 0
 
def verb_pos_tag_stack5(tweet):
        
    verb_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("V"):
            verb_count += 1
            
    value = verb_count/float(numberOfWords)        
    if value > 0.8:
        return 1
    else:
        return 0

def adj_pos_tag(tweet):
        
    adj_count = 0
     
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("JJ"):
            adj_count += 1
     
    return adj_count

def adj_pos_tag_01(tweet):
        
    adj_count = 0
     
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("JJ"):
            adj_count += 1
     
    return adj_count/float(numberOfWords)

def adj_pos_tag_stack1(tweet):
        
    adj_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("JJ"):
            adj_count += 1
    
    value = adj_count/float(numberOfWords)        
    if value > 0:
        return 1
    else:
        return 0

def adj_pos_tag_stack2(tweet):
        
    adj_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("JJ"):
            adj_count += 1
            
    value = adj_count/float(numberOfWords)        
    if value > 0.2:
        return 1
    else:
        return 0

def adj_pos_tag_stack3(tweet):
        
    adj_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("JJ"):
            adj_count += 1
            
    value = adj_count/float(numberOfWords)        
    if value > 0.4:
        return 1
    else:
        return 0

def adj_pos_tag_stack4(tweet):
        
    adj_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("JJ"):
            adj_count += 1
            
    value = adj_count/float(numberOfWords)        
    if value > 0.6:
        return 1
    else:
        return 0

def adj_pos_tag_stack5(tweet):
        
    adj_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("JJ"):
            adj_count += 1
            
    value = adj_count/float(numberOfWords)        
    if value > 0.8:
        return 1
    else:
        return 0

def adv_pos_tag(tweet):
        
    adv_count = 0
     
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("RB"):
            adv_count += 1
     
    return adv_count

def adv_pos_tag_01(tweet):
        
    adv_count = 0
     
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("RB"):
            adv_count += 1
     
    return adv_count/float(numberOfWords)

def adv_pos_tag_stack1(tweet):
        
    adv_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("RB"):
            adv_count += 1
    
    value = adv_count/float(numberOfWords)     
    if value > 0:
        return 1
    else:
        return 0

def adv_pos_tag_stack2(tweet):
        
    adv_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("RB"):
            adv_count += 1
            
    value = adv_count/float(numberOfWords)     
    if value > 0.2:
        return 1
    else:
        return 0

def adv_pos_tag_stack3(tweet):
        
    adv_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("RB"):
            adv_count += 1
            
    value = adv_count/float(numberOfWords)     
    if value > 0.4:
        return 1
    else:
        return 0

def adv_pos_tag_stack4(tweet):
        
    adv_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("RB"):
            adv_count += 1
            
    value = adv_count/float(numberOfWords)     
    if value > 0.6:
        return 1
    else:
        return 0

def adv_pos_tag_stack5(tweet):
        
    adv_count = 0
    
    numberOfWords = len(tweet.taggedTweet)
    
    for i in range(numberOfWords):
        pos = tweet.taggedTweet[i].pos
        if pos.startswith("RB"):
            adv_count += 1
            
    value = adv_count/float(numberOfWords)     
    if value > 0.8:
        return 1
    else:
        return 0

def sentiment_score_gap(tweet):
      
    numberOfWords = len(tweet.words)
      
    score = 0
      
    min_score = 0
    max_score = 0
      
    score_dict = loadSentimentScore()
     
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode('utf-8')
        
        if word in score_dict.keys():
            score = score_dict[word]
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score
    
    gap = abs(max_score - min_score)
    
    return gap

def sentiment_score_gap_01(tweet):
      
    numberOfWords = len(tweet.words)
      
    score = 0
      
    min_score = 0
    max_score = 0
      
    score_dict = loadSentimentScore()
     
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode('utf-8')
        
        if word in score_dict.keys():
            score = score_dict[word]
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score
    
    gap = abs(max_score - min_score)
    
    return gap/float(max_sentiment_score_gap)

def sentiment_score_gap_stack1(tweet):
      
    numberOfWords = len(tweet.words)
      
    score = 0
      
    min_score = 0
    max_score = 0
      
    score_dict = loadSentimentScore()
     
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode('utf-8')
        
        if word in score_dict.keys():
            score = score_dict[word]
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score
    
         
    gap = abs(max_score - min_score)
    
    value = gap/float(max_sentiment_score_gap)
    if value > 0:
        return 1
    else:
        return 0

def sentiment_score_gap_stack2(tweet):
      
    numberOfWords = len(tweet.words)
      
    score = 0
      
    min_score = 0
    max_score = 0
      
    score_dict = loadSentimentScore()
     
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode('utf-8')
        
        if word in score_dict.keys():
            score = score_dict[word]
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score
    
         
    gap = abs(max_score - min_score)
    value = gap/float(max_sentiment_score_gap)
    if value > 0.2:
        return 1
    else:
        return 0 

def sentiment_score_gap_stack3(tweet):
      
    numberOfWords = len(tweet.words)
      
    score = 0
      
    min_score = 0
    max_score = 0
      
    score_dict = loadSentimentScore()
     
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode('utf-8')
        
        if word in score_dict.keys():
            score = score_dict[word]
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score
    
         
    gap = abs(max_score - min_score)
    
    value = gap/float(max_sentiment_score_gap)
    if value > 0.4:
        return 1
    else:
        return 0 

def sentiment_score_gap_stack4(tweet):
      
    numberOfWords = len(tweet.words)
      
    score = 0
      
    min_score = 0
    max_score = 0
      
    score_dict = loadSentimentScore()
     
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode('utf-8')
        
        if word in score_dict.keys():
            score = score_dict[word]
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score
    
         
    gap = abs(max_score - min_score)
    
    value = gap/float(max_sentiment_score_gap)
    if value > 0.6:
        return 1
    else:
        return 0 

def sentiment_score_gap_stack5(tweet):
      
    numberOfWords = len(tweet.words)
      
    score = 0
      
    min_score = 0
    max_score = 0
      
    score_dict = loadSentimentScore()
     
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode('utf-8')
        
        if word in score_dict.keys():
            score = score_dict[word]
            if score < min_score:
                min_score = score
            if score > max_score:
                max_score = score
    
         
    gap = abs(max_score - min_score)
    
    value = gap/float(max_sentiment_score_gap)
    if value > 0.8:
        return 1
    else:
        return 0 

def stopwords(tweet):
    stopword_count = 0
    stopword_list = loadStopwords()
    numberOfWords = len(tweet.words)
    for i in range(numberOfWords):
        if tweet.words[i].text in stopword_list:
            stopword_count += 1
         
    return stopword_count

def stopwords_01(tweet):
    stopword_count = 0
    stopword_list = loadStopwords()
    numberOfWords = len(tweet.words)
    for i in range(numberOfWords):
        if tweet.words[i].text in stopword_list:
            stopword_count += 1
         
    return stopword_count/float(numberOfWords)

def stopwords_stack1(tweet):
    stopword_count = 0
    stopword_list = loadStopwords()
    numberOfWords = len(tweet.words)
    for i in range(numberOfWords):
        if tweet.words[i].text in stopword_list:
            stopword_count += 1
    
    value = stopword_count/float(numberOfWords)
    if value > 0:
        return 1
    else:
        return 0

def stopwords_stack2(tweet):
    stopword_count = 0
    stopword_list = loadStopwords()
    numberOfWords = len(tweet.words)
    for i in range(numberOfWords):
        if tweet.words[i].text in stopword_list:
            stopword_count += 1
    
    value = stopword_count/float(numberOfWords)
    if value > 0.2:
        return 1
    else:
        return 0

def stopwords_stack3(tweet):
    stopword_count = 0
    stopword_list = loadStopwords()
    numberOfWords = len(tweet.words)
    for i in range(numberOfWords):
        if tweet.words[i].text in stopword_list:
            stopword_count += 1
    
    value = stopword_count/float(numberOfWords)
    if value > 0.4:
        return 1
    else:
        return 0
    
def stopwords_stack4(tweet):
    stopword_count = 0
    stopword_list = loadStopwords()
    numberOfWords = len(tweet.words)
    for i in range(numberOfWords):
        if tweet.words[i].text in stopword_list:
            stopword_count += 1
    
    value = stopword_count/float(numberOfWords)
    if value > 0.6:
        return 1
    else:
        return 0
    
def stopwords_stack5(tweet):
    stopword_count = 0
    stopword_list = loadStopwords()
    numberOfWords = len(tweet.words)
    for i in range(numberOfWords):
        
        if tweet.words[i].text in stopword_list:
            stopword_count += 1
    
    value = stopword_count/float(numberOfWords)
    if value > 0.8:
        return 1
    else:
        return 0

def interjection_feature(tweet):
    """
    number of interjections
    """
    interjection_list = loadInterjections()
     
    interjection_number = 0
     
    numberOfWords = len(tweet.words)
     
    for i in range(numberOfWords):
        if tweet.words[i].text in interjection_list:
            interjection_number += 1
    
    return interjection_number

def interjection_feature_01(tweet):
    """
    number of interjections
    """
    interjection_list = loadInterjections()
     
    interjection_number = 0
     
    numberOfWords = len(tweet.words)
     
    for i in range(numberOfWords):
        if tweet.words[i].text in interjection_list:
            interjection_number += 1
    
    return interjection_number/float(numberOfWords)

def interjection_feature_stack1(tweet):
    """
    number of interjections
    """
    interjection_list = loadInterjections()
     
    interjection_number = 0
     
    numberOfWords = len(tweet.words)
     
    for i in range(numberOfWords):
        if tweet.words[i].text in interjection_list:
            interjection_number += 1
    
    value = interjection_number/float(numberOfWords)
    if value > 0:
        return 1
    else:  
        return 0 

def interjection_feature_stack2(tweet):
    """
    number of interjections
    """
    interjection_list = loadInterjections()
     
    interjection_number = 0
     
    numberOfWords = len(tweet.words)
     
    for i in range(numberOfWords):
        if tweet.words[i].text in interjection_list:
            interjection_number += 1
    
    value = interjection_number/float(numberOfWords)
    if value > 0.2:
        return 1
    else:  
        return 0

def interjection_feature_stack3(tweet):
    """
    number of interjections
    """
    interjection_list = loadInterjections()
     
    interjection_number = 0
     
    numberOfWords = len(tweet.words)
     
    for i in range(numberOfWords):
        if tweet.words[i].text in interjection_list:
            interjection_number += 1
    
    value = interjection_number/float(numberOfWords)
    if value > 0.4:
        return 1
    else:  
        return 0

def interjection_feature_stack4(tweet):
    """
    number of interjections
    """
    interjection_list = loadInterjections()
     
    interjection_number = 0
     
    numberOfWords = len(tweet.words)
     
    for i in range(numberOfWords):
        if tweet.words[i].text in interjection_list:
            interjection_number += 1
    
    value = interjection_number/float(numberOfWords)
    if value > 0.6:
        return 1
    else:  
        return 0
    
def interjection_feature_stack5(tweet):
    """
    number of interjections
    """
    interjection_list = loadInterjections()
     
    interjection_number = 0
     
    numberOfWords = len(tweet.words)
     
    for i in range(numberOfWords):
        if tweet.words[i].text in interjection_list:
            interjection_number += 1
    
    value = interjection_number/float(numberOfWords)
    if value > 0.8:
        return 1
    else:  
        return 0

def capitalized_words(tweet):
     
    numberOfWords = len(tweet.words)
     
    capitalized = 0
     
    for i in range(numberOfWords):
        if tweet.words[i].text.isupper():
            capitalized += 1

    return capitalized

def capitalized_words_01(tweet):
     
    numberOfWords = len(tweet.words)
     
    capitalized = 0
     
    for i in range(numberOfWords):
        if tweet.words[i].text.isupper():
            capitalized += 1

    return capitalized/float(numberOfWords)


def capitalized_words_stack1(tweet):
    
    numberOfWords = len(tweet.words)
     
    capitalized = 0
     
    for i in range(numberOfWords):
        if tweet.words[i].text.isupper():
            capitalized += 1
    
    value = capitalized/float(numberOfWords)
    if value > 0:
        return 1
    else:
        return 0

def capitalized_words_stack2(tweet):
    
    numberOfWords = len(tweet.words)
     
    capitalized = 0
     
    for i in range(numberOfWords):
        if tweet.words[i].text.isupper():
            capitalized += 1
    
    value = capitalized/float(numberOfWords)
    if value > 0.2:
        return 1
    else:
        return 0 

def capitalized_words_stack3(tweet):
    
    numberOfWords = len(tweet.words)
     
    capitalized = 0
     
    for i in range(numberOfWords):
        if tweet.words[i].text.isupper():
            capitalized += 1
    
    value = capitalized/float(numberOfWords)
    if value > 0.4:
        return 1
    else:
        return 0

def capitalized_words_stack4(tweet):
    
    numberOfWords = len(tweet.words)
     
    capitalized = 0
     
    for i in range(numberOfWords):
        if tweet.words[i].text.isupper():
            capitalized += 1
    
    value = capitalized/float(numberOfWords)
    if value > 0.6:
        return 1
    else:
        return 0

def capitalized_words_stack5(tweet):
    
    numberOfWords = len(tweet.words)
     
    capitalized = 0
     
    for i in range(numberOfWords):
        if tweet.words[i].text.isupper():
            capitalized += 1
    
    value = capitalized/float(numberOfWords)
    if value > 0.8:
        return 1
    else:
        return 0

def tweet_length(tweet):
    length= tweet.numberOfWords()
    
    return length

def tweet_length_01(tweet):
    length= tweet.numberOfWords()
    
    return length/float(max_tweet_length)

def tweet_length_stack1(tweet):
    length= tweet.numberOfWords()
    
    value = length/float(max_tweet_length)
    if value > 0:
        return 1
    else:
        return 0

def tweet_length_stack2(tweet):
    length= tweet.numberOfWords()
    
    value = length/float(max_tweet_length)
    if value > 0.2:
        return 1
    else:
        return 0

def tweet_length_stack3(tweet):
    length= tweet.numberOfWords()
    
    value = length/float(max_tweet_length)
    if value > 0.4:
        return 1
    else:
        return 0

def tweet_length_stack4(tweet):
    length= tweet.numberOfWords()
    
    value = length/float(max_tweet_length)
    if value > 0.6:
        return 1
    else:
        return 0
    
def tweet_length_stack5(tweet):
    length= tweet.numberOfWords()
    
    value = length/float(max_tweet_length)
    if value > 0.8:
        return 1
    else:
        return 0

def symbols(tweet):
    _, smileyDict, _ = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    nr = 0
    
    symbol = set()
    symbol.add("S")
    symbol.add("L")
    symbol.add("F")
         
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label in symbol:
                nr += 1
    
    return nr

def symbols_01(tweet):
    _, smileyDict, _ = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    nr = 0
    
    symbol = set()
    symbol.add("S")
    symbol.add("L")
    symbol.add("F")
         
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label in symbol:
                nr += 1
    
    return nr/float(numberOfWords)

def symbols_stack1(tweet):
    _, smileyDict, _ = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    nr = 0
    
    symbol = set()
    symbol.add("S")
    symbol.add("L")
    symbol.add("F")
         
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label in symbol:
                nr += 1
    
    value = nr/float(numberOfWords)            
    if value > 0:
        return 1
    else:
        return 0

def symbols_stack2(tweet):
    _, smileyDict, _ = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    nr = 0
    
    symbol = set()
    symbol.add("S")
    symbol.add("L")
    symbol.add("F")
         
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label in symbol:
                nr += 1
                
    value = nr/float(numberOfWords)            
    if value > 0.2:
        return 1
    else:
        return 0

def symbols_stack3(tweet):
    _, smileyDict, _ = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    nr = 0
    
    symbol = set()
    symbol.add("S")
    symbol.add("L")
    symbol.add("F")
         
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label in symbol:
                nr += 1
                
    value = nr/float(numberOfWords)            
    if value > 0.4:
        return 1
    else:
        return 0

def symbols_stack4(tweet):
    _, smileyDict, _ = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    nr = 0
    
    symbol = set()
    symbol.add("S")
    symbol.add("L")
    symbol.add("F")
         
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label in symbol:
                nr += 1
                
    value = nr/float(numberOfWords)            
    if value > 0.6:
        return 1
    else:
        return 0

def symbols_stack5(tweet):
    _, smileyDict, _ = loadSmileys()
        
    numberOfWords = len(tweet.words)
    
    nr = 0
    
    symbol = set()
    symbol.add("S")
    symbol.add("L")
    symbol.add("F")
         
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label in symbol:
                nr += 1
                
    value = nr/float(numberOfWords)            
    if value > 0.8:
        return 1
    else:
        return 0


def symbols_binary(tweet):
    _, smileyDict, _ = loadSmileys()
        
    numberOfWords = len(tweet.words)
        
    symbol = set()
    symbol.add("S")
    symbol.add("L")
    symbol.add("F")
         
    for i in range(numberOfWords):
        if tweet.words[i].text in smileyDict.keys():
            label = smileyDict[tweet.words[i].text]
            if label in symbol:
                return 1
    
    return 0

def sentiment_score(tweet):
     
    numberOfWords = len(tweet.words)
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
            
    
    return score

def sentiment_score_pos_01(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_pos = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    if score > 0:   
        return score/float(max_sentiment_score_pos)
    else:
        return 0
    
def sentiment_score_neg_01(tweet):
        
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_neg = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    if score < 0:   
        return abs(score)/float(max_sentiment_score_neg)
    else:
        return 0

def sentiment_score_neut_01(tweet):
     
    numberOfWords = len(tweet.words)
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    if score == 0:   
        return 1
    else:
        return 0
    
def sentiment_score_binary(tweet):
     
    numberOfWords = len(tweet.words)
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]

    if score > 0:
        return 1
    else:
        return 0

def sentiment_score_pos_stack1(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_pos = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = score/float(max_sentiment_score_pos)
    if value > 0:
        return 1
    else:
        return 0

def sentiment_score_pos_stack2(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_pos = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = score/float(max_sentiment_score_pos)
    if value > 0.2:
        return 1
    else:
        return 0

def sentiment_score_pos_stack3(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_pos = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = score/float(max_sentiment_score_pos)
    if value > 0.4:
        return 1
    else:
        return 0

def sentiment_score_pos_stack4(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_pos = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = score/float(max_sentiment_score_pos)
    if value > 0.6:
        return 1
    else:
        return 0

def sentiment_score_pos_stack5(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_pos = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = score/float(max_sentiment_score_pos)
    if value > 0.8:
        return 1
    else:
        return 0

def sentiment_score_neut_stack(tweet):
     
    numberOfWords = len(tweet.words)
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    if score == 0:
        return 1
    else:
        return 0

def sentiment_score_neg_stack1(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_neg = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = abs(score)/float(max_sentiment_score_neg)
    if value > 0:
        return 1
    else:
        return 0

def sentiment_score_neg_stack2(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_neg = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = abs(score)/float(max_sentiment_score_neg)
    if value > 0.2:
        return 1
    else:
        return 0

def sentiment_score_neg_stack3(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_neg = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = abs(score)/float(max_sentiment_score_neg)
    if value > 0.4:
        return 1
    else:
        return 0

def sentiment_score_neg_stack4(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_neg = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = abs(score)/float(max_sentiment_score_neg)
    if value > 0.6:
        return 1
    else:
        return 0
    
def sentiment_score_neg_stack5(tweet):
     
    numberOfWords = len(tweet.words)
    
    max_sentiment_score_neg = numberOfWords * 5
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
    
    value = abs(score)/float(max_sentiment_score_neg)
    if value > 0.8:
        return 1
    else:
        return 0

# ----- binaere Features --------
def negation_binary(tweet):
    """ binary feature: if tweet contains negation """
    numberOfWords = len(tweet.words)
    shift = 0
    for i in range(numberOfWords-1):
        if tweet.words[i].text == "not" or tweet.words[i].text == "n't" or tweet.words[i].text.endswith("n't"):
            shift = 1

    return shift

def repeating_feature(tweet):
    """ It's so perfect, just perfect --> repetition within n words? """
    numberOfWords = len(tweet.words)
    n = 4
    for i in range(numberOfWords):
        previousWords = tweet.words[(i-n if i > n-1 else 0):i]
        nextWords = tweet.words[i+1:(i+n+1 if i < (numberOfWords-n) else numberOfWords)]
        if tweet.words[i] in previousWords or tweet.words[i] in nextWords:
            return 1
          
    return 0

def more_positive_scores(tweet):
    """ tweet contains more words with positive than with negative score """
    numberOfWords = len(tweet.words)
     
    score = 0
     
    score_dict = loadSentimentScore()
    
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
     
    if score > 0:
        return 1
     
    return 0
     
def more_negative_scores(tweet):
    """ tweet contains more words with negative than with positive score """
    numberOfWords = len(tweet.words)
      
    score = 0
      
    score_dict = loadSentimentScore()
     
    for i in range(numberOfWords):
        word = tweet.words[i].text.encode("utf-8")
        if word in score_dict.keys():
            score += score_dict[word]
      
    if score < 0:
        return 1
      
    return 0

def to_user(tweet):
    """ tweet contains user reference """
    numberOfWords = len(tweet.words)
        
    for i in range(numberOfWords):
        if tweet.words[i].text == "@" or tweet.words[i].text.startswith("@"):
            return 1
     
    return 0

def url(tweet):
    """ tweet contains url """
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet.text)
    if len(urls) > 0:
        return 1
     
    return 0
 
def other_hashtags_binary(tweet):
    """ tweet contains additional hashtags (apart from class label) """ 
    hashtag = re.findall('\#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet.text)
        
    if len(hashtag) > 0:
        return 1
 
    return 0

def other_hashtags(tweet):
    """ tweet contains additional hashtags (apart from class label) """ 
    hashtag = re.findall('\#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet.text)
    
    if len(hashtag) > 0:
        return len(hashtag)
 
    return 0

def other_hashtags_01(tweet):
    """normalized version: tweet contains additional hashtags (apart from class label) """
    numberOfWords = len(tweet.words)
    
    hashtag = re.findall('\#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet.text)
        
    if len(hashtag) > 0:
        return len(hashtag)/float(numberOfWords)
 
    return 0

def other_hashtags_stack1(tweet):
    """ tweet contains additional hashtags (apart from class label) """
    numberOfWords = len(tweet.words)
    
    hashtag = re.findall('\#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet.text)
        
    if len(hashtag) > 0:
        value = len(hashtag)/float(numberOfWords)
        if value > 0:
            return 1
 
    return 0

def other_hashtags_stack2(tweet):
    """ tweet contains additional hashtags (apart from class label) """
    numberOfWords = len(tweet.words)
    
    hashtag = re.findall('\#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet.text)
        
    if len(hashtag) > 0:
        value = len(hashtag)/float(numberOfWords)
        if value > 0.2:
            return 1
 
    return 0

def other_hashtags_stack3(tweet):
    """ tweet contains additional hashtags (apart from class label) """
    numberOfWords = len(tweet.words)
    
    hashtag = re.findall('\#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet.text)
        
    if len(hashtag) > 0:
        value = len(hashtag)/float(numberOfWords)
        if value > 0.4:
            return 1
 
    return 0

def other_hashtags_stack4(tweet):
    """ tweet contains additional hashtags (apart from class label) """
    numberOfWords = len(tweet.words)
    
    hashtag = re.findall('\#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet.text)
        
    if len(hashtag) > 0:
        value = len(hashtag)/float(numberOfWords)
        if value > 0.6:
            return 1
 
    return 0

def other_hashtags_stack5(tweet):
    """ tweet contains additional hashtags (apart from class label) """
    numberOfWords = len(tweet.words)
    
    hashtag = re.findall('\#(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet.text)
        
    if len(hashtag) > 0:
        value = len(hashtag)/float(numberOfWords)
        if value > 0.8:
            return 1
 
    return 0

def interjectionPlusPunctuation(tweet, pattern=r"(!!|!\?|\?!|\?|\?\?)"):
    """Searches for an interjection followed by the given pattern."""
    exclamation = re.compile(pattern, flags=re.UNICODE|re.VERBOSE)
    interjections = loadInterjections()
    numberOfWords = len(tweet.words)
 
    for i in range(numberOfWords-1):
        if exclamation.findall(tweet.words[i].text + tweet.words[i+1].text):
            
            # Get the 2 previous words
            previousWords = tweet.words[(i-2 if i > 1 else 0):i] 
            for w in previousWords:
                if w.text in interjections:
                    return 1
    return 0

def valence_shift(tweet, n=4):
    """ tweet contains shift to opposite sentiment within a 4 word window. """
    numberOfWords = len(tweet.words)
    opposite = {"positive": "negative", "negative": "positive"}
    for i in range(numberOfWords-1):
        pol = tweet.words[i].polarity
        # Get (maximal) the 4 previous words
        if pol == "neutral":
            continue
        else:
            previousWords = tweet.words[(i-n if i > (n-1) else 0):i]
            previousPolarity = [w.polarity for w in previousWords ]
            if opposite[pol] in previousPolarity:
                return 1
    return 0
                
# ---- Feature templates from Buschmeier Code ----
def scareQuotes(tweet):
    """
    Searches for scare quotes surrounding one or two positive noun, 
    adjective or adverb.
    """
    nounCategories = ["NN", "NNS", "NNP", "NNPS"]
    adjCategories = ["JJ", "JJR", "JJS"]
    advCategories = ["RB", "RBR", "RBS"]

    polarityCategories = nounCategories + adjCategories + advCategories
    openingMarks = [u"\"", u"`", u"``", u"*"]
    closingMarks = [u"\"", u"'", u"''", u"*"]

    numberOfWords = len(tweet.taggedTweet)
    for i in range(numberOfWords):
        # Opening quotes
        if tweet.taggedTweet[i].text in openingMarks:
            # One on the two words in quotes is a positive adjective of noun
            if (i + 3 < numberOfWords-1 and 
                tweet.taggedTweet[i+3].text in closingMarks):
                if (tweet.taggedTweet[i+1].polarity == "positive" and 
                tweet.taggedTweet[i+1].pos in polarityCategories or
                tweet.taggedTweet[i+2].polarity == "positive" and
                tweet.taggedTweet[i+2].pos in polarityCategories):
                    return 1

            # The word in quotes is a positive adjective of noun
            if (i + 2 < numberOfWords-1 and 
                tweet.taggedTweet[i+2].text in closingMarks):
                if (tweet.taggedTweet[i+1].polarity == "positive" and 
                tweet.taggedTweet[i+1].pos in polarityCategories):
                    return 1
    return 0

def scareQuotesNegative(tweet):
    """
    Searches for scare quotes surrounding one or two negative noun, 
    adjective or adverb.
    """
    nounCategories = ["NN", "NNS", "NNP", "NNPS"]
    adjCategories = ["JJ", "JJR", "JJS"]
    advCategories = ["RB", "RBR", "RBS"]

    polarityCategories = nounCategories + adjCategories + advCategories
    openingMarks = [u"\"", u"`", u"``", u"*"]
    closingMarks = [u"\"", u"'", u"''", u"*"]

    numberOfWords = len(tweet.taggedTweet)
    for i in range(numberOfWords):
        # Opening quotes
        if tweet.taggedTweet[i].text in openingMarks:
            # One of the two words in quotes is a negative adjective of noun
            if (i + 3 < numberOfWords-1 and 
                tweet.taggedTweet[i+3].text in closingMarks):
                if (tweet.taggedTweet[i+1].polarity == "negative" and 
                tweet.taggedTweet[i+1].pos in polarityCategories or
                tweet.taggedTweet[i+2].polarity == "negative" and
                tweet.taggedTweet[i+2].pos in polarityCategories):
                    return 1

            # The word in quotes is a negative adjective of noun
            if (i + 2 < numberOfWords-1 and 
                tweet.taggedTweet[i+2].text in closingMarks):
                if (tweet.taggedTweet[i+1].polarity == "negative" and 
                tweet.taggedTweet[i+1].pos in polarityCategories):
                    return 1
    return 0


def positiveNGramPlusPunctuation(tweet, n=4, pattern=r"(!!|!\?|\?!)"):
    """
    Searches (4-Gram+) followed by the given pattern.
    """
    exclamation = re.compile(pattern, flags=re.UNICODE|re.VERBOSE)
    numberOfWords = len(tweet.words)

    for i in range(numberOfWords-1):
        if exclamation.findall(tweet.words[i].text + tweet.words[i+1].text):
            # Get (maximal) the 4 previous words
            previousWords = tweet.words[(i-n if i > (n-1) else 0):i] 
            positiveWords = [w for w in previousWords 
                            if w.polarity == "positive"]
            negativeWords = [w for w in previousWords 
                            if w.polarity == "negative"]
            if positiveWords and not negativeWords:
                return 1
    return 0

def negativeNGramPlusPunctuation(tweet, n=4, pattern=r"(!!|!\?|\?!)"):
    """Searches a (4-Gram-) followed by the given pattern."""
    exclamation = re.compile(pattern, flags=re.UNICODE|re.VERBOSE)
    numberOfWords = len(tweet.words)

    for i in range(numberOfWords-1): 
        if exclamation.findall(tweet.words[i].text + tweet.words[i+1].text):
            # Get (maximal) the 4 previous words
            previousWords = tweet.words[((i-n) if i > (n-1) else 0):i] 
            positiveWords = [w for w in previousWords 
                            if w.polarity == "positive"]
            negativeWords = [w for w in previousWords 
                            if w.polarity == "negative"]
            if not positiveWords and negativeWords:
                return 1
    return 0

def ellipsisPlusPunctuation(tweet, pattern=r"(!!|!\?|\?!|\?)"):
    """
    Searches for an ellipsis followed by the given pattern.
    """
    exclamation = re.compile(pattern, flags=re.UNICODE|re.VERBOSE)
    ellipsis = re.compile(r"(\.\.|\. \. \.)$")
    numberOfWords = len(tweet.words)

    for i in range(numberOfWords-1):
        if exclamation.findall(tweet.words[i].text + tweet.words[i+1].text):
            # Get the 2 previous words
            previousWords = tweet.words[(i-2 if i > 1 else 0):i] 

            if ellipsis.findall("".join([w.text for w in previousWords])):
                return 1
    return 0


def positiveStreak(tweet, length=3):
    """Searches for streaks of positive words."""
    numberOfWords = len(tweet.words)

    for i in range(numberOfWords-length):
        if tweet.words[i].polarity == "positive":
            if all([True if w.polarity == "positive" else False 
                        for w in tweet.words[i+1:i+length]]):
                return 1
    return 0

def negativeStreak(tweet, length=3):
    """Searches for streaks of negative words."""
    numberOfWords = len(tweet.words)
    
    for i in range(numberOfWords-length):
        
        if tweet.words[i].polarity == "negative":
            if all([True if w.polarity == "negative" else False 
                        for w in tweet.words[i+1:i+length]]):
                return 1
    return 0


#--------------------- BAG OF WORDS -----------------------------
def loadSmileysDict(tweets):
    s = set()
    for tweet in tweets.values():
        
        for each in tweet.words:
            if not re.findall('[{}\(\)=+#@><~*a-zA-z!?:.,-;&\"\'$|]+|[0-9]+', each.text):
                    if each.text not in s:
                        s.add(each.text)
    return s

def createTopBow(tweets):
    """ create frequency ranking for bag of words """
    dictionary = {}
    index = 0
    smileys, _, _ = loadSmileys()
    stopword_list = loadStopwords()
    
    for tweet in tweets.values():
        
        for each in tweet.words:
            # do some corrections for avoiding errors.
            if each.text == '"':
                word = "'"
            elif each.text == "\\":
                word = "\\\\"
            elif each.text == ":\\":
                word = ":\\\\" 
            elif not each.text in smileys:
                word = each.text.lower()
            else:
                word = each.text
                    
            try:
                dictionary[word] += 1
            except:
                dictionary[word] = 1
    
    dict_list = sorted(dictionary.iteritems(), key=lambda x: x[1], reverse=True)

    return [pair[0] for pair in dict_list]

def createBagOfWordsDictionary(tweets):
    """Create the dictionary of all words."""
    dictionary = {}
    index = 0
    smileys, _, _ = loadSmileys()
    
    topBoW = createTopBow(tweets)
    #print len(topBoW)    
    for tweet in tweets.values():
        
        for each in tweet.words:
            if each.text == '"':
                word = "'"
            elif each.text == "\\":
                word = "\\\\"
            elif each.text == ":\\":
                word = ":\\\\" 
            elif not each.text in smileys:
                word = each.text.lower()
            else:
                word = each.text
                
            if not word in dictionary and word in topBoW:
                
                dictionary[word] = index
                index += 1

    # print("Unigram index:", index)
    return dictionary

def fillBagOfWords(bowDictionary, tweet):
    """Fill a bag with words of one tweet."""
    # initialise with zeros
    bag = [0] * len(bowDictionary)
    smileys, _, _ = loadSmileys()
    
    for word in tweet.words:
        try:
            if word.text == '"':
                bag[bowDictionary["'"]] = 1
            elif word.text == "\\":
                bag[bowDictionary["\\\\"]] = 1
            elif word.text == ":\\":
                bag[bowDictionary[":\\\\"]] = 1    
            elif not word.text in smileys:
                bag[bowDictionary[word.text.lower()]] = 1
            else:
                bag[bowDictionary[word.text]] = 1
        except:
            continue       
    
    return bag

def verifyBagOfWords(tweet, bowDictionary ):
    initialWords = set(tweet.words) 
    bag = fillBagOfWords(bowDictionary, tweet)
    indices = [i for i, v in enumerate(bag) if v == 1]
    unpackedWords = {word for i, v in enumerate(bag) if v == 1 
                    for word, index in bowDictionary .items() if index == i}
    print("Initial words:", len(initialWords), "Bag words:", len(unpackedWords))
    return not initialWords - unpackedWords


def createTopBigrams(tweets):
    """ create frequency ranking of bigrams """
    dictionary = {}
    
    smileys, _, _ = loadSmileys()
            
    for tweet in tweets.values():
        # Transform bigrams of tokens to strings.
        
        for word1, word2 in tweet.bigrams:
            if word1.text == '"':
                first = "'"
            elif word1.text == "\\":
                first = "\\\\"
            elif word1.text == ":\\":
                first = ":\\\\" 
            elif not word1.text in smileys:
                first = word1.text.lower()
            else:
                first = word1.text
            
            if word2.text == '"':
                second = "'"
            elif word2.text == "\\":
                second = "\\\\"
            elif word2.text == ":\\":
                second = ":\\\\" 
            elif not word2.text in smileys:
                second = word2.text.lower()
            else:
                second = word2.text
             
            bigram = (first, second)
        
            try:
                dictionary[bigram] += 1
            except:
                dictionary[bigram] = 1
    
    dict_list = sorted(dictionary.iteritems(), key=lambda x: x[1], reverse=True)  
    
    # use of 50 000 most frequent bigrams in corpus.
    return [pair[0] for pair in dict_list[:50000]]


def createBagOfBigramsDictionary (tweets):
    """Create a dictionary of bigrams."""
    dictionary = {}
    index = 0
    smileys, _, _ = loadSmileys()
    topBigrams = createTopBigrams(tweets)
            
    for tweet in tweets.values():
        # Transform bigrams of tokens to strings.
        for word1, word2 in tweet.bigrams:
            if word1.text == '"':
                first = "'"
            elif word1.text == "\\":
                first = "\\\\"
            elif word1.text == ":\\":
                first = ":\\\\" 
            elif not word1.text in smileys:
                first = word1.text.lower()
            else:
                first = word1.text
            
            if word2.text == '"':
                second = "'"
            elif word2.text == "\\":
                second = "\\\\"
            elif word2.text == ":\\":
                second = ":\\\\" 
            elif not word2.text in smileys:
                second = word2.text.lower()
            else:
                second = word2.text
                
            bigram = (first, second)
                    
            if not bigram in dictionary and bigram in topBigrams:
                
                dictionary[bigram] = index
                index += 1
                

    # print("Bigram index:", index)
    return dictionary

def fillBagOfBigrams(bigramDictionary, tweet):
    """Fill a bag with bigrams of one tweet."""
    # initialise with zeros
    bag = [0] * len(bigramDictionary)
    smileys, _, _ = loadSmileys()
            
    for word1, word2 in tweet.bigrams:
        if word1.text == '"':
            first = "'"
        elif word1.text == "\\":
            first = "\\\\"
        elif word1.text == ":\\":
            first = ":\\\\" 
        elif not word1.text in smileys:
            first = word1.text.lower()
        else:
            first = word1.text
        
        if word2.text == '"':
            second = "'"
        elif word2.text == "\\":
            second = "\\\\"
        elif word2.text == ":\\":
            second = ":\\\\" 
        elif not word2.text in smileys:
            second = word2.text.lower()
        else:
            second = word2.text
            
        bigram = (first, second)
        
        try:
            bag[bigramDictionary[bigram]] = 1
        except:
            continue
       
    return bag

#############################################################################################

###################### SENTIMENT FEATURE ####################################################
def sentiment(tweet):
        """Returns a vector for the sentiment of a given tweet."""

        result = [0] * 1
        polarity = len(tweet.positiveWords) - len(tweet.negativeWords)
        if polarity > 0:
            result[0] = 1

        return result

############################################################################################

#----------------------------FEATURE GENERATION --------------------------------------------

def createFeatures(featureConfig=None, feat = True, regExp = True, stack_binning=True, binary_combination = False, sentiment = True, bigram = True):
    """Returns a list of features created from configurations.""" 
             
    allFeatureConfig = {u"Positive Quotes B": (u"\"..\"", scareQuotes),
                    u"Negative Quotes B": (u"\"--\"", scareQuotesNegative),
                    u"Pos&Punctuation B": (u"w+!?", positiveNGramPlusPunctuation),
                    u"Neg&Punctuation B": (u"w-!?", negativeNGramPlusPunctuation),
                    u"Positive Hyperbole B": (u"3w+", positiveStreak),
                    u"Negative Hyperbole B": (u"3w-", negativeStreak),
                    u"Ellipsis and Punctuation B": (u"..?!", ellipsisPlusPunctuation),
                    u"Pos&Ellipsis B": (u"w+..", lambda x: positiveNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
                    u"Neg&Ellipsis B": (u"w-..", lambda x: negativeNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
                    # ----- my features ---------------------------------------------
                    u"Smileys": (u":)", numberOfSmileys),
                    u"Following Smileys": (u":):)", numberOfFollowingSmileys),
                    u"Sentiment Smileys Pos": (u":)+", sentiment_smileys_pos),
                    u"Sentiment Smileys Neg": (u":(-", sentiment_smileys_neg),
                    u"Length of sentences": (u"sent", sentence_length_gap),
                    u"Subjective Pronomina": (u"pron", subjective_pronomina),
                    u"Number of Nouns": (u"NN", noun_pos_tag),
                    u"Number of Verbs": (u"V", verb_pos_tag),
                    u"Number of Adj": (u"JJ", adj_pos_tag),
                    u"Number of Adv": (u"RB", adv_pos_tag),
                    u"Sentiment Score Gap": (u"gap", sentiment_score_gap),
                    u"Number of Stopwords": (u"stop", stopwords),
                    u"Number of Interjections": (u"ITJ_num", interjection_feature),
                    u"Capitalized Words": (u"WW", capitalized_words),
                    u"Length of Tweet": (u"len", tweet_length),
                    u"Symbols": (u"symbol", symbols),
                    u"Sentiment Score": (u"+/-", sentiment_score),
                    u"Other Hashtags": (u"#", other_hashtags),
                    # ---------------- binary --------------------------------------                 
                    u"Sentiment Smileys Pos Binary": (u":)b+", sentiment_smileys_pos_binary),
                    u"Sentiment Smileys Neg Binary": (u":)b-", sentiment_smileys_neg_binary),                        
                    u"Symbols Binary": (u"symbol", symbols_binary),
                    u"Sentiment Score Binary": (u"+/-", sentiment_score_binary),
                    u"Repeated Words Binary": (u"rep", repeating_feature),
                    u"More positive Sentiment Scores Binary": (u"+", more_positive_scores),
                    u"More negative Sentiment Scores Binary": (u"-", more_negative_scores),
                    u"@User Binary": (u"@user", to_user),
                    u"URL Binary": (u"url", url),
                    u"Other Hashtags Binary": (u"#", other_hashtags_binary),
                    u"Negation Binary": (u"shift", negation_binary),
                    u"Interjektion?! Binary": (u"int?!", interjectionPlusPunctuation),
                    u"Valence Shift Binary": (u"+<>-", valence_shift),
                    # ---------------- normalized ---------------------------
                    u"Smileys normalized": (u":)n", numberOfSmileys_01),
                    u"Following Smileys normalized": (u":):)", numberOfFollowingSmileys_01),                        
                    u"Length of sentences normalized": (u"sent-n", sentence_length_gap_01),                        
                    u"Sentiment Smileys normalized": (u":)n", sentiment_smileys_01),                        
                    u"Subjective Pronomina normalized": (u"pron-n", subjective_pronomina_01),                        
                    u"Number of Nouns normalized": (u"NN-n", noun_pos_tag_01),
                    u"Number of Verbs normalized": (u"V-n", verb_pos_tag_01),
                    u"Number of Adj normalized": (u"JJ-n", adj_pos_tag_01),
                    u"Number of Adv normalized": (u"RB-n", adv_pos_tag_01),
                    u"Symbols normalized": (u"symbol-n", symbols_01),
                    u"Sentiment Score Gap normalized": (u"gap-n", sentiment_score_gap_01),
                    u"Number of Stopwords normalized": (u"stop-n", stopwords_01),
                    u"Number of Interjections normalized": (u"ITJ_num-n", interjection_feature_01),
                    u"Capitalized Words normalized": (u"WW-n", capitalized_words_01),
                    u"Tweet Length normalized": (u"len", tweet_length_01),
                    u"Other Hashtags normalized": (u"#-n", other_hashtags_01),
                    u"Sentiment Score Pos normalized": (u"+n", sentiment_score_pos_01),
                    u"Sentiment Score Neg normalized": (u"-n", sentiment_score_neg_01),
                    u"Sentiment Score Neut normalized": (u"0n", sentiment_score_neut_01),
                    u"Sentiment Smiley Pos normalized": (u":)+n", sentiment_smileys_pos_01),
                    u"Sentiment Smiley Neg normalized": (u":(-n", sentiment_smileys_neg_01),
                    # ----------------- Stacks -----------------------------------------
                    u"Smileys Stack 1": (u":)1", numberOfSmileys_stack1),
                    u"Following Smileys Stack 1": (u":):)1", numberOfFollowingSmileys_stack1),
                    u"Sentiment Smileys Stack 1": (u":)(1", sentiment_smileys_stack1),
                    u"Sentiment Smileys Pos Stack 1": (u":)+1", sentiment_smileys_pos_stack1),
                    u"Sentiment Smileys Neg Stack 1": (u":)-1", sentiment_smileys_neg_stack1),
                    u"Length of sentences Stack 1": (u"sent1", sentence_length_gap_stack1),                       
                    u"Subjective Pronomina Stack 1": (u"pron1", subjective_pronomina_stack1),
                    u"Number of Nouns Stack 1": (u"NN", noun_pos_tag_stack1),
                    u"Number of Verbs Stack 1": (u"V", verb_pos_tag_stack1),
                    u"Number of Adj Stack 1": (u"JJ", adj_pos_tag_stack1),
                    u"Number of Adv Stack 1": (u"RB", adv_pos_tag_stack1),
                    u"Sentiment Score Gap Stack 1": (u"gap", sentiment_score_gap_stack1),
                    u"Number of Stopwords Stack 1": (u"stop", stopwords_stack1),
                    u"Number of Interjections Stack 1": (u"ITJ_num", interjection_feature_stack1),
                    u"Capitalized Words Stack 1": (u"WW", capitalized_words_stack1),
                    u"Length of Tweet Stack 1": (u"len", tweet_length_stack1),
                    u"Symbols Stack 1": (u"symbol", symbols_stack1),
                    u"Sentiment Score Pos Stack 1": (u"+", sentiment_score_pos_stack1),
                    u"Sentiment Score Neg Stack 1": (u"-", sentiment_score_neg_stack1),
                    u"Other Hashtags Stack 1": (u"#", other_hashtags_stack1),
                    u"Smileys Stack 2": (u":)2", numberOfSmileys_stack2),
                    u"Following Smileys Stack 2": (u":):)2", numberOfFollowingSmileys_stack2),
                    u"Sentiment Smileys Stack 2": (u":)(1", sentiment_smileys_stack2),                      
                    u"Sentiment Smileys Pos Stack 2": (u":)+2", sentiment_smileys_pos_stack2),
                    u"Sentiment Smileys Neg Stack 2": (u":)-2", sentiment_smileys_neg_stack2),
                    u"Length of sentences Stack 2": (u"sent2", sentence_length_gap_stack2),                       
                    u"Subjective Pronomina Stack 2": (u"pron2", subjective_pronomina_stack2),
                    u"Number of Nouns Stack 2": (u"NN", noun_pos_tag_stack2),
                    u"Number of Verbs Stack 2": (u"V", verb_pos_tag_stack2),
                    u"Number of Adj Stack 2": (u"JJ", adj_pos_tag_stack2),
                    u"Number of Adv Stack 2": (u"RB", adv_pos_tag_stack2),
                    u"Sentiment Score Gap Stack 2": (u"gap", sentiment_score_gap_stack2),
                    u"Number of Stopwords Stack 2": (u"stop", stopwords_stack2),
                    u"Number of Interjections Stack 2": (u"ITJ_num", interjection_feature_stack2),
                    u"Capitalized Words Stack 2": (u"WW", capitalized_words_stack2),
                    u"Length of Tweet Stack 2": (u"len", tweet_length_stack2),
                    u"Symbols Stack 2": (u"symbol", symbols_stack2),
                    u"Sentiment Score Pos Stack 2": (u"+", sentiment_score_pos_stack2),
                    u"Sentiment Score Neg Stack 2": (u"-", sentiment_score_neg_stack2),
                    u"Other Hashtags Stack 2": (u"#", other_hashtags_stack2),
                    u"Smileys Stack 3": (u":)3", numberOfSmileys_stack3),
                    u"Following Smileys Stack 3": (u":):)3", numberOfFollowingSmileys_stack3),
                    u"Sentiment Smileys Stack 3": (u":)(1", sentiment_smileys_stack3),
                    u"Sentiment Smileys Pos Stack 3": (u":)+3", sentiment_smileys_pos_stack3),
                    u"Sentiment Smileys Neg Stack 3": (u":)-3", sentiment_smileys_neg_stack3),               
                    u"Length of sentences Stack 3": (u"sent3", sentence_length_gap_stack3),                       
                    u"Subjective Pronomina Stack 3": (u"pron3", subjective_pronomina_stack3),
                    u"Number of Nouns Stack 3": (u"NN", noun_pos_tag_stack3),
                    u"Number of Verbs Stack 3": (u"V", verb_pos_tag_stack3),
                    u"Number of Adj Stack 3": (u"JJ", adj_pos_tag_stack3),
                    u"Number of Adv Stack 3": (u"RB", adv_pos_tag_stack3),
                    u"Sentiment Score Gap Stack 3": (u"gap", sentiment_score_gap_stack3),
                    u"Number of Stopwords Stack 3": (u"stop", stopwords_stack3),
                    u"Number of Interjections Stack 3": (u"ITJ_num", interjection_feature_stack3),
                    u"Capitalized Words Stack 3": (u"WW", capitalized_words_stack3),
                    u"Length of Tweet Stack 3": (u"len", tweet_length_stack3),
                    u"Symbols Stack 3": (u"symbol", symbols_stack3),
                    u"Sentiment Score Pos Stack 3": (u"+", sentiment_score_pos_stack3),
                    u"Sentiment Score Neg Stack 3": (u"-", sentiment_score_neg_stack3),
                    u"Other Hashtags Stack 3": (u"#", other_hashtags_stack3),
                    u"Smileys Stack 4": (u":)4", numberOfSmileys_stack4),
                    u"Following Smileys Stack 4": (u":):)4", numberOfFollowingSmileys_stack4),
                    u"Sentiment Smileys Stack 4": (u":)(1", sentiment_smileys_stack4),
                    u"Sentiment Smileys Pos Stack 4": (u":)+4", sentiment_smileys_pos_stack4),
                    u"Sentiment Smileys Neg Stack 4": (u":)-4", sentiment_smileys_neg_stack4),                       
                    u"Length of sentences Stack 4": (u"sent4", sentence_length_gap_stack4),                       
                    u"Subjective Pronomina Stack 4": (u"pron4", subjective_pronomina_stack4),
                    u"Number of Nouns Stack 4": (u"NN", noun_pos_tag_stack4),
                    u"Number of Verbs Stack 4": (u"V", verb_pos_tag_stack4),
                    u"Number of Adj Stack 4": (u"JJ", adj_pos_tag_stack4),
                    u"Number of Adv Stack 4": (u"RB", adv_pos_tag_stack4),
                    u"Sentiment Score Gap Stack 4": (u"gap", sentiment_score_gap_stack4),
                    u"Number of Stopwords Stack 4": (u"stop", stopwords_stack4),
                    u"Number of Interjections Stack 4": (u"ITJ_num", interjection_feature_stack4),
                    u"Capitalized Words Stack 4": (u"WW", capitalized_words_stack4),
                    u"Length of Tweet Stack 4": (u"len", tweet_length_stack4),
                    u"Symbols Stack 4": (u"symbol", symbols_stack4),
                    u"Sentiment Score Pos Stack 4": (u"+", sentiment_score_pos_stack4),
                    u"Sentiment Score Neg Stack 4": (u"-", sentiment_score_neg_stack4),
                    u"Other Hashtags Stack 4": (u"#", other_hashtags_stack4),
                    u"Smileys Stack 5": (u":)5", numberOfSmileys_stack5),
                    u"Following Smileys Stack 5": (u":):)5", numberOfFollowingSmileys_stack5),
                    u"Sentiment Smileys Stack 5": (u":)(1", sentiment_smileys_stack5),
                    u"Sentiment Smileys Pos Stack 5": (u":)+5", sentiment_smileys_pos_stack5),
                    u"Sentiment Smileys Neg Stack 5": (u":)-5", sentiment_smileys_neg_stack5),
                    u"Length of sentences Stack 5": (u"sent5", sentence_length_gap_stack5),                       
                    u"Subjective Pronomina Stack 5": (u"pron5", subjective_pronomina_stack5),
                    u"Number of Nouns Stack 5": (u"NN", noun_pos_tag_stack5),
                    u"Number of Verbs Stack 5": (u"V", verb_pos_tag_stack5),
                    u"Number of Adj Stack 5": (u"JJ", adj_pos_tag_stack5),
                    u"Number of Adv Stack 5": (u"RB", adv_pos_tag_stack5),
                    u"Sentiment Score Gap Stack 5": (u"gap", sentiment_score_gap_stack5),
                    u"Number of Stopwords Stack 5": (u"stop", stopwords_stack5),
                    u"Number of Interjections Stack 5": (u"ITJ_num", interjection_feature_stack5),
                    u"Capitalized Words Stack 5": (u"WW", capitalized_words_stack5),
                    u"Length of Tweet Stack 5": (u"len", tweet_length_stack5),
                    u"Symbols Stack 5": (u"symbol", symbols_stack5),
                    u"Sentiment Score Pos Stack 5": (u"+", sentiment_score_pos_stack5),
                    u"Sentiment Score Neg Stack 5": (u"-", sentiment_score_neg_stack5),
                    u"Other Hashtags Stack 5": (u"#", other_hashtags_stack5),
                    u"Sentiment Score Neut Stack": (u"0", sentiment_score_neut_stack)
                    
    }
    
    buschmeierFeatureConfig = {u"Positive Quotes B": (u"\"..\"", scareQuotes),
                    u"Negative Quotes B": (u"\"--\"", scareQuotesNegative),
                    u"Pos&Punctuation B": (u"w+!?", positiveNGramPlusPunctuation),
                    u"Neg&Punctuation B": (u"w-!?", negativeNGramPlusPunctuation),
                    u"Positive Hyperbole B": (u"3w+", positiveStreak),
                    u"Negative Hyperbole B": (u"3w-", negativeStreak),
                    u"Ellipsis and Punctuation B": (u"..?!", ellipsisPlusPunctuation),
                    u"Pos&Ellipsis B": (u"w+..", lambda x: positiveNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
                    u"Neg&Ellipsis B": (u"w-..", lambda x: negativeNGramPlusPunctuation(x, pattern=r"(\.\.|\. \. \.)$")),
    }
    
    myBinaryFeaturesConfig = {u"Sentiment Smileys Pos Binary": (u":)b+", sentiment_smileys_pos_binary),
                              u"Sentiment Smileys Neg Binary": (u":)b-", sentiment_smileys_neg_binary),                                                
                    u"Symbols Binary": (u"symbol", symbols_binary),
                    u"Sentiment Score Binary": (u"+/-", sentiment_score_binary),
                    u"Repeated Words Binary": (u"rep", repeating_feature),
                    u"More positive Sentiment Scores Binary": (u"+", more_positive_scores),
                    u"More negative Sentiment Scores Binary": (u"-", more_negative_scores),
                    u"@User Binary": (u"@user", to_user),
                    u"URL Binary": (u"url", url),
                    u"Other Hashtags Binary": (u"#", other_hashtags_binary),
                    u"Interjektion?! Binary": (u"int?!", interjectionPlusPunctuation),
                    u"Valence Shift Binary": (u"+<>-", valence_shift)}
    
    myNormalizedFeaturesConfig = {u"Smileys normalized": (u":)n", numberOfSmileys_01),
                    u"Following Smileys normalized": (u":):)", numberOfFollowingSmileys_01),                        
                    u"Length of sentences normalized": (u"sent-n", sentence_length_gap_01),                        
                    u"Sentiment Smileys normalized": (u":)n", sentiment_smileys_01),                        
                    u"Subjective Pronomina normalized": (u"pron-n", subjective_pronomina_01),                        
                    u"Number of Nouns normalized": (u"NN-n", noun_pos_tag_01),
                    u"Number of Verbs normalized": (u"V-n", verb_pos_tag_01),
                    u"Number of Adj normalized": (u"JJ-n", adj_pos_tag_01),
                    u"Number of Adv normalized": (u"RB-n", adv_pos_tag_01),
                    u"Symbols normalized": (u"symbol-n", symbols_01),
                    u"Sentiment Score Gap normalized": (u"gap-n", sentiment_score_gap_01),
                    u"Number of Stopwords normalized": (u"stop-n", stopwords_01),
                    u"Number of Interjections normalized": (u"ITJ_num-n", interjection_feature_01),
                    u"Capitalized Words normalized": (u"WW-n", capitalized_words_01),
                    u"Other Hashtags normalized": (u"#-n", other_hashtags_01),
                    u"Sentiment Score Pos normalized": (u"+n", sentiment_score_pos_01),
                    u"Sentiment Score Neg normalized": (u"-n", sentiment_score_neg_01),
                    u"Sentiment Score Neut normalized": (u"0n", sentiment_score_neut_01),
                    u"Sentiment Smiley Pos normalized": (u":)+n", sentiment_smileys_pos_01),
                    u"Sentiment Smiley Neg normalized": (u":(-n", sentiment_smileys_neg_01),
                    u"Tweet Length normalized": (u"len-n", tweet_length_01)
                    }
        
    # featureConfig can be defined with list of numbers for corresponding features.       
    if featureConfig is None:
        featureConfig = allFeatureConfig
    else:
        featureNumberList = featureConfig
        featureConfig = {}
        for number in featureNumberList:
            name = feature_names[number]  
            featureConfig.update({name: allFeatureConfig[name]})
        
    features = []
    normalized_functions = []
    stack_list = []
    cross_product = [] 
    
    configuration = featureConfig
    
    # add stack binning of all normalized features to feature set.   
    if stack_binning:
        for name, config in configuration.items():
            for i in range(4):
                
                if name.endswith("Stack " + str(i+1)):
                    stack_list.append(name)
        
        if feat:
            for name, config in configuration.items():
                if name not in stack_list:
                    feat = Feature(name, config[0], config[1])
                    features.append(feat)
    
        for name, config in myNormalizedFeaturesConfig.items():
             
            if name.endswith("normalized"):
                normalized_functions.append(Feature(name, config[0], config[1]))

    else:
        if feat:
            for name, config in configuration.items():
                feat = Feature(name, config[0], config[1])
                features.append(feat)
    
    # binary combinations of all binary features.
    if binary_combination:
        myBinaryFeaturesConfig.update(buschmeierFeatureConfig)
        
        temp = []
        for name, config in myBinaryFeaturesConfig.items():
            temp.append(Feature(name, config[0], config[1]))
        
        if regExp:
            for name, config in REGEX_FEATURE_CONFIG_GROUPS.items():
                temp.append(RegularExpressionFeature(name, config[0], config[1]))
                
        for combi in combinations(temp,2):
            cross_product.append(combi)
    
    # add regular expressions to feature set.
    if regExp and not binary_combination:
        # Add regularExpression features
        for name, config in REGEX_FEATURE_CONFIG_SPECIAL.items():
            features.append(RegularExpressionFeature(name, config[0], config[1]))
    
    # returns a list of feature names which should be extracted.
    return features, normalized_functions, cross_product


def stacks(tweet, normalized_function):
    """ uses normalized value of feature to do stack binning. """
    
    stack = []
    name_list = []
    for function in normalized_function:
        norm = function.extract(tweet)
        
        name = function.name
        
        #Stack 1: > 0
        name_list.append(name.replace("normalized","Stack1"))
        if norm > float(0):
            stack.append(1)
        else:
            stack.append(0)
        
        #Stack 2: > 0.2
        name_list.append(name.replace("normalized","Stack2"))
        if norm > float(0.2):
            stack.append(1)
        else:
            stack.append(0)
        
        #Stack 3: > 0.4
        name_list.append(name.replace("normalized","Stack3"))
        if norm > float(0.4):
            stack.append(1)
        else:
            stack.append(0)
        
        #Stack 4: > 0.6
        name_list.append(name.replace("normalized","Stack4"))
        if norm > float(0.6):
            stack.append(1)
        else:
            stack.append(0)
        
        #Stack 5 > 0.8
        name_list.append(name.replace("normalized","Stack5"))
        if norm > float(0.8):
            stack.append(1)
        else:
            stack.append(0)    
    
    return stack, name_list


def feature_combinations(tweet, list):
    """ combinates all binary features with each other. """
    feature_list = []
    name_list = []
    
    for functions in list:
        
        function1 = functions[0]
        function2 = functions[1]
        
        feat1 = function1.extract(tweet)
        feat2 = function2.extract(tweet)
        
        name1 = function1.name
        name2 = function2.name
        
        # Case 1: Both true
        if feat1 and feat2:
            feature_list.append(1)
        else:
            feature_list.append(0)
            
        name_list.append(name1 + " and " + name2)
        
        # Case 2a: One is true 
        if feat1 and not feat2:
            feature_list.append(1)
        else:
            feature_list.append(0)
        
        name_list.append(name1 + " and not " + name2)
        
        # Case 2b: One is true
        if not feat1 and feat2:
            feature_list.append(1)
        else:
            feature_list.append(0)
        
        name_list.append("Not " + name1 + " and " + name2)
        
        # Case 3: None is true
        if not feat1 and not feat2:
            feature_list.append(1)
        else:
            feature_list.append(0)
        
        name_list.append("Not " + name1 + " and not " + name2)     

    return feature_list, name_list


def extractFeatures(class1, class2, mode, arff_path, tweetIDs, tweets, features=None, feat=False, regExp=False, stack_binning=False, binary_combination=False, sentiment_feature=False,bigram=False, createARFF="", bowDictionary=None, bigramDictionary=None):
    """Returns lists of the used features and its result."""

    if mode =='all' or mode == 'specific':
        features, normalized_functions, cross_product = createFeatures(features, feat, regExp, stack_binning, binary_combination, sentiment_feature, bigram)
    
    
    if mode == 'all' or mode == 'bow':
        if bowDictionary is None:
            bowDictionary = createBagOfWordsDictionary(tweets)
            
    if bigram and bigramDictionary is None:
        bigramDictionary = createBagOfBigramsDictionary(tweets)

    featureVectors = {}
    
    for ID in tweetIDs:
        tweet = tweets[ID]
        featureVectors[ID] = []
        
        if mode=='all' or mode == 'specific':
            # twitter specific features
            for feature in features:
                feat = feature.extract(tweets[ID])
                featureVectors[ID].append(feat)
            #print("Specific Features:", len(features))
            
            # Stack
            if stack_binning:
                stack_extraction = stacks(tweet, normalized_functions)
                stack = stack_extraction[0]
                stack_names = stack_extraction[1]
                featureVectors[ID].extend(stack)
                
            # all binary combinations of all twitter specific features.   
            if binary_combination:
                combi_extraction = feature_combinations(tweet, cross_product)
                combination = combi_extraction[0]
                combi_names = combi_extraction[1]
                featureVectors[ID].extend(combination)
             
            # Sentiment
            if sentiment_feature:
                featureVectors[ID].extend(sentiment(tweet))
            
        # Bag of words
        if mode=='all' or mode == 'bow':
            featureVectors[ID].extend(fillBagOfWords(bowDictionary, tweet))
            
        if bigram:
            # Bag of bigrams
            featureVectors[ID].extend(fillBagOfBigrams(bigramDictionary, tweet))
            
    # Save extracted features in a file.
    if len(createARFF) > 0:
        attributes = []
        
        if mode =='all' or mode == 'specific':
            attributes.extend(["\"{name}\"".format(name=feature.name).decode() for feature in features])
            if stack_binning:
                attributes.extend(["\"{name}\"".format(name=name).decode() for name in stack_names])
            if binary_combination:
                attributes.extend(["\"{name}\"".format(name=name).decode() for name in combi_names])
            if sentiment_feature:
                attributes.append("\"positive sentiment\"".decode())
        
        if mode =='all' or mode == 'bow':
            attributes.extend(["\"word={word}\"".format(word=word.encode('utf-8')).decode('utf-8') for word in sorted(bowDictionary, key=bowDictionary.get)])
        
        if bigram:
            attributes.extend(["\"bigram={bigram1}_{bigram2}\"".format(bigram1=bigram[0].encode('utf-8'), bigram2=bigram[1].encode("utf-8")).decode("utf-8") for bigram in sorted(bigramDictionary, key=bigramDictionary.get)])

        categories= {ID: str(class1) if tweets[ID].label==class1 else str(class2) 
                    for ID in tweetIDs}

        createARFFFile(class1, class2, attributes, featureVectors, categories, arff_path + createARFF)
        
    with open("info_" + createARFF + ".txt", "a") as info:
        info.write("Features: " + str(len(featureVectors[tweetIDs[0]])) + "\n")
    print("Features:", len(featureVectors[tweetIDs[0]]))
                
    return features, featureVectors


def createARFFFile(class1, class2, features, data, categories, filename="./features"):
    """Create an ARFF file, that contains the extracted features."""
    
    relation = ("@RELATION " + class1).decode()
    
    classline = ("@ATTRIBUTE class {" + str(class1) + "," + str(class2) + "}").decode()
   
    with codecs.open(filename + ".arff", "w", encoding="utf-8") as arffFile:
        arffFile.write(relation + "\n")
        
        x = 0
        allSmileys, _,_ = loadSmileys()
        
        # TODO: solve problem with encoding of quotes in ARFF (weka can not read it). 
        for feature in features:
            temp1 = "@ATTRIBUTE ".decode()
            temp2 = "\tNUMERIC\n".decode()
            out = temp1 + feature + temp2
            arffFile.write(out)                
        
        arffFile.write(classline)
        
        arffFile.write("\n@DATA\n".decode())
        
        for ID, featureVector in data.items():
            arffFile.write(",".decode().join(str(value).decode() for value in featureVector) + ",".decode() + categories[ID].decode() + "\n".decode())

## -------------------- feature analysis method -----------------------------------
def showFeatureOccurrence(features, featureVectors, gold=None, classification=None):
    """Shows the features' occurrence."""
    MAX_ID_LENGTH = 23
    MAX_NAME_LENGTH = 4
    MAX_FEATURES = 29

    print "Using the following features:"
    print ", ".join(["{0} ({1})".format(f.name, f.short) for f in features])

    headline = "ID \t\t\tCorrect | {0}".format(
                                    " ".join([f.short + " "*(4 - len(f.short))
                                            for f in features[:MAX_FEATURES]]))    
    print headline
    for ID, vec in featureVectors.items():
        print "{0}{1}{2}\t| {3}".format(ID, 
                    "\t"*(MAX_ID_LENGTH/len(ID)), 
                    "Yes " if gold and gold[ID]==classification[ID] else "___",
                    " ".join(["Yes " if v == 1 else "_"*4 for v in vec[:MAX_FEATURES]]))

    print headline
    vec = [vector[:MAX_FEATURES] for vector in featureVectors.values()]

    if not classification == None and not gold == None:
        correct = sum([1 if gold and gold[ID] == p else 0 
                        for ID, p in classification.items()])
    else:
        correct = 0

    occurrences = [sum([1 if v[i] else 0 for v in vec]) 
                    for i in range(len(features[:MAX_FEATURES]))]
    print "Summation\t\t{0}\t| {1}".format(correct,
        " ".join([" "*(4-len(str(s))) + str(s) for s in occurrences]))
        
