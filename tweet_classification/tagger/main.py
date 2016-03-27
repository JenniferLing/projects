"""
@copyright: Jennifer Ling

Tagging is done before starting classification 
(more efficient than tagging each tweet separately!!) 

For POS-Tagging each tweet of the corpus is tokenized, 
the corresponding search hashtags are removed
and the tweet is written into files (each containing 1000 tweets).
Then the twitIE jar is called and tags each file separately (-> tagger.py).

A special processing for smileys has to be used 
since TwitIE Tagger can not handle Unicode encoded smilyes.
Therefore, they will be replaced by a specific unique character combination
before tagging and re-replaced after tagging. 
All smileys handled with this process get the tag SMILEY.
"""

import os
import shutil
from time import *
import codecs, re, sys
import itertools
import tagger

from tokenizer import MyTokenizer

# ---- Smiley Encodings ----
sm = [u"\U0001F601", u"\U0001F602", u"\U0001F603", u"\U0001F604", u"\U0001F605", u"\U0001F606", u"\U0001F609", u"\U0001F60A", 
          u"\U0001F60B", u"\U0001F60C", u"\U0001F60D", u"\U0001F60F", u"\U0001F612", u"\U0001F613", u"\U0001F614", u"\U0001F616", 
          u"\U0001F618", u"\U0001F61A", u"\U0001F61C", u"\U0001F61D", u"\U0001F61E", u"\U0001F620", u"\U0001F621", u"\U0001F622", 
          u"\U0001F623", u"\U0001F624", u"\U0001F625", u"\U0001F628", u"\U0001F629", u"\U0001F62A", u"\U0001F62B", u"\U0001F62D", 
          u"\U0001F630", u"\U0001F631", u"\U0001F632", u"\U0001F633", u"\U0001F635", u"\U0001F637", u"\U0001F638", u"\U0001F639", 
          u"\U0001F63A", u"\U0001F63B", u"\U0001F63C", u"\U0001F63D", u"\U0001F63E", u"\U0001F63F", u"\U0001F640", u"\U0001F645", 
          u"\U0001F646", u"\U0001F647", u"\U0001F648", u"\U0001F649", u"\U0001F64A", u"\U0001F64B", u"\U0001F64C", u"\U0001F64D", 
          u"\U0001F64E", u"\U0001F64F", u"\U00002702", u"\U00002705", u"\U00002708", u"\U00002709", u"\U0000270A", u"\U0000270B", 
          u"\U0000270C", u"\U0000270F", u"\U00002712", u"\U00002714", u"\U00002716", u"\U00002728", u"\U00002733", u"\U00002734", 
          u"\U00002744", u"\U00002747", u"\U0000274C", u"\U0000274E", u"\U00002753", u"\U00002754", u"\U00002755", u"\U00002757", 
          u"\U00002764", u"\U00002795", u"\U00002796", u"\U00002797", u"\U000027A1", u"\U000027B0", u"\U0001F680", u"\U0001F683", 
          u"\U0001F684", u"\U0001F685", u"\U0001F687", u"\U0001F689", u"\U0001F68C", u"\U0001F68F", u"\U0001F691", u"\U0001F692", 
          u"\U0001F693", u"\U0001F695", u"\U0001F697", u"\U0001F699", u"\U0001F69A", u"\U0001F6A2", u"\U0001F6A4", u"\U0001F6A5", 
          u"\U0001F6A7", u"\U0001F6A8", u"\U0001F6A9", u"\U0001F6AA", u"\U0001F6AB", u"\U0001F6AC", u"\U0001F6AD", u"\U0001F6B2", 
          u"\U0001F6B6", u"\U0001F6B9", u"\U0001F6BA", u"\U0001F6BB", u"\U0001F6BC", u"\U0001F6BD", u"\U0001F6BE", u"\U0001F6C0", 
          u"\U000024C2", u"\U0001F170", u"\U0001F171", u"\U0001F17E", u"\U0001F17F", u"\U0001F18E", u"\U0001F191", u"\U0001F192", 
          u"\U0001F193", u"\U0001F194", u"\U0001F195", u"\U0001F196", u"\U0001F197", u"\U0001F198", u"\U0001F199", u"\U0001F19A", 
          u"\U0001F1E9 \U0001F1EA", u"\U0001F1EC \U0001F1E7", u"\U0001F1E8 \U0001F1F3", u"\U0001F1EF \U0001F1F5", u"\U0001F1F0 \U0001F1F7", 
          u"\U0001F1EB \U0001F1F7", u"\U0001F1EA \U0001F1F8", u"\U0001F1EE \U0001F1F9", u"\U0001F1FA \U0001F1F8", u"\U0001F1F7 \U0001F1FA", 
          u"\U0001F201", u"\U0001F202", u"\U0001F21A", u"\U0001F22F", u"\U0001F232", u"\U0001F233", u"\U0001F234", u"\U0001F235", 
          u"\U0001F236", u"\U0001F237", u"\U0001F238", u"\U0001F239", u"\U0001F23A", u"\U0001F250", u"\U0001F251", u"\U000000A9", 
          u"\U000000AE", u"\U0000203C", u"\U00002049", u"\U00000038 \U000020E3", u"\U00000039 \U000020E3", u"\U00000037 \U000020E3", 
          u"\U00000036 \U000020E3", u"\U00000031 \U000020E3", u"\U00000030 \U000020E3", u"\U00000032 \U000020E3", u"\U00000033 \U000020E3", 
          u"\U00000035 \U000020E3", u"\U00000034 \U000020E3", u"\U00000023 \U000020E3", u"\U00002122", u"\U00002139", u"\U00002194", 
          u"\U00002195", u"\U00002196", u"\U00002197", u"\U00002198", u"\U00002199", u"\U000021A9", u"\U000021AA", u"\U0000231A", 
          u"\U0000231B", u"\U000023E9", u"\U000023EA", u"\U000023EB", u"\U000023EC", u"\U000023F0", u"\U000023F3", u"\U000025AA", 
          u"\U000025AB", u"\U000025B6", u"\U000025C0", u"\U000025FB", u"\U000025FC", u"\U000025FD", u"\U000025FE", u"\U00002600", 
          u"\U00002601", u"\U0000260E", u"\U00002611", u"\U00002614", u"\U00002615", u"\U0000261D", u"\U0000263A", u"\U00002648", 
          u"\U00002649", u"\U0000264A", u"\U0000264B", u"\U0000264C", u"\U0000264D", u"\U0000264E", u"\U0000264F", u"\U00002650", 
          u"\U00002651", u"\U00002652", u"\U00002653", u"\U00002660", u"\U00002663", u"\U00002665", u"\U00002666", u"\U00002668", 
          u"\U0000267B", u"\U0000267F", u"\U00002693", u"\U000026A0", u"\U000026A1", u"\U000026AA", u"\U000026AB", u"\U000026BD", 
          u"\U000026BE", u"\U000026C4", u"\U000026C5", u"\U000026CE", u"\U000026D4", u"\U000026EA", u"\U000026F2", u"\U000026F3", 
          u"\U000026F5", u"\U000026FA", u"\U000026FD", u"\U00002934", u"\U00002935", u"\U00002B05", u"\U00002B06", u"\U00002B07",          
          u"\U00002B1B", u"\U00002B1C", u"\U00002B50", u"\U00002B55", u"\U00003030", u"\U0000303D", u"\U00003297", u"\U00003299", 
          u"\U0001F004", u"\U0001F0CF", u"\U0001F300", u"\U0001F301", u"\U0001F302", u"\U0001F303", u"\U0001F304", u"\U0001F305", 
          u"\U0001F306", u"\U0001F307", u"\U0001F308", u"\U0001F309", u"\U0001F30A", u"\U0001F30B", u"\U0001F30C", u"\U0001F30F", 
          u"\U0001F311", u"\U0001F313", u"\U0001F314", u"\U0001F315", u"\U0001F319", u"\U0001F31B", u"\U0001F31F", u"\U0001F320", 
          u"\U0001F330", u"\U0001F331", u"\U0001F334", u"\U0001F335", u"\U0001F337", u"\U0001F338", u"\U0001F339", u"\U0001F33A", 
          u"\U0001F33B", u"\U0001F33C", u"\U0001F33D", u"\U0001F33E", u"\U0001F33F", u"\U0001F340", u"\U0001F341", u"\U0001F342", 
          u"\U0001F343", u"\U0001F344", u"\U0001F345", u"\U0001F346", u"\U0001F347", u"\U0001F348", u"\U0001F349", u"\U0001F34A", 
          u"\U0001F34C", u"\U0001F34D", u"\U0001F34E", u"\U0001F34F", u"\U0001F351", u"\U0001F352", u"\U0001F353", u"\U0001F354", 
          u"\U0001F355", u"\U0001F356", u"\U0001F357", u"\U0001F358", u"\U0001F359", u"\U0001F35A", u"\U0001F35B", u"\U0001F35C", 
          u"\U0001F35D", u"\U0001F35E", u"\U0001F35F", u"\U0001F360", u"\U0001F361", u"\U0001F362", u"\U0001F363", u"\U0001F364", 
          u"\U0001F365", u"\U0001F366", u"\U0001F367", u"\U0001F368", u"\U0001F369", u"\U0001F36A", u"\U0001F36B", u"\U0001F36C", 
          u"\U0001F36D", u"\U0001F36E", u"\U0001F36F", u"\U0001F370", u"\U0001F371", u"\U0001F372", u"\U0001F373", u"\U0001F374", 
          u"\U0001F375", u"\U0001F376", u"\U0001F377", u"\U0001F378", u"\U0001F379", u"\U0001F37A", u"\U0001F37B", u"\U0001F380", 
          u"\U0001F381", u"\U0001F382", u"\U0001F383", u"\U0001F384", u"\U0001F385", u"\U0001F386", u"\U0001F387", u"\U0001F388", 
          u"\U0001F389", u"\U0001F38A", u"\U0001F38B", u"\U0001F38C", u"\U0001F38D", u"\U0001F38E", u"\U0001F38F", u"\U0001F390", 
          u"\U0001F391", u"\U0001F392", u"\U0001F393", u"\U0001F3A0", u"\U0001F3A1", u"\U0001F3A2", u"\U0001F3A3", u"\U0001F3A4", 
          u"\U0001F3A5", u"\U0001F3A6", u"\U0001F3A7", u"\U0001F3A8", u"\U0001F3A9", u"\U0001F3AA", u"\U0001F3AB", u"\U0001F3AC", 
          u"\U0001F3AD", u"\U0001F3AE", u"\U0001F3AF", u"\U0001F3B0", u"\U0001F3B1", u"\U0001F3B2", u"\U0001F3B3", u"\U0001F3B4", 
          u"\U0001F3B5", u"\U0001F3B6", u"\U0001F3B7", u"\U0001F3B8", u"\U0001F3B9", u"\U0001F3BA", u"\U0001F3BB", u"\U0001F3BC", 
          u"\U0001F3BD", u"\U0001F3BE", u"\U0001F3BF", u"\U0001F3C0", u"\U0001F3C1", u"\U0001F3C2", u"\U0001F3C3", u"\U0001F3C4", 
          u"\U0001F3C6", u"\U0001F3C8", u"\U0001F3CA", u"\U0001F3E0", u"\U0001F3E1", u"\U0001F3E2", u"\U0001F3E3", u"\U0001F3E5", 
          u"\U0001F3E6", u"\U0001F3E7", u"\U0001F3E8", u"\U0001F3E9", u"\U0001F3EA", u"\U0001F3EB", u"\U0001F3EC", u"\U0001F3ED", 
          u"\U0001F3EE", u"\U0001F3EF", u"\U0001F3F0", u"\U0001F40C", u"\U0001F40D", u"\U0001F40E", u"\U0001F411", u"\U0001F412",
          u"\U0001F414", u"\U0001F417", u"\U0001F418", u"\U0001F419", u"\U0001F41A", u"\U0001F41B", u"\U0001F41C", u"\U0001F41D", 
          u"\U0001F41E", u"\U0001F41F", u"\U0001F420", u"\U0001F421", u"\U0001F422", u"\U0001F423", u"\U0001F424", u"\U0001F425", 
          u"\U0001F426", u"\U0001F427", u"\U0001F428", u"\U0001F429", u"\U0001F42B", u"\U0001F42C", u"\U0001F42D", u"\U0001F42E", 
          u"\U0001F42F", u"\U0001F430", u"\U0001F431", u"\U0001F432", u"\U0001F433", u"\U0001F434", u"\U0001F435", u"\U0001F436", 
          u"\U0001F437", u"\U0001F438", u"\U0001F439", u"\U0001F43A", u"\U0001F43B", u"\U0001F43C", u"\U0001F43D", u"\U0001F43E", 
          u"\U0001F440", u"\U0001F442", u"\U0001F443", u"\U0001F444", u"\U0001F445", u"\U0001F446", u"\U0001F447", u"\U0001F448", 
          u"\U0001F449", u"\U0001F44A", u"\U0001F44B", u"\U0001F44C", u"\U0001F44D", u"\U0001F44E", u"\U0001F44F", u"\U0001F450", 
          u"\U0001F451", u"\U0001F452", u"\U0001F453", u"\U0001F454", u"\U0001F455", u"\U0001F456", u"\U0001F457", u"\U0001F458", 
          u"\U0001F459", u"\U0001F45A", u"\U0001F45B", u"\U0001F45C", u"\U0001F45D", u"\U0001F45E", u"\U0001F45F", u"\U0001F460", 
          u"\U0001F461", u"\U0001F462", u"\U0001F463", u"\U0001F464", u"\U0001F466", u"\U0001F467", u"\U0001F468", u"\U0001F469", 
          u"\U0001F46A", u"\U0001F46B", u"\U0001F46E", u"\U0001F46F", u"\U0001F470", u"\U0001F471", u"\U0001F472", u"\U0001F473", 
          u"\U0001F474", u"\U0001F475", u"\U0001F476", u"\U0001F477", u"\U0001F478", u"\U0001F479", u"\U0001F47A", u"\U0001F47B", 
          u"\U0001F47C", u"\U0001F47D", u"\U0001F47E", u"\U0001F47F", u"\U0001F480", u"\U0001F481", u"\U0001F482", u"\U0001F483", 
          u"\U0001F484", u"\U0001F485", u"\U0001F486", u"\U0001F487", u"\U0001F488", u"\U0001F489", u"\U0001F48A", u"\U0001F48B", 
          u"\U0001F48C", u"\U0001F48D", u"\U0001F48E", u"\U0001F48F", u"\U0001F490", u"\U0001F491", u"\U0001F492", u"\U0001F493", 
          u"\U0001F494", u"\U0001F495", u"\U0001F496", u"\U0001F497", u"\U0001F498", u"\U0001F499", u"\U0001F49A", u"\U0001F49B", 
          u"\U0001F49C", u"\U0001F49D", u"\U0001F49E", u"\U0001F49F", u"\U0001F4A0", u"\U0001F4A1", u"\U0001F4A2", u"\U0001F4A3", 
          u"\U0001F4A4", u"\U0001F4A5", u"\U0001F4A6", u"\U0001F4A7", u"\U0001F4A8", u"\U0001F4A9", u"\U0001F4AA", u"\U0001F4AB", 
          u"\U0001F4AC", u"\U0001F4AE", u"\U0001F4AF", u"\U0001F4B0", u"\U0001F4B1", u"\U0001F4B2", u"\U0001F4B3", u"\U0001F4B4", 
          u"\U0001F4B5", u"\U0001F4B8", u"\U0001F4B9", u"\U0001F4BA", u"\U0001F4BB", u"\U0001F4BC", u"\U0001F4BD", u"\U0001F4BE", 
          u"\U0001F4BF", u"\U0001F4C0", u"\U0001F4C1", u"\U0001F4C2", u"\U0001F4C3", u"\U0001F4C4", u"\U0001F4C5", u"\U0001F4C6", 
          u"\U0001F4C7", u"\U0001F4C8", u"\U0001F4C9", u"\U0001F4CA", u"\U0001F4CB", u"\U0001F4CC", u"\U0001F4CD", u"\U0001F4CE", 
          u"\U0001F4CF", u"\U0001F4D0", u"\U0001F4D1", u"\U0001F4D2", u"\U0001F4D3", u"\U0001F4D4", u"\U0001F4D5", u"\U0001F4D6", 
          u"\U0001F4D7", u"\U0001F4D8", u"\U0001F4D9", u"\U0001F4DA", u"\U0001F4DB", u"\U0001F4DC", u"\U0001F4DD", u"\U0001F4DE", 
          u"\U0001F4DF", u"\U0001F4E0", u"\U0001F4E1", u"\U0001F4E2", u"\U0001F4E3", u"\U0001F4E4", u"\U0001F4E5", u"\U0001F4E6", 
          u"\U0001F4E7", u"\U0001F4E8", u"\U0001F4E9", u"\U0001F4EA", u"\U0001F4EB", u"\U0001F4EE", u"\U0001F4F0", u"\U0001F4F1", 
          u"\U0001F4F2", u"\U0001F4F3", u"\U0001F4F4", u"\U0001F4F6", u"\U0001F4F7", u"\U0001F4F9", u"\U0001F4FA", u"\U0001F4FB", 
          u"\U0001F4FC", u"\U0001F503", u"\U0001F50A", u"\U0001F50B", u"\U0001F50C", u"\U0001F50D", u"\U0001F50E", u"\U0001F50F", 
          u"\U0001F510", u"\U0001F511", u"\U0001F512", u"\U0001F513", u"\U0001F514", u"\U0001F516", u"\U0001F517", u"\U0001F518", 
          u"\U0001F519", u"\U0001F51A", u"\U0001F51B", u"\U0001F51C", u"\U0001F51D", u"\U0001F51E", u"\U0001F51F", u"\U0001F520", 
          u"\U0001F521", u"\U0001F522", u"\U0001F523", u"\U0001F524", u"\U0001F525", u"\U0001F526", u"\U0001F527", u"\U0001F528", 
          u"\U0001F529", u"\U0001F52A", u"\U0001F52B", u"\U0001F52E", u"\U0001F52F", u"\U0001F530", u"\U0001F531", u"\U0001F532", 
          u"\U0001F533", u"\U0001F534", u"\U0001F535", u"\U0001F536", u"\U0001F537", u"\U0001F538", u"\U0001F539", u"\U0001F53A", 
          u"\U0001F53B", u"\U0001F53C", u"\U0001F53D", u"\U0001F550", u"\U0001F551", u"\U0001F552", u"\U0001F553", u"\U0001F554", 
          u"\U0001F555", u"\U0001F556", u"\U0001F557", u"\U0001F558", u"\U0001F559", u"\U0001F55A", u"\U0001F55B", u"\U0001F5FB", 
          u"\U0001F5FC", u"\U0001F5FD", u"\U0001F5FE", u"\U0001F5FF", u"\U0001F600", u"\U0001F607", u"\U0001F608", u"\U0001F60E", 
          u"\U0001F610", u"\U0001F611", u"\U0001F615", u"\U0001F617", u"\U0001F619", u"\U0001F61B", u"\U0001F61F", u"\U0001F626", 
          u"\U0001F627", u"\U0001F62C", u"\U0001F62E", u"\U0001F62F", u"\U0001F634", u"\U0001F636", u"\U0001F681", u"\U0001F682", 
          u"\U0001F686", u"\U0001F688", u"\U0001F68A", u"\U0001F68D", u"\U0001F68E", u"\U0001F690", u"\U0001F694", u"\U0001F696", 
          u"\U0001F698", u"\U0001F69B", u"\U0001F69C", u"\U0001F69D", u"\U0001F69E", u"\U0001F69F", u"\U0001F6A0", u"\U0001F6A1", 
          u"\U0001F6A3", u"\U0001F6A6", u"\U0001F6AE", u"\U0001F6AF", u"\U0001F6B0", u"\U0001F6B1", u"\U0001F6B3", u"\U0001F6B4", 
          u"\U0001F6B5", u"\U0001F6B7", u"\U0001F6B8", u"\U0001F6BF", u"\U0001F6C1", u"\U0001F6C2", u"\U0001F6C3", u"\U0001F6C4", 
          u"\U0001F6C5", u"\U0001F30D", u"\U0001F30E", u"\U0001F310", u"\U0001F312", u"\U0001F316", u"\U0001F317", u"\U0001F318", 
          u"\U0001F31A", u"\U0001F31C", u"\U0001F31D", u"\U0001F31E", u"\U0001F332", u"\U0001F333", u"\U0001F34B", u"\U0001F350", 
          u"\U0001F37C", u"\U0001F3C7", u"\U0001F3C9", u"\U0001F3E4", u"\U0001F400", u"\U0001F401", u"\U0001F402", u"\U0001F403", 
          u"\U0001F404", u"\U0001F405", u"\U0001F406", u"\U0001F407", u"\U0001F408", u"\U0001F409", u"\U0001F40A", u"\U0001F40B", 
          u"\U0001F40F", u"\U0001F410", u"\U0001F413", u"\U0001F415", u"\U0001F416", u"\U0001F42A", u"\U0001F465", u"\U0001F46C", 
          u"\U0001F46D", u"\U0001F4AD", u"\U0001F4B6", u"\U0001F4B7", u"\U0001F4EC", u"\U0001F4ED", u"\U0001F4EF", u"\U0001F4F5", 
          u"\U0001F500", u"\U0001F501", u"\U0001F502", u"\U0001F504", u"\U0001F505", u"\U0001F506", u"\U0001F507", u"\U0001F509", 
          u"\U0001F515", u"\U0001F52C", u"\U0001F52D", u"\U0001F55C", u"\U0001F55D", u"\U0001F55E", u"\U0001F55F", u"\U0001F560", 
          u"\U0001F561", u"\U0001F562", u"\U0001F563", u"\U0001F564", u"\U0001F565", u"\U0001F566", u"\U0001F567"]

# ------------------------------------------------------------------------------------------------------------------------------
# constants:
TEMP_PATH_FOR_TAGGED_FILES = "./files_for_tagging/"
PATH_FOR_TAGGED_FILES = "../../corpora/tagged_tweets/"

class PreprocessForTagging():    
    """
    Tokenize text and remove search hashtags.
    """
    
    def __init__(self):
        # load Tokenizer.
        self.tok = MyTokenizer(preserve_case=True)
        self.tag_file_counter = 0
        self.tag_file_closed = True
        
    def main(self):
        """ main method for preprocessing where all paths are defined"""
        
        self.nr_tag_files = len(os.listdir(TEMP_PATH_FOR_TAGGED_FILES))+1
                
        pre_path = "../../corpora/"
        folder1 = ["test/", "train/"]
        #folder2 = ["irony", "sarcasm", "regular", "figurative"]
        folder1 = ["test_3000/"]
        folder2 = ["irony", "sarcasm"]
        
        for folder in folder1:
            for label in folder2:
                self.path = pre_path + folder + label + ".csv"
                
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
                
                self.hashtags = remove_hashtags

                self.preprocess()
    
        print "Finished preprocessing!"
             
    def preprocess(self, id_files=False):
        """
        checks if corpus consists of id files or 
        one file with all tweets for one label
        and calls corresponding method.
        """
        
        if not id_files:
            self.readIn()
        else:
            file_list = os.listdir(self.path)    
     
            for filename in file_list:
                self.readIn(filename)
                
    def readIn(self, filename=None):
        """ reads tweets from file(s) """
        
        # corpus consists of one file per label
        if not filename:
            with codecs.open(self.path, "r", "utf-8") as input:
                for line in input:
                    tweet = line.strip().split("\t")
                    tweet_text = tweet[4]
                    tweet_id = tweet[3]
                    self.writeTagFile(tweet_id, tweet_text)
        
        # corpus consists of id files where every tweet is one file
        # (not that efficient for further processing)
        else:
            with codecs.open(filename, "r",encoding='utf-8') as text:
                tweet = text.read().split("\n")[:-1]
            tweet_text = tweet[4]
            tweet_id = tweet[3]
            self.writeTagFile(tweet_id, tweet_text)
                
    def writeTagFile(self, tweet_id, tweet_text):
        """ writes modified tweets to file which is used for tagging """
        
        # open new tag file
        if self.tag_file_closed:
            self.tag_file = codecs.open(TEMP_PATH_FOR_TAGGED_FILES + "preprocessed_" + str(self.nr_tag_files) + ".txt", "w", encoding='utf-8')
            self.tag_file_closed = False
            self.tag_file_counter = 0
        
        # write tweet id in tag file for unique identification of tweets
        self.tag_file.write(tweet_id + "\n")  
    
        # replacement of smileys
        for i in range(len(sm)):
            if sm[i] in tweet_text:
                tweet_text = tweet_text.replace(sm[i], u" x" + str((i+1)).decode("cp1252") + u"zSMILEY ")
        
        # remove search hashtags (necessary to remove labels for machine learning)
        text = tweet_text
        text = text.split()
        for hashtag in self.hashtags:
            for i in range(len(text)):
                if text[i].lower().startswith(hashtag):
                    text.remove(text[i])
                    break

        text = " ".join(text)
        
        # tokenize the modified tweet; preprocessing step before tagging             
        tokenized_text = self.tok.tokenize(text)
        for token in tokenized_text[:len(tokenized_text)-1]:
            self.tag_file.write(token + " ")
        
        # write to tag file   
        self.tag_file.write(tokenized_text[len(tokenized_text)-1] + "\n")
        
        self.tag_file_counter += 1
        
        # open new tag file every 1000 tweets
        if self.tag_file_counter >= 1000:
            self.tag_file.close()
            self.tag_file_closed = True
            self.nr_tag_files += 1
            

def pos_tagger_filelist(input_path=TEMP_PATH_FOR_TAGGED_FILES):
    """ calls tagger and creates tagged files for a list of files """    
    file_list = os.listdir(input_path)

    for datei in file_list:
                 
        tagger.runFile(input_path + datei, TEMP_PATH_FOR_TAGGED_FILES + "tagged_" + datei)


def pos_tagger_single_file(path, inputfile):
    """ calls tagger and creates tagged file """
    tagger.runFile(path + inputfile, TEMP_PATH_FOR_TAGGED_FILES + "tagged_" + inputfile)
   

def replaceTempNumbers(path=TEMP_PATH_FOR_TAGGED_FILES):
    """ reverse the replacement of smileys to get original tweets in tagged form """
    file_list = os.listdir(path)
    
    for filename in file_list:
        if filename.startswith("tagged_"):
            with codecs.open(TEMP_PATH_FOR_TAGGED_FILES + filename, "r",encoding='utf-8') as text:
                with codecs.open(PATH_FOR_TAGGED_FILES + "tagged_" + filename[-5:], "w", encoding='utf-8') as output:
                    text = text.read().split("\n")[:-1]
                    for i in range(0,len(text)):
                                          
                        if i % 2 == 0:
                            output.write(text[i].strip() + "\n")
                            continue
                        else:
                            sent = text[i].strip()
                            
                            words = sent.split()
                            for i in range(len(words)):
                                if words[i].split("_")[0] == words[i]:
                                    words[i] = words[i] + u"_X"
                                     
                            sent = " ".join(words)
                                                            
                            for word in sent.split():
                                
                                if word.split("_")[0].startswith("x") and word.split("_")[0].endswith("zSMILEY"):
                                    try:
                                        w = re.findall(r"x([0-9]*)zSMILEY", word)[0]
                                         
                                        sent = sent.replace(word, sm[int(w)-1] +"_SMILEY")
                                    except:
                                        continue
    
                            output.write(sent + "\n")

def clearFolders():
    """ remove old files to start a complete new tag process """
    files = os.listdir(TEMP_PATH_FOR_TAGGED_FILES)
    
    for filename in files:
        os.remove(TEMP_PATH_FOR_TAGGED_FILES + filename)

if __name__ == "__main__":
    
    clearFolders()
    
    pre = PreprocessForTagging()
    pre.main()
         
    pos_tagger_filelist()
    replaceTempNumbers()          
            
    