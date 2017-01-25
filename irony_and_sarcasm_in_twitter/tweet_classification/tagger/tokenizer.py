#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@copyright: Jennifer Ling
Adapted from Christopher Potts (http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py)
"""

######################################################################

import re,codecs
import htmlentitydefs

# The components of the tokenizer:
# all emoticons from http://apps.timwhitlock.info/emoji/tables/unicode and general tokenize points.
regex_strings = (ur"""\U0001F601""",)
regex_strings += (ur"""[:=8;\*)][-o\*\']?[\)\(\{\}\</@\|\\\>\]DdPpOo\*]""",) # punctuation smileys
regex_strings += (ur"""[\*DdPpOo\)\(\{\}\</@\|\\\>\]][-o\*\']?[;:=8\*]""",) # punctuation smileys
regex_strings += (ur"""\*{1}[(\w\s)]+\*{1}""",) # words with asterisks: *grin*
regex_strings += (ur"""\^_*\^|\^[-oO]?\^""",) # punctuation smileys
regex_strings += (ur"""\U0001F602""",)
regex_strings += (ur"""\U0001F603""",)
regex_strings += (ur"""\U0001F604""",)
regex_strings += (ur"""\U0001F605""",)
regex_strings += (ur"""\U0001F606""",)
regex_strings += (ur"""\U0001F609""",)
regex_strings += (ur"""\U0001F60A""",)
regex_strings += (ur"""\U0001F60B""",)
regex_strings += (ur"""\U0001F60C""",)
regex_strings += (ur"""\U0001F60D""",)
regex_strings += (ur"""\U0001F60F""",)
regex_strings += (ur"""\U0001F612""",)
regex_strings += (ur"""\U0001F613""",)
regex_strings += (ur"""\U0001F614""",)
regex_strings += (ur"""\U0001F616""",)
regex_strings += (ur"""\U0001F618""",)
regex_strings += (ur"""\U0001F61A""",)
regex_strings += (ur"""\U0001F61C""",)
regex_strings += (ur"""\U0001F61D""",)
regex_strings += (ur"""\U0001F61E""",)
regex_strings += (ur"""\U0001F620""",)
regex_strings += (ur"""\U0001F621""",)
regex_strings += (ur"""\U0001F622""",)
regex_strings += (ur"""\U0001F623""",)
regex_strings += (ur"""\U0001F624""",)
regex_strings += (ur"""\U0001F625""",)
regex_strings += (ur"""\U0001F628""",)
regex_strings += (ur"""\U0001F629""",)
regex_strings += (ur"""\U0001F62A""",)
regex_strings += (ur"""\U0001F62B""",)
regex_strings += (ur"""\U0001F62D""",)
regex_strings += (ur"""\U0001F630""",)
regex_strings += (ur"""\U0001F631""",)
regex_strings += (ur"""\U0001F632""",)
regex_strings += (ur"""\U0001F633""",)
regex_strings += (ur"""\U0001F635""",)
regex_strings += (ur"""\U0001F637""",)
regex_strings += (ur"""\U0001F638""",)
regex_strings += (ur"""\U0001F639""",)
regex_strings += (ur"""\U0001F63A""",)
regex_strings += (ur"""\U0001F63B""",)
regex_strings += (ur"""\U0001F63C""",)
regex_strings += (ur"""\U0001F63D""",)
regex_strings += (ur"""\U0001F63E""",)
regex_strings += (ur"""\U0001F63F""",)
regex_strings += (ur"""\U0001F640""",)
regex_strings += (ur"""\U0001F645""",)
regex_strings += (ur"""\U0001F646""",)
regex_strings += (ur"""\U0001F647""",)
regex_strings += (ur"""\U0001F648""",)
regex_strings += (ur"""\U0001F649""",)
regex_strings += (ur"""\U0001F64A""",)
regex_strings += (ur"""\U0001F64B""",)
regex_strings += (ur"""\U0001F64C""",)
regex_strings += (ur"""\U0001F64D""",)
regex_strings += (ur"""\U0001F64E""",)
regex_strings += (ur"""\U0001F64F""",)
regex_strings += (ur"""\U00002702""",)
regex_strings += (ur"""\U00002705""",)
regex_strings += (ur"""\U00002708""",)
regex_strings += (ur"""\U00002709""",)
regex_strings += (ur"""\U0000270A""",)
regex_strings += (ur"""\U0000270B""",)
regex_strings += (ur"""\U0000270C""",)
regex_strings += (ur"""\U0000270F""",)
regex_strings += (ur"""\U00002712""",)
regex_strings += (ur"""\U00002714""",)
regex_strings += (ur"""\U00002716""",)
regex_strings += (ur"""\U00002728""",)
regex_strings += (ur"""\U00002733""",)
regex_strings += (ur"""\U00002734""",)
regex_strings += (ur"""\U00002744""",)
regex_strings += (ur"""\U00002747""",)
regex_strings += (ur"""\U0000274C""",)
regex_strings += (ur"""\U0000274E""",)
regex_strings += (ur"""\U00002753""",)
regex_strings += (ur"""\U00002754""",)
regex_strings += (ur"""\U00002755""",)
regex_strings += (ur"""\U00002757""",)
regex_strings += (ur"""\U00002764""",)
regex_strings += (ur"""\U00002795""",)
regex_strings += (ur"""\U00002796""",)
regex_strings += (ur"""\U00002797""",)
regex_strings += (ur"""\U000027A1""",)
regex_strings += (ur"""\U000027B0""",)
regex_strings += (ur"""\U0001F680""",)
regex_strings += (ur"""\U0001F683""",)
regex_strings += (ur"""\U0001F684""",)
regex_strings += (ur"""\U0001F685""",)
regex_strings += (ur"""\U0001F687""",)
regex_strings += (ur"""\U0001F689""",)
regex_strings += (ur"""\U0001F68C""",)
regex_strings += (ur"""\U0001F68F""",)
regex_strings += (ur"""\U0001F691""",)
regex_strings += (ur"""\U0001F692""",)
regex_strings += (ur"""\U0001F693""",)
regex_strings += (ur"""\U0001F695""",)
regex_strings += (ur"""\U0001F697""",)
regex_strings += (ur"""\U0001F699""",)
regex_strings += (ur"""\U0001F69A""",)
regex_strings += (ur"""\U0001F6A2""",)
regex_strings += (ur"""\U0001F6A4""",)
regex_strings += (ur"""\U0001F6A5""",)
regex_strings += (ur"""\U0001F6A7""",)
regex_strings += (ur"""\U0001F6A8""",)
regex_strings += (ur"""\U0001F6A9""",)
regex_strings += (ur"""\U0001F6AA""",)
regex_strings += (ur"""\U0001F6AB""",)
regex_strings += (ur"""\U0001F6AC""",)
regex_strings += (ur"""\U0001F6AD""",)
regex_strings += (ur"""\U0001F6B2""",)
regex_strings += (ur"""\U0001F6B6""",)
regex_strings += (ur"""\U0001F6B9""",)
regex_strings += (ur"""\U0001F6BA""",)
regex_strings += (ur"""\U0001F6BB""",)
regex_strings += (ur"""\U0001F6BC""",)
regex_strings += (ur"""\U0001F6BD""",)
regex_strings += (ur"""\U0001F6BE""",)
regex_strings += (ur"""\U0001F6C0""",)
regex_strings += (ur"""\U000024C2""",)
regex_strings += (ur"""\U0001F170""",)
regex_strings += (ur"""\U0001F171""",)
regex_strings += (ur"""\U0001F17E""",)
regex_strings += (ur"""\U0001F17F""",)
regex_strings += (ur"""\U0001F18E""",)
regex_strings += (ur"""\U0001F191""",)
regex_strings += (ur"""\U0001F192""",)
regex_strings += (ur"""\U0001F193""",)
regex_strings += (ur"""\U0001F194""",)
regex_strings += (ur"""\U0001F195""",)
regex_strings += (ur"""\U0001F196""",)
regex_strings += (ur"""\U0001F197""",)
regex_strings += (ur"""\U0001F198""",)
regex_strings += (ur"""\U0001F199""",)
regex_strings += (ur"""\U0001F19A""",)
regex_strings += (ur"""\U0001F1E9 \U0001F1EA""",)
regex_strings += (ur"""\U0001F1EC \U0001F1E7""",)
regex_strings += (ur"""\U0001F1E8 \U0001F1F3""",)
regex_strings += (ur"""\U0001F1EF \U0001F1F5""",)
regex_strings += (ur"""\U0001F1F0 \U0001F1F7""",)
regex_strings += (ur"""\U0001F1EB \U0001F1F7""",)
regex_strings += (ur"""\U0001F1EA \U0001F1F8""",)
regex_strings += (ur"""\U0001F1EE \U0001F1F9""",)
regex_strings += (ur"""\U0001F1FA \U0001F1F8""",)
regex_strings += (ur"""\U0001F1F7 \U0001F1FA""",)
regex_strings += (ur"""\U0001F201""",)
regex_strings += (ur"""\U0001F202""",)
regex_strings += (ur"""\U0001F21A""",)
regex_strings += (ur"""\U0001F22F""",)
regex_strings += (ur"""\U0001F232""",)
regex_strings += (ur"""\U0001F233""",)
regex_strings += (ur"""\U0001F234""",)
regex_strings += (ur"""\U0001F235""",)
regex_strings += (ur"""\U0001F236""",)
regex_strings += (ur"""\U0001F237""",)
regex_strings += (ur"""\U0001F238""",)
regex_strings += (ur"""\U0001F239""",)
regex_strings += (ur"""\U0001F23A""",)
regex_strings += (ur"""\U0001F250""",)
regex_strings += (ur"""\U0001F251""",)
regex_strings += (ur"""\U000000A9""",)
regex_strings += (ur"""\U000000AE""",)
regex_strings += (ur"""\U0000203C""",)
regex_strings += (ur"""\U00002049""",)
regex_strings += (ur"""\U00000038 \U000020E3""",)
regex_strings += (ur"""\U00000039 \U000020E3""",)
regex_strings += (ur"""\U00000037 \U000020E3""",)
regex_strings += (ur"""\U00000036 \U000020E3""",)
regex_strings += (ur"""\U00000031 \U000020E3""",)
regex_strings += (ur"""\U00000030 \U000020E3""",)
regex_strings += (ur"""\U00000032 \U000020E3""",)
regex_strings += (ur"""\U00000033 \U000020E3""",)
regex_strings += (ur"""\U00000035 \U000020E3""",)
regex_strings += (ur"""\U00000034 \U000020E3""",)
# regex_strings += (ur"""\U00000023 \U000020E3""",)
regex_strings += (ur"""\U00002122""",)
regex_strings += (ur"""\U00002139""",)
regex_strings += (ur"""\U00002194""",)
regex_strings += (ur"""\U00002195""",)
regex_strings += (ur"""\U00002196""",)
regex_strings += (ur"""\U00002197""",)
regex_strings += (ur"""\U00002198""",)
regex_strings += (ur"""\U00002199""",)
regex_strings += (ur"""\U000021A9""",)
regex_strings += (ur"""\U000021AA""",)
regex_strings += (ur"""\U0000231A""",)
regex_strings += (ur"""\U0000231B""",)
regex_strings += (ur"""\U000023E9""",)
regex_strings += (ur"""\U000023EA""",)
regex_strings += (ur"""\U000023EB""",)
regex_strings += (ur"""\U000023EC""",)
regex_strings += (ur"""\U000023F0""",)
regex_strings += (ur"""\U000023F3""",)
regex_strings += (ur"""\U000025AA""",)
regex_strings += (ur"""\U000025AB""",)
regex_strings += (ur"""\U000025B6""",)
regex_strings += (ur"""\U000025C0""",)
regex_strings += (ur"""\U000025FB""",)
regex_strings += (ur"""\U000025FC""",)
regex_strings += (ur"""\U000025FD""",)
regex_strings += (ur"""\U000025FE""",)
regex_strings += (ur"""\U00002600""",)
regex_strings += (ur"""\U00002601""",)
regex_strings += (ur"""\U0000260E""",)
regex_strings += (ur"""\U00002611""",)
regex_strings += (ur"""\U00002614""",)
regex_strings += (ur"""\U00002615""",)
regex_strings += (ur"""\U0000261D""",)
regex_strings += (ur"""\U0000263A""",)
regex_strings += (ur"""\U00002648""",)
regex_strings += (ur"""\U00002649""",)
regex_strings += (ur"""\U0000264A""",)
regex_strings += (ur"""\U0000264B""",)
regex_strings += (ur"""\U0000264C""",)
regex_strings += (ur"""\U0000264D""",)
regex_strings += (ur"""\U0000264E""",)
regex_strings += (ur"""\U0000264F""",)
regex_strings += (ur"""\U00002650""",)
regex_strings += (ur"""\U00002651""",)
regex_strings += (ur"""\U00002652""",)
regex_strings += (ur"""\U00002653""",)
regex_strings += (ur"""\U00002660""",)
regex_strings += (ur"""\U00002663""",)
regex_strings += (ur"""\U00002665""",)
regex_strings += (ur"""\U00002666""",)
regex_strings += (ur"""\U00002668""",)
regex_strings += (ur"""\U0000267B""",)
regex_strings += (ur"""\U0000267F""",)
regex_strings += (ur"""\U00002693""",)
regex_strings += (ur"""\U000026A0""",)
regex_strings += (ur"""\U000026A1""",)
regex_strings += (ur"""\U000026AA""",)
regex_strings += (ur"""\U000026AB""",)
regex_strings += (ur"""\U000026BD""",)
regex_strings += (ur"""\U000026BE""",)
regex_strings += (ur"""\U000026C4""",)
regex_strings += (ur"""\U000026C5""",)
regex_strings += (ur"""\U000026CE""",)
regex_strings += (ur"""\U000026D4""",)
regex_strings += (ur"""\U000026EA""",)
regex_strings += (ur"""\U000026F2""",)
regex_strings += (ur"""\U000026F3""",)
regex_strings += (ur"""\U000026F5""",)
regex_strings += (ur"""\U000026FA""",)
regex_strings += (ur"""\U000026FD""",)
regex_strings += (ur"""\U00002934""",)
regex_strings += (ur"""\U00002935""",)
regex_strings += (ur"""\U00002B05""",)
regex_strings += (ur"""\U00002B06""",)
regex_strings += (ur"""\U00002B07""",)
regex_strings += (ur"""\U00002B1B""",)
regex_strings += (ur"""\U00002B1C""",)
regex_strings += (ur"""\U00002B50""",)
regex_strings += (ur"""\U00002B55""",)
regex_strings += (ur"""\U00003030""",)
regex_strings += (ur"""\U0000303D""",)
regex_strings += (ur"""\U00003297""",)
regex_strings += (ur"""\U00003299""",)
regex_strings += (ur"""\U0001F004""",)
regex_strings += (ur"""\U0001F0CF""",)
regex_strings += (ur"""\U0001F300""",)
regex_strings += (ur"""\U0001F301""",)
regex_strings += (ur"""\U0001F302""",)
regex_strings += (ur"""\U0001F303""",)
regex_strings += (ur"""\U0001F304""",)
regex_strings += (ur"""\U0001F305""",)
regex_strings += (ur"""\U0001F306""",)
regex_strings += (ur"""\U0001F307""",)
regex_strings += (ur"""\U0001F308""",)
regex_strings += (ur"""\U0001F309""",)
regex_strings += (ur"""\U0001F30A""",)
regex_strings += (ur"""\U0001F30B""",)
regex_strings += (ur"""\U0001F30C""",)
regex_strings += (ur"""\U0001F30F""",)
regex_strings += (ur"""\U0001F311""",)
regex_strings += (ur"""\U0001F313""",)
regex_strings += (ur"""\U0001F314""",)
regex_strings += (ur"""\U0001F315""",)
regex_strings += (ur"""\U0001F319""",)
regex_strings += (ur"""\U0001F31B""",)
regex_strings += (ur"""\U0001F31F""",)
regex_strings += (ur"""\U0001F320""",)
regex_strings += (ur"""\U0001F330""",)
regex_strings += (ur"""\U0001F331""",)
regex_strings += (ur"""\U0001F334""",)
regex_strings += (ur"""\U0001F335""",)
regex_strings += (ur"""\U0001F337""",)
regex_strings += (ur"""\U0001F338""",)
regex_strings += (ur"""\U0001F339""",)
regex_strings += (ur"""\U0001F33A""",)
regex_strings += (ur"""\U0001F33B""",)
regex_strings += (ur"""\U0001F33C""",)
regex_strings += (ur"""\U0001F33D""",)
regex_strings += (ur"""\U0001F33E""",)
regex_strings += (ur"""\U0001F33F""",)
regex_strings += (ur"""\U0001F340""",)
regex_strings += (ur"""\U0001F341""",)
regex_strings += (ur"""\U0001F342""",)
regex_strings += (ur"""\U0001F343""",)
regex_strings += (ur"""\U0001F344""",)
regex_strings += (ur"""\U0001F345""",)
regex_strings += (ur"""\U0001F346""",)
regex_strings += (ur"""\U0001F347""",)
regex_strings += (ur"""\U0001F348""",)
regex_strings += (ur"""\U0001F349""",)
regex_strings += (ur"""\U0001F34A""",)
regex_strings += (ur"""\U0001F34C""",)
regex_strings += (ur"""\U0001F34D""",)
regex_strings += (ur"""\U0001F34E""",)
regex_strings += (ur"""\U0001F34F""",)
regex_strings += (ur"""\U0001F351""",)
regex_strings += (ur"""\U0001F352""",)
regex_strings += (ur"""\U0001F353""",)
regex_strings += (ur"""\U0001F354""",)
regex_strings += (ur"""\U0001F355""",)
regex_strings += (ur"""\U0001F356""",)
regex_strings += (ur"""\U0001F357""",)
regex_strings += (ur"""\U0001F358""",)
regex_strings += (ur"""\U0001F359""",)
regex_strings += (ur"""\U0001F35A""",)
regex_strings += (ur"""\U0001F35B""",)
regex_strings += (ur"""\U0001F35C""",)
regex_strings += (ur"""\U0001F35D""",)
regex_strings += (ur"""\U0001F35E""",)
regex_strings += (ur"""\U0001F35F""",)
regex_strings += (ur"""\U0001F360""",)
regex_strings += (ur"""\U0001F361""",)
regex_strings += (ur"""\U0001F362""",)
regex_strings += (ur"""\U0001F363""",)
regex_strings += (ur"""\U0001F364""",)
regex_strings += (ur"""\U0001F365""",)
regex_strings += (ur"""\U0001F366""",)
regex_strings += (ur"""\U0001F367""",)
regex_strings += (ur"""\U0001F368""",)
regex_strings += (ur"""\U0001F369""",)
regex_strings += (ur"""\U0001F36A""",)
regex_strings += (ur"""\U0001F36B""",)
regex_strings += (ur"""\U0001F36C""",)
regex_strings += (ur"""\U0001F36D""",)
regex_strings += (ur"""\U0001F36E""",)
regex_strings += (ur"""\U0001F36F""",)
regex_strings += (ur"""\U0001F370""",)
regex_strings += (ur"""\U0001F371""",)
regex_strings += (ur"""\U0001F372""",)
regex_strings += (ur"""\U0001F373""",)
regex_strings += (ur"""\U0001F374""",)
regex_strings += (ur"""\U0001F375""",)
regex_strings += (ur"""\U0001F376""",)
regex_strings += (ur"""\U0001F377""",)
regex_strings += (ur"""\U0001F378""",)
regex_strings += (ur"""\U0001F379""",)
regex_strings += (ur"""\U0001F37A""",)
regex_strings += (ur"""\U0001F37B""",)
regex_strings += (ur"""\U0001F380""",)
regex_strings += (ur"""\U0001F381""",)
regex_strings += (ur"""\U0001F382""",)
regex_strings += (ur"""\U0001F383""",)
regex_strings += (ur"""\U0001F384""",)
regex_strings += (ur"""\U0001F385""",)
regex_strings += (ur"""\U0001F386""",)
regex_strings += (ur"""\U0001F387""",)
regex_strings += (ur"""\U0001F388""",)
regex_strings += (ur"""\U0001F389""",)
regex_strings += (ur"""\U0001F38A""",)
regex_strings += (ur"""\U0001F38B""",)
regex_strings += (ur"""\U0001F38C""",)
regex_strings += (ur"""\U0001F38D""",)
regex_strings += (ur"""\U0001F38E""",)
regex_strings += (ur"""\U0001F38F""",)
regex_strings += (ur"""\U0001F390""",)
regex_strings += (ur"""\U0001F391""",)
regex_strings += (ur"""\U0001F392""",)
regex_strings += (ur"""\U0001F393""",)
regex_strings += (ur"""\U0001F3A0""",)
regex_strings += (ur"""\U0001F3A1""",)
regex_strings += (ur"""\U0001F3A2""",)
regex_strings += (ur"""\U0001F3A3""",)
regex_strings += (ur"""\U0001F3A4""",)
regex_strings += (ur"""\U0001F3A5""",)
regex_strings += (ur"""\U0001F3A6""",)
regex_strings += (ur"""\U0001F3A7""",)
regex_strings += (ur"""\U0001F3A8""",)
regex_strings += (ur"""\U0001F3A9""",)
regex_strings += (ur"""\U0001F3AA""",)
regex_strings += (ur"""\U0001F3AB""",)
regex_strings += (ur"""\U0001F3AC""",)
regex_strings += (ur"""\U0001F3AD""",)
regex_strings += (ur"""\U0001F3AE""",)
regex_strings += (ur"""\U0001F3AF""",)
regex_strings += (ur"""\U0001F3B0""",)
regex_strings += (ur"""\U0001F3B1""",)
regex_strings += (ur"""\U0001F3B2""",)
regex_strings += (ur"""\U0001F3B3""",)
regex_strings += (ur"""\U0001F3B4""",)
regex_strings += (ur"""\U0001F3B5""",)
regex_strings += (ur"""\U0001F3B6""",)
regex_strings += (ur"""\U0001F3B7""",)
regex_strings += (ur"""\U0001F3B8""",)
regex_strings += (ur"""\U0001F3B9""",)
regex_strings += (ur"""\U0001F3BA""",)
regex_strings += (ur"""\U0001F3BB""",)
regex_strings += (ur"""\U0001F3BC""",)
regex_strings += (ur"""\U0001F3BD""",)
regex_strings += (ur"""\U0001F3BE""",)
regex_strings += (ur"""\U0001F3BF""",)
regex_strings += (ur"""\U0001F3C0""",)
regex_strings += (ur"""\U0001F3C1""",)
regex_strings += (ur"""\U0001F3C2""",)
regex_strings += (ur"""\U0001F3C3""",)
regex_strings += (ur"""\U0001F3C4""",)
regex_strings += (ur"""\U0001F3C6""",)
regex_strings += (ur"""\U0001F3C8""",)
regex_strings += (ur"""\U0001F3CA""",)
regex_strings += (ur"""\U0001F3E0""",)
regex_strings += (ur"""\U0001F3E1""",)
regex_strings += (ur"""\U0001F3E2""",)
regex_strings += (ur"""\U0001F3E3""",)
regex_strings += (ur"""\U0001F3E5""",)
regex_strings += (ur"""\U0001F3E6""",)
regex_strings += (ur"""\U0001F3E7""",)
regex_strings += (ur"""\U0001F3E8""",)
regex_strings += (ur"""\U0001F3E9""",)
regex_strings += (ur"""\U0001F3EA""",)
regex_strings += (ur"""\U0001F3EB""",)
regex_strings += (ur"""\U0001F3EC""",)
regex_strings += (ur"""\U0001F3ED""",)
regex_strings += (ur"""\U0001F3EE""",)
regex_strings += (ur"""\U0001F3EF""",)
regex_strings += (ur"""\U0001F3F0""",)
regex_strings += (ur"""\U0001F40C""",)
regex_strings += (ur"""\U0001F40D""",)
regex_strings += (ur"""\U0001F40E""",)
regex_strings += (ur"""\U0001F411""",)
regex_strings += (ur"""\U0001F412""",)
regex_strings += (ur"""\U0001F414""",)
regex_strings += (ur"""\U0001F417""",)
regex_strings += (ur"""\U0001F418""",)
regex_strings += (ur"""\U0001F419""",)
regex_strings += (ur"""\U0001F41A""",)
regex_strings += (ur"""\U0001F41B""",)
regex_strings += (ur"""\U0001F41C""",)
regex_strings += (ur"""\U0001F41D""",)
regex_strings += (ur"""\U0001F41E""",)
regex_strings += (ur"""\U0001F41F""",)
regex_strings += (ur"""\U0001F420""",)
regex_strings += (ur"""\U0001F421""",)
regex_strings += (ur"""\U0001F422""",)
regex_strings += (ur"""\U0001F423""",)
regex_strings += (ur"""\U0001F424""",)
regex_strings += (ur"""\U0001F425""",)
regex_strings += (ur"""\U0001F426""",)
regex_strings += (ur"""\U0001F427""",)
regex_strings += (ur"""\U0001F428""",)
regex_strings += (ur"""\U0001F429""",)
regex_strings += (ur"""\U0001F42B""",)
regex_strings += (ur"""\U0001F42C""",)
regex_strings += (ur"""\U0001F42D""",)
regex_strings += (ur"""\U0001F42E""",)
regex_strings += (ur"""\U0001F42F""",)
regex_strings += (ur"""\U0001F430""",)
regex_strings += (ur"""\U0001F431""",)
regex_strings += (ur"""\U0001F432""",)
regex_strings += (ur"""\U0001F433""",)
regex_strings += (ur"""\U0001F434""",)
regex_strings += (ur"""\U0001F435""",)
regex_strings += (ur"""\U0001F436""",)
regex_strings += (ur"""\U0001F437""",)
regex_strings += (ur"""\U0001F438""",)
regex_strings += (ur"""\U0001F439""",)
regex_strings += (ur"""\U0001F43A""",)
regex_strings += (ur"""\U0001F43B""",)
regex_strings += (ur"""\U0001F43C""",)
regex_strings += (ur"""\U0001F43D""",)
regex_strings += (ur"""\U0001F43E""",)
regex_strings += (ur"""\U0001F440""",)
regex_strings += (ur"""\U0001F442""",)
regex_strings += (ur"""\U0001F443""",)
regex_strings += (ur"""\U0001F444""",)
regex_strings += (ur"""\U0001F445""",)
regex_strings += (ur"""\U0001F446""",)
regex_strings += (ur"""\U0001F447""",)
regex_strings += (ur"""\U0001F448""",)
regex_strings += (ur"""\U0001F449""",)
regex_strings += (ur"""\U0001F44A""",)
regex_strings += (ur"""\U0001F44B""",)
regex_strings += (ur"""\U0001F44C""",)
regex_strings += (ur"""\U0001F44D""",)
regex_strings += (ur"""\U0001F44E""",)
regex_strings += (ur"""\U0001F44F""",)
regex_strings += (ur"""\U0001F450""",)
regex_strings += (ur"""\U0001F451""",)
regex_strings += (ur"""\U0001F452""",)
regex_strings += (ur"""\U0001F453""",)
regex_strings += (ur"""\U0001F454""",)
regex_strings += (ur"""\U0001F455""",)
regex_strings += (ur"""\U0001F456""",)
regex_strings += (ur"""\U0001F457""",)
regex_strings += (ur"""\U0001F458""",)
regex_strings += (ur"""\U0001F459""",)
regex_strings += (ur"""\U0001F45A""",)
regex_strings += (ur"""\U0001F45B""",)
regex_strings += (ur"""\U0001F45C""",)
regex_strings += (ur"""\U0001F45D""",)
regex_strings += (ur"""\U0001F45E""",)
regex_strings += (ur"""\U0001F45F""",)
regex_strings += (ur"""\U0001F460""",)
regex_strings += (ur"""\U0001F461""",)
regex_strings += (ur"""\U0001F462""",)
regex_strings += (ur"""\U0001F463""",)
regex_strings += (ur"""\U0001F464""",)
regex_strings += (ur"""\U0001F466""",)
regex_strings += (ur"""\U0001F467""",)
regex_strings += (ur"""\U0001F468""",)
regex_strings += (ur"""\U0001F469""",)
regex_strings += (ur"""\U0001F46A""",)
regex_strings += (ur"""\U0001F46B""",)
regex_strings += (ur"""\U0001F46E""",)
regex_strings += (ur"""\U0001F46F""",)
regex_strings += (ur"""\U0001F470""",)
regex_strings += (ur"""\U0001F471""",)
regex_strings += (ur"""\U0001F472""",)
regex_strings += (ur"""\U0001F473""",)
regex_strings += (ur"""\U0001F474""",)
regex_strings += (ur"""\U0001F475""",)
regex_strings += (ur"""\U0001F476""",)
regex_strings += (ur"""\U0001F477""",)
regex_strings += (ur"""\U0001F478""",)
regex_strings += (ur"""\U0001F479""",)
regex_strings += (ur"""\U0001F47A""",)
regex_strings += (ur"""\U0001F47B""",)
regex_strings += (ur"""\U0001F47C""",)
regex_strings += (ur"""\U0001F47D""",)
regex_strings += (ur"""\U0001F47E""",)
regex_strings += (ur"""\U0001F47F""",)
regex_strings += (ur"""\U0001F480""",)
regex_strings += (ur"""\U0001F481""",)
regex_strings += (ur"""\U0001F482""",)
regex_strings += (ur"""\U0001F483""",)
regex_strings += (ur"""\U0001F484""",)
regex_strings += (ur"""\U0001F485""",)
regex_strings += (ur"""\U0001F486""",)
regex_strings += (ur"""\U0001F487""",)
regex_strings += (ur"""\U0001F488""",)
regex_strings += (ur"""\U0001F489""",)
regex_strings += (ur"""\U0001F48A""",)
regex_strings += (ur"""\U0001F48B""",)
regex_strings += (ur"""\U0001F48C""",)
regex_strings += (ur"""\U0001F48D""",)
regex_strings += (ur"""\U0001F48E""",)
regex_strings += (ur"""\U0001F48F""",)
regex_strings += (ur"""\U0001F490""",)
regex_strings += (ur"""\U0001F491""",)
regex_strings += (ur"""\U0001F492""",)
regex_strings += (ur"""\U0001F493""",)
regex_strings += (ur"""\U0001F494""",)
regex_strings += (ur"""\U0001F495""",)
regex_strings += (ur"""\U0001F496""",)
regex_strings += (ur"""\U0001F497""",)
regex_strings += (ur"""\U0001F498""",)
regex_strings += (ur"""\U0001F499""",)
regex_strings += (ur"""\U0001F49A""",)
regex_strings += (ur"""\U0001F49B""",)
regex_strings += (ur"""\U0001F49C""",)
regex_strings += (ur"""\U0001F49D""",)
regex_strings += (ur"""\U0001F49E""",)
regex_strings += (ur"""\U0001F49F""",)
regex_strings += (ur"""\U0001F4A0""",)
regex_strings += (ur"""\U0001F4A1""",)
regex_strings += (ur"""\U0001F4A2""",)
regex_strings += (ur"""\U0001F4A3""",)
regex_strings += (ur"""\U0001F4A4""",)
regex_strings += (ur"""\U0001F4A5""",)
regex_strings += (ur"""\U0001F4A6""",)
regex_strings += (ur"""\U0001F4A7""",)
regex_strings += (ur"""\U0001F4A8""",)
regex_strings += (ur"""\U0001F4A9""",)
regex_strings += (ur"""\U0001F4AA""",)
regex_strings += (ur"""\U0001F4AB""",)
regex_strings += (ur"""\U0001F4AC""",)
regex_strings += (ur"""\U0001F4AE""",)
regex_strings += (ur"""\U0001F4AF""",)
regex_strings += (ur"""\U0001F4B0""",)
regex_strings += (ur"""\U0001F4B1""",)
regex_strings += (ur"""\U0001F4B2""",)
regex_strings += (ur"""\U0001F4B3""",)
regex_strings += (ur"""\U0001F4B4""",)
regex_strings += (ur"""\U0001F4B5""",)
regex_strings += (ur"""\U0001F4B8""",)
regex_strings += (ur"""\U0001F4B9""",)
regex_strings += (ur"""\U0001F4BA""",)
regex_strings += (ur"""\U0001F4BB""",)
regex_strings += (ur"""\U0001F4BC""",)
regex_strings += (ur"""\U0001F4BD""",)
regex_strings += (ur"""\U0001F4BE""",)
regex_strings += (ur"""\U0001F4BF""",)
regex_strings += (ur"""\U0001F4C0""",)
regex_strings += (ur"""\U0001F4C1""",)
regex_strings += (ur"""\U0001F4C2""",)
regex_strings += (ur"""\U0001F4C3""",)
regex_strings += (ur"""\U0001F4C4""",)
regex_strings += (ur"""\U0001F4C5""",)
regex_strings += (ur"""\U0001F4C6""",)
regex_strings += (ur"""\U0001F4C7""",)
regex_strings += (ur"""\U0001F4C8""",)
regex_strings += (ur"""\U0001F4C9""",)
regex_strings += (ur"""\U0001F4CA""",)
regex_strings += (ur"""\U0001F4CB""",)
regex_strings += (ur"""\U0001F4CC""",)
regex_strings += (ur"""\U0001F4CD""",)
regex_strings += (ur"""\U0001F4CE""",)
regex_strings += (ur"""\U0001F4CF""",)
regex_strings += (ur"""\U0001F4D0""",)
regex_strings += (ur"""\U0001F4D1""",)
regex_strings += (ur"""\U0001F4D2""",)
regex_strings += (ur"""\U0001F4D3""",)
regex_strings += (ur"""\U0001F4D4""",)
regex_strings += (ur"""\U0001F4D5""",)
regex_strings += (ur"""\U0001F4D6""",)
regex_strings += (ur"""\U0001F4D7""",)
regex_strings += (ur"""\U0001F4D8""",)
regex_strings += (ur"""\U0001F4D9""",)
regex_strings += (ur"""\U0001F4DA""",)
regex_strings += (ur"""\U0001F4DB""",)
regex_strings += (ur"""\U0001F4DC""",)
regex_strings += (ur"""\U0001F4DD""",)
regex_strings += (ur"""\U0001F4DE""",)
regex_strings += (ur"""\U0001F4DF""",)
regex_strings += (ur"""\U0001F4E0""",)
regex_strings += (ur"""\U0001F4E1""",)
regex_strings += (ur"""\U0001F4E2""",)
regex_strings += (ur"""\U0001F4E3""",)
regex_strings += (ur"""\U0001F4E4""",)
regex_strings += (ur"""\U0001F4E5""",)
regex_strings += (ur"""\U0001F4E6""",)
regex_strings += (ur"""\U0001F4E7""",)
regex_strings += (ur"""\U0001F4E8""",)
regex_strings += (ur"""\U0001F4E9""",)
regex_strings += (ur"""\U0001F4EA""",)
regex_strings += (ur"""\U0001F4EB""",)
regex_strings += (ur"""\U0001F4EE""",)
regex_strings += (ur"""\U0001F4F0""",)
regex_strings += (ur"""\U0001F4F1""",)
regex_strings += (ur"""\U0001F4F2""",)
regex_strings += (ur"""\U0001F4F3""",)
regex_strings += (ur"""\U0001F4F4""",)
regex_strings += (ur"""\U0001F4F6""",)
regex_strings += (ur"""\U0001F4F7""",)
regex_strings += (ur"""\U0001F4F9""",)
regex_strings += (ur"""\U0001F4FA""",)
regex_strings += (ur"""\U0001F4FB""",)
regex_strings += (ur"""\U0001F4FC""",)
regex_strings += (ur"""\U0001F503""",)
regex_strings += (ur"""\U0001F50A""",)
regex_strings += (ur"""\U0001F50B""",)
regex_strings += (ur"""\U0001F50C""",)
regex_strings += (ur"""\U0001F50D""",)
regex_strings += (ur"""\U0001F50E""",)
regex_strings += (ur"""\U0001F50F""",)
regex_strings += (ur"""\U0001F510""",)
regex_strings += (ur"""\U0001F511""",)
regex_strings += (ur"""\U0001F512""",)
regex_strings += (ur"""\U0001F513""",)
regex_strings += (ur"""\U0001F514""",)
regex_strings += (ur"""\U0001F516""",)
regex_strings += (ur"""\U0001F517""",)
regex_strings += (ur"""\U0001F518""",)
regex_strings += (ur"""\U0001F519""",)
regex_strings += (ur"""\U0001F51A""",)
regex_strings += (ur"""\U0001F51B""",)
regex_strings += (ur"""\U0001F51C""",)
regex_strings += (ur"""\U0001F51D""",)
regex_strings += (ur"""\U0001F51E""",)
regex_strings += (ur"""\U0001F51F""",)
regex_strings += (ur"""\U0001F520""",)
regex_strings += (ur"""\U0001F521""",)
regex_strings += (ur"""\U0001F522""",)
regex_strings += (ur"""\U0001F523""",)
regex_strings += (ur"""\U0001F524""",)
regex_strings += (ur"""\U0001F525""",)
regex_strings += (ur"""\U0001F526""",)
regex_strings += (ur"""\U0001F527""",)
regex_strings += (ur"""\U0001F528""",)
regex_strings += (ur"""\U0001F529""",)
regex_strings += (ur"""\U0001F52A""",)
regex_strings += (ur"""\U0001F52B""",)
regex_strings += (ur"""\U0001F52E""",)
regex_strings += (ur"""\U0001F52F""",)
regex_strings += (ur"""\U0001F530""",)
regex_strings += (ur"""\U0001F531""",)
regex_strings += (ur"""\U0001F532""",)
regex_strings += (ur"""\U0001F533""",)
regex_strings += (ur"""\U0001F534""",)
regex_strings += (ur"""\U0001F535""",)
regex_strings += (ur"""\U0001F536""",)
regex_strings += (ur"""\U0001F537""",)
regex_strings += (ur"""\U0001F538""",)
regex_strings += (ur"""\U0001F539""",)
regex_strings += (ur"""\U0001F53A""",)
regex_strings += (ur"""\U0001F53B""",)
regex_strings += (ur"""\U0001F53C""",)
regex_strings += (ur"""\U0001F53D""",)
regex_strings += (ur"""\U0001F550""",)
regex_strings += (ur"""\U0001F551""",)
regex_strings += (ur"""\U0001F552""",)
regex_strings += (ur"""\U0001F553""",)
regex_strings += (ur"""\U0001F554""",)
regex_strings += (ur"""\U0001F555""",)
regex_strings += (ur"""\U0001F556""",)
regex_strings += (ur"""\U0001F557""",)
regex_strings += (ur"""\U0001F558""",)
regex_strings += (ur"""\U0001F559""",)
regex_strings += (ur"""\U0001F55A""",)
regex_strings += (ur"""\U0001F55B""",)
regex_strings += (ur"""\U0001F5FB""",)
regex_strings += (ur"""\U0001F5FC""",)
regex_strings += (ur"""\U0001F5FD""",)
regex_strings += (ur"""\U0001F5FE""",)
regex_strings += (ur"""\U0001F5FF""",)
regex_strings += (ur"""\U0001F600""",)
regex_strings += (ur"""\U0001F607""",)
regex_strings += (ur"""\U0001F608""",)
regex_strings += (ur"""\U0001F60E""",)
regex_strings += (ur"""\U0001F610""",)
regex_strings += (ur"""\U0001F611""",)
regex_strings += (ur"""\U0001F615""",)
regex_strings += (ur"""\U0001F617""",)
regex_strings += (ur"""\U0001F619""",)
regex_strings += (ur"""\U0001F61B""",)
regex_strings += (ur"""\U0001F61F""",)
regex_strings += (ur"""\U0001F626""",)
regex_strings += (ur"""\U0001F627""",)
regex_strings += (ur"""\U0001F62C""",)
regex_strings += (ur"""\U0001F62E""",)
regex_strings += (ur"""\U0001F62F""",)
regex_strings += (ur"""\U0001F634""",)
regex_strings += (ur"""\U0001F636""",)
regex_strings += (ur"""\U0001F681""",)
regex_strings += (ur"""\U0001F682""",)
regex_strings += (ur"""\U0001F686""",)
regex_strings += (ur"""\U0001F688""",)
regex_strings += (ur"""\U0001F68A""",)
regex_strings += (ur"""\U0001F68D""",)
regex_strings += (ur"""\U0001F68E""",)
regex_strings += (ur"""\U0001F690""",)
regex_strings += (ur"""\U0001F694""",)
regex_strings += (ur"""\U0001F696""",)
regex_strings += (ur"""\U0001F698""",)
regex_strings += (ur"""\U0001F69B""",)
regex_strings += (ur"""\U0001F69C""",)
regex_strings += (ur"""\U0001F69D""",)
regex_strings += (ur"""\U0001F69E""",)
regex_strings += (ur"""\U0001F69F""",)
regex_strings += (ur"""\U0001F6A0""",)
regex_strings += (ur"""\U0001F6A1""",)
regex_strings += (ur"""\U0001F6A3""",)
regex_strings += (ur"""\U0001F6A6""",)
regex_strings += (ur"""\U0001F6AE""",)
regex_strings += (ur"""\U0001F6AF""",)
regex_strings += (ur"""\U0001F6B0""",)
regex_strings += (ur"""\U0001F6B1""",)
regex_strings += (ur"""\U0001F6B3""",)
regex_strings += (ur"""\U0001F6B4""",)
regex_strings += (ur"""\U0001F6B5""",)
regex_strings += (ur"""\U0001F6B7""",)
regex_strings += (ur"""\U0001F6B8""",)
regex_strings += (ur"""\U0001F6BF""",)
regex_strings += (ur"""\U0001F6C1""",)
regex_strings += (ur"""\U0001F6C2""",)
regex_strings += (ur"""\U0001F6C3""",)
regex_strings += (ur"""\U0001F6C4""",)
regex_strings += (ur"""\U0001F6C5""",)
regex_strings += (ur"""\U0001F30D""",)
regex_strings += (ur"""\U0001F30E""",)
regex_strings += (ur"""\U0001F310""",)
regex_strings += (ur"""\U0001F312""",)
regex_strings += (ur"""\U0001F316""",)
regex_strings += (ur"""\U0001F317""",)
regex_strings += (ur"""\U0001F318""",)
regex_strings += (ur"""\U0001F31A""",)
regex_strings += (ur"""\U0001F31C""",)
regex_strings += (ur"""\U0001F31D""",)
regex_strings += (ur"""\U0001F31E""",)
regex_strings += (ur"""\U0001F332""",)
regex_strings += (ur"""\U0001F333""",)
regex_strings += (ur"""\U0001F34B""",)
regex_strings += (ur"""\U0001F350""",)
regex_strings += (ur"""\U0001F37C""",)
regex_strings += (ur"""\U0001F3C7""",)
regex_strings += (ur"""\U0001F3C9""",)
regex_strings += (ur"""\U0001F3E4""",)
regex_strings += (ur"""\U0001F400""",)
regex_strings += (ur"""\U0001F401""",)
regex_strings += (ur"""\U0001F402""",)
regex_strings += (ur"""\U0001F403""",)
regex_strings += (ur"""\U0001F404""",)
regex_strings += (ur"""\U0001F405""",)
regex_strings += (ur"""\U0001F406""",)
regex_strings += (ur"""\U0001F407""",)
regex_strings += (ur"""\U0001F408""",)
regex_strings += (ur"""\U0001F409""",)
regex_strings += (ur"""\U0001F40A""",)
regex_strings += (ur"""\U0001F40B""",)
regex_strings += (ur"""\U0001F40F""",)
regex_strings += (ur"""\U0001F410""",)
regex_strings += (ur"""\U0001F413""",)
regex_strings += (ur"""\U0001F415""",)
regex_strings += (ur"""\U0001F416""",)
regex_strings += (ur"""\U0001F42A""",)
regex_strings += (ur"""\U0001F465""",)
regex_strings += (ur"""\U0001F46C""",)
regex_strings += (ur"""\U0001F46D""",)
regex_strings += (ur"""\U0001F4AD""",)
regex_strings += (ur"""\U0001F4B6""",)
regex_strings += (ur"""\U0001F4B7""",)
regex_strings += (ur"""\U0001F4EC""",)
regex_strings += (ur"""\U0001F4ED""",)
regex_strings += (ur"""\U0001F4EF""",)
regex_strings += (ur"""\U0001F4F5""",)
regex_strings += (ur"""\U0001F500""",)
regex_strings += (ur"""\U0001F501""",)
regex_strings += (ur"""\U0001F502""",)
regex_strings += (ur"""\U0001F504""",)
regex_strings += (ur"""\U0001F505""",)
regex_strings += (ur"""\U0001F506""",)
regex_strings += (ur"""\U0001F507""",)
regex_strings += (ur"""\U0001F509""",)
regex_strings += (ur"""\U0001F515""",)
regex_strings += (ur"""\U0001F52C""",)
regex_strings += (ur"""\U0001F52D""",)
regex_strings += (ur"""\U0001F55C""",)
regex_strings += (ur"""\U0001F55D""",)
regex_strings += (ur"""\U0001F55E""",)
regex_strings += (ur"""\U0001F55F""",)
regex_strings += (ur"""\U0001F560""",)
regex_strings += (ur"""\U0001F561""",)
regex_strings += (ur"""\U0001F562""",)
regex_strings += (ur"""\U0001F563""",)
regex_strings += (ur"""\U0001F564""",)
regex_strings += (ur"""\U0001F565""",)
regex_strings += (ur"""\U0001F566""",)
regex_strings += (ur"""\U0001F567""",)
regex_strings += (ur"""(?:@[\w_]+)""",) # twitter user name.
regex_strings += (ur"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",) # twitter hashtags.
regex_strings += (ur"""<[^>]+>""",) # html tags. 
regex_strings += (ur"""http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+""",) # urls.
regex_strings += (ur"""(?:(?:\+?[01][\-\s.]*)?(?:[\(]?\d{3}[\-\s.\)]*)?\d{3}[\-\s.]*\d{4})""",) # phone numbers
regex_strings += (ur"""(?:[a-z][a-z'\-_]+[a-z])""",) # Words with apostrophes or dashes.
regex_strings += (ur"""(?:[+\-]?\d+[,/.:-]\d+[+\-]?)""",)  # Numbers, including fractions, decimals.
regex_strings += (ur"""(?:[\w_]+)""",) # Words without apostrophes or dashes.
regex_strings += (ur"""(?:\.(?:\s*\.){1,})""",) # Ellipsis dots. 
regex_strings += (ur"""(?:\S)""",) # Everything else that isn't whitespace.

# This is the core tokenizing regex:
    
word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)

# The emoticon string gets its own regex so that we can preserve case for them as needed:
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)

# These are for regularizing HTML entities to Unicode:
html_entity_digit_re = re.compile(r"&#\d+;")
html_entity_alpha_re = re.compile(r"&\w+;")
amp = "&amp;"

class MyTokenizer:
    def __init__(self, preserve_case=False):
        self.preserve_case = preserve_case
        
    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; concatenating this list returns the original string if preserve_case=False
        """        
        # Try to ensure unicode:
        try:
            s = unicode(s)
        except UnicodeDecodeError:
            s = str(s).encode('string_escape')
            s = unicode(s)
            
        # Fix HTML character entitites:
        s = self.__html2unicode(s)
        
        # Tokenize:
        words = word_re.findall(s)
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:            
            words = map((lambda x : x if emoticon_re.search(x) else x.lower()), words)
        return words

    def __html2unicode(self, s):
        """
        Internal metod that seeks to replace all the HTML entities in
        s with their corresponding unicode characters.
        """
        # First the digits:
        ents = set(html_entity_digit_re.findall(s))
        if len(ents) > 0:
            for ent in ents:
                entnum = ent[2:-1]
                try:
                    entnum = int(entnum)
                    s = s.replace(ent, unichr(entnum))    
                except:
                    pass
        # Now the alpha versions:
        ents = set(html_entity_alpha_re.findall(s))
        ents = filter((lambda x : x != amp), ents)
        for ent in ents:
            entname = ent[1:-1]
            try:            
                s = s.replace(ent, unichr(htmlentitydefs.name2codepoint[entname]))
            except:
                pass                    
            s = s.replace(amp, " and ")
        return s

if __name__ == '__main__':
    tok = Tokenizer(preserve_case=True)
    s = "I'm a example sentence."
    tokenized = tok.tokenize(s)
    print tokenized