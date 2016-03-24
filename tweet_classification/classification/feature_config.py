"""
@copyright: Jennifer Ling
Adapted from Konstantin Buschmeier (code for amazon review sentiment prediction)

Regular Expressions are used to encode emoticons and punctuation marks as features.
Unicodes are adapted from http://apps.timwhitlock.info/emoji/tables/unicode.
"""
#--------------------------------- FEATURE CONFIG --------------------------------------------
REGEX_FEATURE_CONFIG_SPECIAL = {
    u"Emotion Tongue": (u":-P", 
                        ur"""
                        \U0001F61C|                    # FACE WITH STUCK-OUT TONGUE AND WINKING EYE
                        \U0001F61D|                    # FACE WITH STUCK-OUT TONGUE AND TIGHTLY-CLOSED EYES
                        \U0001F61B|                    # FACE WITH STUCK-OUT TONGUE
                        [:=][-]?[pqP]|            # :-P :P
                        [pqP][-]?[:=]                # q-: P-:
                        """
    ), 
    u"Emoticon Unhappy": (u":-/", 
                                ur""" 
                                \U0001F60F|                    # SMIRKING FACE
                                \U0001F612|                    # UNAMUSED FACE
                                \U0001F613|                    # FACE WITH COLD SWEAT
                                \U0001F614|                    # PENSIVE FACE
                                \U0001F616|                    # CONFOUNDED FACE
                                \U0001F623|                    # PERSEVERING FACE
                                \U0001F610|                    # NEUTRAL FACE
                                \U0001F611|                    # EXPRESSIONLESS FACE
                                \U0001F615|                    # CONFUSED FACE
                                \U0001F626|                    # FROWNING FACE WITH OPEN MOUTH
                                \U0001F627|                    # ANGUISHED FACE
                                \U0001F62C|                    # GRIMACING FACE              
                                [:=][-o]?[\/\\|]|           # :-/ :-\ :-| :/
                                [\/\\|][-o]?[:=]|           # \-: \:
                                -_+-                        # -_- -___-
                                """
    ),
    u"Punctuation Marks": (u"!?..", 
                       ur"""
                       \U00002753|                    # BLACK QUESTION MARK ORNAMENT
                       \U00002754                     # WHITE QUESTION MARK ORNAMENT
                       \?{2,}|
                       \U00002755|                    # WHITE EXCLAMATION MARK ORNAMENT
                       \U00002757|                    # HEAVY EXCLAMATION MARK SYMBOL
                       \U0000203C                     # DOUBLE EXCLAMATION MARK
                       \!{2,}|
                        \U00002049| 
                        \?| 
                        \?\!|                  # EXCLAMATION QUESTION MARK
                        \!?\?+\!+\?*|              # ?!
                        \??\!+\?+\!*|             # !?
                        \.{2,}|                         # .. ...
                        \.(\ \.){2,}
                       """
    ),                                   # Unicode interrobang: U+203D
    u"Punctuation": (u"?!.", 
                    r"""
                    \?| 
                    \?\!|                  # EXCLAMATION QUESTION MARK
                    \!?\?+\!+\?*|              # ?!
                    \??\!+\?+\!*|
                    \?{2,}|
                    \!{2,}|
                    \.{2,}|                         # .. ...
                    \.(\ \.){2,}                                # . . .
                    """
    ),                                          # Unicode Ellipsis: U+2026
    # ---- Markup----
#     u"Hashtag": (u"#", r"""\#(irony|ironic|sarcasm|sarcastic)"""),  # #irony
#     u"Pseudo-Tag": (u"Tag", 
#                     r"""([<\[][\/\\]
#                     (irony|ironic|sarcasm|sarcastic)        # </irony>
#                     [>\]])|                                 #
#                     ((?<!(\w|[<\[]))[\/\\]                  #
#                     (irony|ironic|sarcasm|sarcastic)        # /irony
#                     (?![>\]]))
#                     """
#     ),
   
    # ---- Acronyms, onomatopoeia ----
    u"Acroym for Laughter": (u"lol", 
                    r"""(?<!\w)                             # Boundary
                    (l(([oua]|aw)l)+([sz]?|wut)|            # lol, lawl, luls
                    rot?fl(mf?ao)?)|                        # rofl, roflmao
                    lmf?ao                                  # lmao, lmfao
                    (?!\w)                                  # Boundary
                    """
    ),                                    
    u"Acronym for Grin": (u"*g*", 
                        r"""\*([Gg]{1,2}|                   # *g* *gg*
                        grin)\*                             # *grin*
                        """
    ),
    u"Onomatopoeia for Laughter": (u"haha", 
                        r"""(?<!\w)                         # Boundary
                        (mu|ba)?                            # mu- ba-
                        (ha|h(e|3)|hi){2,}                  # haha, hehe, hihi
                        (?!\w)                              # Boundary
                        """
    ),
}

REGEX_FEATURE_CONFIG_GROUPS = {
                  
    # ---- Emoticons ----
    u"Emoticon Happy": (u":-)", 
                        ur"""
                        \U0001F60A|                    # SMILING FACE WITH SMILING EYES
                        \U0001F60B|                    # FACE SAVOURING DELICIOUS FOOD
                        \U0001F60C|                    # RELIEVED FACE
                        \U0001F60D|                    # SMILING FACE WITH HEART-SHAPED EYES
                        \U0001F60E|                    # SMILING FACE WITH SUNGLASSES
                        [:=][o-]?[)}>\]]|
                        [({<\[][o-]?[:=]|
                        \^_*\^|
                        \^[-oO]?\^"""                  
    ), 
    u"Emoticon Laughing": (u":-D", 
                            ur"""
                            \U0001F61C|
                            \U0001F601|                    # GRINNING FACE WITH SMILING EYES
                            \U0001F602|                    # FACE WITH TEARS OF JOY
                            \U0001F603|                    # SMILING FACE WITH OPEN MOUTH
                            \U0001F604|                    # SMILING FACE WITH OPEN MOUTH AND SMILING EYES
                            \U0001F605|                    # SMILING FACE WITH OPEN MOUTH AND COLD SWEAT
                            \U0001F606|                    # SMILING FACE WITH OPEN MOUTH AND TIGHTLY-CLOSED EYES
                            \U0001F600|
                            [:=][-]?[D]|
                            x[D]
                            """
    ),   # :-D xD
    u"Emoticon Angel": (u"0:)", 
                        ur"""
                        \U0001F607|                    # SMILING FACE WITH HALO
                        [0Oo][:=][-]?[D\)\}]
                        """
    ),
    u"Emoticon Winking": (u";-)", 
                        ur"""
                        \U0001F609|                    # WINKING FACE
                        [;\*][-o]?[\)\}>\]]|              # ;-) ;o) ;)
                        [\(\{<\[][-o]?[;\*]                   # (-; (
                        """
    ), 
    u"Emoticon Tongue": (u":-P", 
                        ur"""
                        \U0001F61C|                    # FACE WITH STUCK-OUT TONGUE AND WINKING EYE
                        \U0001F61D|                    # FACE WITH STUCK-OUT TONGUE AND TIGHTLY-CLOSED EYES
                        \U0001F61B|                    # FACE WITH STUCK-OUT TONGUE
                        [:=][-]?[pqP]|            # :-P :P
                        [pqP][-]?[:=]                # q-: P-:
                        """
    ),  
    u"Emoticon Surprise": (u":-O", 
                            ur"""
                            \U0001F62E|                    # FACE WITH OPEN MOUTH
                            [:=]-?[oO0]|                   # :-O
                            [oO0]-?[:=]|                    # O-:
                            [oO]_*[oO]|
                            [oO]\.[oO]               # Oo O____o O.o
                            """
    ), 
    u"Emoticon Dissatisfied": (u":-/", 
                                ur""" 
                                \U0001F60F|                    # SMIRKING FACE
                                \U0001F612|                    # UNAMUSED FACE
                                \U0001F613|                    # FACE WITH COLD SWEAT
                                \U0001F614|                    # PENSIVE FACE
                                \U0001F616|                    # CONFOUNDED FACE
                                \U0001F623|                    # PERSEVERING FACE
                                \U0001F610|                    # NEUTRAL FACE
                                \U0001F611|                    # EXPRESSIONLESS FACE
                                \U0001F615|                    # CONFUSED FACE
                                \U0001F626|                    # FROWNING FACE WITH OPEN MOUTH
                                \U0001F627|                    # ANGUISHED FACE
                                \U0001F62C|                    # GRIMACING FACE              
                                [:=][-o]?[\/\\|]|           # :-/ :-\ :-| :/
                                [\/\\|][-o]?[:=]|           # \-: \:
                                -_+-                        # -_- -___-
                                """
    ), 
    u"Emoticon Sad": (u":-(", 
                      ur"""
                      \U0001F61E|                    # DISAPPOINTED FACE
                      \U0001F494|                    # BROKEN HEART
                      [:=][o-]?[\(\{<\[]|               # :-( :(
                      [\)\}>\[][o-]?[:=]                    # )-: ): )o: 
                      """
    ), 
    u"Emoticon Crying": (u";-(", 
                        ur"""
                        \U0001F622|                    # CRYING FACE
                        \U0001F62D|                    # LOUDLY CRYING FACE
                        [:=]'[o-]?[\(\{<\[]|
                        ;'?[o-]?[\(\{<\[]|    # ;-( :'(
                        [\)\}>\[][o-]?'[:=]|
                        [\)\}>\[][o-]?'?;          # )-; )-';
                        """
    ),
    u"Emoticon Angry/Evil": (u"anger", 
                        ur"""
                        \U0001F620|                    # ANGRY FACE
                        \U0001F621|                    # POUTING FACE
                        \U0001F624|                    # FACE WITH LOOK OF TRIUMPH|
                        \U0001F4A2|                    # ANGER SYMBOL
                        \U0001F4A3|                    # BOMB
                        \U0001F4A5|                    # COLLISION SYMBOL
                        \U0001F608                     # SMILING FACE WITH HORNS
                        """
    ), 
    u"Emoticon Fearful/Worried": (u"O_O", 
                        ur"""
                        \U0001F625|                    # DISAPPOINTED BUT RELIEVED FACE
                        \U0001F628|                    # FEARFUL FACE
                        \U0001F629|                    # WEARY FACE
                        \U0001F62A|                    # SLEEPY FACE
                        \U0001F62B|                    # TIRED FACE
                        \U0001F630|                    # FACE WITH OPEN MOUTH AND COLD SWEAT
                        \U0001F631|                    # FACE SCREAMING IN FEAR
                        \U0001F632|                    # ASTONISHED FACE
                        \U0001F633|                    # FLUSHED FACE
                        \U0001F635|                    # DIZZY FACE
                        \U0001F61F|                    # WORRIED FACE
                        \U0001F636                     # FACE WITHOUT MOUTH
                        """
    ), 
    u"Emoticon Cats": (u"miau", 
                        ur"""
                        \U0001F638|                    # GRINNING CAT FACE WITH SMILING EYES
                        \U0001F639|                    # CAT FACE WITH TEARS OF JOY
                        \U0001F63A|                    # SMILING CAT FACE WITH OPEN MOUTH
                        \U0001F63B|                    # SMILING CAT FACE WITH HEART-SHAPED EYES
                        \U0001F63C|                    # CAT FACE WITH WRY SMILE
                        \U0001F63D|                    # KISSING CAT FACE WITH CLOSED EYES
                        \U0001F63E|                    # POUTING CAT FACE
                        \U0001F63F|                    # CRYING CAT FACE
                        \U0001F640                     # WEARY CAT FACE
                        """
    ),
    u"Emoticon Sleeping": (u"zZZZZ", 
                        ur"""
                        \U0001F4A4|                    # SLEEPING SYMBOL
                        \U0001F634                    # SLEEPING FACE
                        """
    ),
    u"Emoticon Kiss": (u":*", 
                        ur"""
                        \U0001F618|                    # FACE THROWING A KISS
                        \U0001F61A|                    # KISSING FACE WITH CLOSED EYES
                        \U0001F617|                    # KISSING FACE
                        \U0001F619|                    # KISSING FACE WITH SMILING EYES
                        \U0001F48B|                    # KISS MARK
                        [:=][-o]?\*+|
                        \*+[-o]?[:=] 
                        """
                
    ),
    u"Emoticon Hearts": (u"<3", 
                        ur"""
                        \U00002764|                    # HEAVY BLACK HEART
                        \U0001F493|                    # BEATING HEART
                        \U0001F495|                    # TWO HEARTS
                        \U0001F496|                    # SPARKLING HEART
                        \U0001F497|                    # GROWING HEART
                        \U0001F498|                    # HEART WITH ARROW
                        \U0001F499|                    # BLUE HEART
                        \U0001F49A|                    # GREEN HEART
                        \U0001F49B|                    # YELLOW HEART
                        \U0001F49C|                    # PURPLE HEART
                        \U0001F49D|                    # HEART WITH RIBBON
                        \U0001F49E|                    # REVOLVING HEARTS
                        \U0001F49F|                    # HEART DECORATION
                        \<3                        
                        """
    ),
    u"Emoticon Monkeys": (u"bananaaaaa", 
                        ur"""
                        \U0001F648|                    # SEE-NO-EVIL MONKEY
                        \U0001F649|                    # HEAR-NO-EVIL MONKEY
                        \U0001F64A                    # SPEAK-NO-EVIL MONKEY
                        """
    ),                                                                                                                                            
    u"Positive": (u"pos", 
                        ur"""
                        \U00002714|                    # HEAVY CHECK MARK
                        \U00002795|                    # HEAVY PLUS SIGN
                        \U0001F646|                    # FACE WITH OK GESTURE
                        \U0001F64C|                    # PERSON RAISING BOTH HANDS IN CELEBRATION
                        \U0000270A|                    # RAISED FIST
                        \U0000270B|                    # RAISED HAND
                        \U0000270C                     # VICTORY HAND
                        """
    ),
    u"Negative": (u"neg", 
                        ur"""
                        \U0001F645|                    # FACE WITH NO GOOD GESTURE
                        \U00002716|                    # HEAVY MULTIPLICATION X
                        \U0000274C|                    # CROSS MARK
                        \U0000274E|                    # NEGATIVE SQUARED CROSS MARK
                        \U00002796|                    # HEAVY MINUS SIGN
                        \U0001F6A7|                    # CONSTRUCTION SIGN
                        \U0001F6A8|                    # POLICE CARS REVOLVING LIGHT
                        \U0001F637|                    # FACE WITH MEDICAL MASK
                        \U0001F647|                    # PERSON BOWING DEEPLY
                        \U0001F64D|                    # PERSON FROWNING
                        \U0001F64E|                    # PERSON WITH POUTING FACE
                        \U0001F64F                     # PERSON WITH FOLDED HANDS
                        """
    ),                              
    # ---- Punctuation----
    # u"AllPunctuation": (u"", r"""((\.{2,}|[?!]{2,})1*)"""),
    u"Question Mark": (u"??", 
                       ur"""
                       \U00002753|                    # BLACK QUESTION MARK ORNAMENT
                       \U00002754                     # WHITE QUESTION MARK ORNAMENT
                       \?{2,}
                       """
    ),                 # ??
    u"Exclamation Mark": (u"!!", 
                          ur"""
                          \U00002755|                    # WHITE EXCLAMATION MARK ORNAMENT
                          \U00002757|                    # HEAVY EXCLAMATION MARK SYMBOL
                          \U0000203C                     # DOUBLE EXCLAMATION MARK
                          \!{2,}
                          """
    ),              # !!
    u"Question and Exclamation Mark": (u"?!", 
                                       ur"""
                                       \U00002049| 
                                       \!\?| 
                                       \?\!|                  # EXCLAMATION QUESTION MARK
                                       \!?\?+\!+\?*|              # ?!
                                       \??\!+\?+\!*               # !?
                                       """
    ),                                          # Unicode interrobang: U+203D
    u"Ellipsis": (u"...", r"""\.{2,}|                         # .. ...
                \.(\ \.){2,}                                # . . .
                """
    ),                                          # Unicode Ellipsis: U+2026
    # ---- Markup----
#     u"Hashtag": (u"#", r"""\#(irony|ironic|sarcasm|sarcastic)"""),  # #irony
#     u"Pseudo-Tag": (u"Tag", 
#                     r"""([<\[][\/\\]
#                     (irony|ironic|sarcasm|sarcastic)        # </irony>
#                     [>\]])|                                 #
#                     ((?<!(\w|[<\[]))[\/\\]                  #
#                     (irony|ironic|sarcasm|sarcastic)        # /irony
#                     (?![>\]]))
#                     """
#     ),
   
    # ---- Acronyms, onomatopoeia ----
    u"Acroym for Laughter": (u"lol", 
                    r"""(?<!\w)                             # Boundary
                    (l(([oua]|aw)l)+([sz]?|wut)|            # lol, lawl, luls
                    rot?fl(mf?ao)?)|                        # rofl, roflmao
                    lmf?ao                                  # lmao, lmfao
                    (?!\w)                                  # Boundary
                    """
    ),                                    
    u"Acronym for Grin": (u"*g*", 
                        r"""\*([Gg]{1,2}|                   # *g* *gg*
                        grin)\*                             # *grin*
                        """
    ),
    u"Onomatopoeia for Laughter": (u"haha", 
                        r"""(?<!\w)                         # Boundary
                        (mu|ba)?                            # mu- ba-
                        (ha|h(e|3)|hi){2,}                  # haha, hehe, hihi
                        (?!\w)                              # Boundary
                        """
    ),
    u"Interjection": (u"ITJ", 
                        r"""(?<!\w)((a+h+a?)|               # ah, aha
                        (e+h+)|                             # eh
                        (u+g?h+)|                           # ugh
                        (huh)|                              # huh
                        ([uo]h( |-)h?[uo]h)|                # uh huh, 
                        (m*hm+)                             # hmm, mhm
                        |(h(u|r)?mp(h|f))|                  # hmpf
                        (ar+gh+)|                           # argh
                        (wow+))(?!\w)                       # wow
                        """
    ),
#     u"Symbols": (u"symb", 
#                 ur"""
#                 \U0001F192|                    # SQUARED COOL
#                 \U0001F193|                    # SQUARED FREE
#                 \U0001F195|                    # SQUARED NEW
#                 \U0001F197|                    # SQUARED OK
#                 \U0001F198|                    # SQUARED SOS
#                 U0001F48C|                    # LOVE LETTER
#                 \U0001F48D|                    # RING
#                 \U0001F48E|                    # GEM STONE
#                 \U0001F48F|                    # KISS
#                 \U0001F490|                    # BOUQUET
#                 \U0001F491|                    # COUPLE WITH HEART
#                 \U0001F492|                    # WEDDING
#                 \U0001F64B|                    # HAPPY PERSON RAISING ONE HAND
#                 \U00002702|                    # BLACK SCISSORS
#                 \U00002705|                    # WHITE HEAVY CHECK MARK
#                 \U00002708|                    # AIRPLANE
#                 \U00002709|                    # ENVELOPE
#                 \U0000270F|                    # PENCIL
#                 \U00002712|                    # BLACK NIB
#                 \U00002728|                    # SPARKLES
#                 \U00002733|                    # EIGHT SPOKED ASTERISK
#                 \U00002734|                    # EIGHT POINTED BLACK STAR
#                 \U00002744|                    # SNOWFLAKE
#                 \U00002747|                    # SPARKLE
#                 \U00002797|                    # HEAVY DIVISION SIGN
#                 \U000027A1|                    # BLACK RIGHTWARDS ARROW
#                 \U000027B0|                    # CURLY LOOP
#                 \U0001F680|                    # ROCKET
#                 \U0001F683|                    # RAILWAY CAR
#                 \U0001F684|                    # HIGH-SPEED TRAIN
#                 \U0001F685|                    # HIGH-SPEED TRAIN WITH BULLET NOSE
#                 \U0001F687|                    # METRO
#                 \U0001F689|                    # STATION
#                 \U0001F68C|                    # BUS
#                 \U0001F68F|                    # BUS STOP
#                 \U0001F691|                    # AMBULANCE
#                 \U0001F692|                    # FIRE ENGINE
#                 \U0001F693|                    # POLICE CAR
#                 \U0001F695|                    # TAXI
#                 \U0001F697|                    # AUTOMOBILE
#                 \U0001F699|                    # RECREATIONAL VEHICLE
#                 \U0001F69A|                    # DELIVERY TRUCK
#                 \U0001F6A2|                    # SHIP
#                 \U0001F6A4|                    # SPEEDBOAT
#                 \U0001F6A5|                    # HORIZONTAL TRAFFIC LIGHT
#                 \U0001F6A9|                    # TRIANGULAR FLAG ON POST
#                 \U0001F6AA|                    # DOOR
#                 \U0001F6AB|                    # NO ENTRY SIGN
#                 \U0001F6AC|                    # SMOKING SYMBOL
#                 \U0001F6AD|                    # NO SMOKING SYMBOL
#                 \U0001F6B2|                    # BICYCLE
#                 \U0001F6B6|                    # PEDESTRIAN
#                 \U0001F6B9|                    # MENS SYMBOL
#                 \U0001F6BA|                    # WOMENS SYMBOL
#                 \U0001F6BB|                    # RESTROOM
#                 \U0001F6BC|                    # BABY SYMBOL
#                 \U0001F6BD|                    # TOILET
#                 \U0001F6BE|                    # WATER CLOSET
#                 \U0001F6C0|                    # BATH
#                 \U000024C2|                    # CIRCLED LATIN CAPITAL LETTER M
#                 \U0001F170|                    # NEGATIVE SQUARED LATIN CAPITAL LETTER A
#                 \U0001F171|                    # NEGATIVE SQUARED LATIN CAPITAL LETTER B
#                 \U0001F17E|                    # NEGATIVE SQUARED LATIN CAPITAL LETTER O
#                 \U0001F17F|                    # NEGATIVE SQUARED LATIN CAPITAL LETTER P
#                 \U0001F18E|                    # NEGATIVE SQUARED AB
#                 \U0001F191|                    # SQUARED CL
#                 \U0001F194|                    # SQUARED ID
#                 \U0001F196|                    # SQUARED NG
#                 \U0001F199|                    # SQUARED UP WITH EXCLAMATION MARK
#                 \U0001F19A|                    # SQUARED VS
#                 \U0001F1E9 \U0001F1EA|                    # REGIONAL INDICATOR SYMBOL LETTER D + REGIONAL INDICATOR SYMBOL LETTER E
#                 \U0001F1EC \U0001F1E7|                    # REGIONAL INDICATOR SYMBOL LETTER G + REGIONAL INDICATOR SYMBOL LETTER B
#                 \U0001F1E8 \U0001F1F3|                    # REGIONAL INDICATOR SYMBOL LETTER C + REGIONAL INDICATOR SYMBOL LETTER N
#                 \U0001F1EF \U0001F1F5|                    # REGIONAL INDICATOR SYMBOL LETTER J + REGIONAL INDICATOR SYMBOL LETTER P
#                 \U0001F1F0 \U0001F1F7|                    # REGIONAL INDICATOR SYMBOL LETTER K + REGIONAL INDICATOR SYMBOL LETTER R
#                 \U0001F1EB \U0001F1F7|                    # REGIONAL INDICATOR SYMBOL LETTER F + REGIONAL INDICATOR SYMBOL LETTER R
#                 \U0001F1EA \U0001F1F8|                    # REGIONAL INDICATOR SYMBOL LETTER E + REGIONAL INDICATOR SYMBOL LETTER S
#                 \U0001F1EE \U0001F1F9|                    # REGIONAL INDICATOR SYMBOL LETTER I + REGIONAL INDICATOR SYMBOL LETTER T
#                 \U0001F1FA \U0001F1F8|                    # REGIONAL INDICATOR SYMBOL LETTER U + REGIONAL INDICATOR SYMBOL LETTER S
#                 \U0001F1F7 \U0001F1FA|                    # REGIONAL INDICATOR SYMBOL LETTER R + REGIONAL INDICATOR SYMBOL LETTER U
#                 \U0001F201|                    # SQUARED KATAKANA KOKO
#                 \U0001F202|                    # SQUARED KATAKANA SA
#                 \U0001F21A|                    # SQUARED CJK UNIFIED IDEOGRAPH-7121
#                 \U0001F22F|                    # SQUARED CJK UNIFIED IDEOGRAPH-6307
#                 \U0001F232|                    # SQUARED CJK UNIFIED IDEOGRAPH-7981
#                 \U0001F233|                    # SQUARED CJK UNIFIED IDEOGRAPH-7A7A
#                 \U0001F234|                    # SQUARED CJK UNIFIED IDEOGRAPH-5408
#                 \U0001F235|                    # SQUARED CJK UNIFIED IDEOGRAPH-6E80
#                 \U0001F236|                    # SQUARED CJK UNIFIED IDEOGRAPH-6709
#                 \U0001F237|                    # SQUARED CJK UNIFIED IDEOGRAPH-6708
#                 \U0001F238|                    # SQUARED CJK UNIFIED IDEOGRAPH-7533
#                 \U0001F239|                    # SQUARED CJK UNIFIED IDEOGRAPH-5272
#                 \U0001F23A|                    # SQUARED CJK UNIFIED IDEOGRAPH-55B6
#                 \U0001F250|                    # CIRCLED IDEOGRAPH ADVANTAGE
#                 \U0001F251|                    # CIRCLED IDEOGRAPH ACCEPT
#                 \U000000A9|                    # COPYRIGHT SIGN
#                 \U000000AE|                    # REGISTERED SIGN
#                 \U00000038 \U000020E3|                    # DIGIT EIGHT + COMBINING ENCLOSING KEYCAP
#                 \U00000039 \U000020E3|                    # DIGIT NINE + COMBINING ENCLOSING KEYCAP
#                 \U00000037 \U000020E3|                    # DIGIT SEVEN + COMBINING ENCLOSING KEYCAP
#                 \U00000036 \U000020E3|                    # DIGIT SIX + COMBINING ENCLOSING KEYCAP
#                 \U00000031 \U000020E3|                    # DIGIT ONE + COMBINING ENCLOSING KEYCAP
#                 \U00000030 \U000020E3|                    # DIGIT ZERO + COMBINING ENCLOSING KEYCAP
#                 \U00000032 \U000020E3|                    # DIGIT TWO + COMBINING ENCLOSING KEYCAP
#                 \U00000033 \U000020E3|                    # DIGIT THREE + COMBINING ENCLOSING KEYCAP
#                 \U00000035 \U000020E3|                    # DIGIT FIVE + COMBINING ENCLOSING KEYCAP
#                 \U00000034 \U000020E3|                    # DIGIT FOUR + COMBINING ENCLOSING KEYCAP
#                 \U00000023 \U000020E3|                    # NUMBER SIGN + COMBINING ENCLOSING KEYCAP
#                 \U00002122|                    # TRADE MARK SIGN
#                 \U00002139|                    # INFORMATION SOURCE
#                 \U00002194|                    # LEFT RIGHT ARROW
#                 \U00002195|                    # UP DOWN ARROW
#                 \U00002196|                    # NORTH WEST ARROW
#                 \U00002197|                    # NORTH EAST ARROW
#                 \U00002198|                    # SOUTH EAST ARROW
#                 \U00002199|                    # SOUTH WEST ARROW
#                 \U000021A9|                    # LEFTWARDS ARROW WITH HOOK
#                 \U000021AA|                    # RIGHTWARDS ARROW WITH HOOK
#                 \U0000231A|                    # WATCH
#                 \U0000231B|                    # HOURGLASS
#                 \U000023E9|                    # BLACK RIGHT-POINTING DOUBLE TRIANGLE
#                 \U000023EA|                    # BLACK LEFT-POINTING DOUBLE TRIANGLE
#                 \U000023EB|                    # BLACK UP-POINTING DOUBLE TRIANGLE
#                 \U000023EC|                    # BLACK DOWN-POINTING DOUBLE TRIANGLE
#                 \U000023F0|                    # ALARM CLOCK
#                 \U000023F3|                    # HOURGLASS WITH FLOWING SAND
#                 \U000025AA|                    # BLACK SMALL SQUARE
#                 \U000025AB|                    # WHITE SMALL SQUARE
#                 \U000025B6|                    # BLACK RIGHT-POINTING TRIANGLE
#                 \U000025C0|                    # BLACK LEFT-POINTING TRIANGLE
#                 \U000025FB|                    # WHITE MEDIUM SQUARE
#                 \U000025FC|                    # BLACK MEDIUM SQUARE
#                 \U000025FD|                    # WHITE MEDIUM SMALL SQUARE
#                 \U000025FE|                    # BLACK MEDIUM SMALL SQUARE
#                 \U00002600|                    # BLACK SUN WITH RAYS
#                 \U00002601|                    # CLOUD
#                 \U0000260E|                    # BLACK TELEPHONE
#                 \U00002611|                    # BALLOT BOX WITH CHECK
#                 \U00002614|                    # UMBRELLA WITH RAIN DROPS
#                 \U00002615|                    # HOT BEVERAGE
#                 \U0000261D|                    # WHITE UP POINTING INDEX
#                 \U0000263A|                    # WHITE SMILING FACE
#                 \U00002648|                    # ARIES
#                 \U00002649|                    # TAURUS
#                 \U0000264A|                    # GEMINI
#                 \U0000264B|                    # CANCER
#                 \U0000264C|                    # LEO
#                 \U0000264D|                    # VIRGO
#                 \U0000264E|                    # LIBRA
#                 \U0000264F|                    # SCORPIUS
#                 \U00002650|                    # SAGITTARIUS
#                 \U00002651|                    # CAPRICORN
#                 \U00002652|                    # AQUARIUS
#                 \U00002653|                    # PISCES
#                 \U00002660|                    # BLACK SPADE SUIT
#                 \U00002663|                    # BLACK CLUB SUIT
#                 \U00002665|                    # BLACK HEART SUIT
#                 \U00002666|                    # BLACK DIAMOND SUIT
#                 \U00002668|                    # HOT SPRINGS
#                 \U0000267B|                    # BLACK UNIVERSAL RECYCLING SYMBOL
#                 \U0000267F|                    # WHEELCHAIR SYMBOL
#                 \U00002693|                    # ANCHOR
#                 \U000026A0|                    # WARNING SIGN
#                 \U000026A1|                    # HIGH VOLTAGE SIGN
#                 \U000026AA|                    # MEDIUM WHITE CIRCLE
#                 \U000026AB|                    # MEDIUM BLACK CIRCLE
#                 \U000026BD|                    # SOCCER BALL
#                 \U000026BE|                    # BASEBALL
#                 \U000026C4|                    # SNOWMAN WITHOUT SNOW
#                 \U000026C5|                    # SUN BEHIND CLOUD
#                 \U000026CE|                    # OPHIUCHUS
#                 \U000026D4|                    # NO ENTRY
#                 \U000026EA|                    # CHURCH
#                 \U000026F2|                    # FOUNTAIN
#                 \U000026F3|                    # FLAG IN HOLE
#                 \U000026F5|                    # SAILBOAT
#                 \U000026FA|                    # TENT
#                 \U000026FD|                    # FUEL PUMP
#                 \U00002934|                    # ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS
#                 \U00002935|                    # ARROW POINTING RIGHTWARDS THEN CURVING DOWNWARDS
#                 \U00002B05|                    # LEFTWARDS BLACK ARROW
#                 \U00002B06|                    # UPWARDS BLACK ARROW
#                 \U00002B07|                    # DOWNWARDS BLACK ARROW
#                 \U00002B1B|                    # BLACK LARGE SQUARE
#                 \U00002B1C|                    # WHITE LARGE SQUARE
#                 \U00002B50|                    # WHITE MEDIUM STAR
#                 \U00002B55|                    # HEAVY LARGE CIRCLE
#                 \U00003030|                    # WAVY DASH
#                 \U0000303D|                    # PART ALTERNATION MARK
#                 \U00003297|                    # CIRCLED IDEOGRAPH CONGRATULATION
#                 \U00003299|                    # CIRCLED IDEOGRAPH SECRET
#                 \U0001F004|                    # MAHJONG TILE RED DRAGON
#                 \U0001F0CF|                    # PLAYING CARD BLACK JOKER
#                 \U0001F300|                    # CYCLONE
#                 \U0001F301|                    # FOGGY
#                 \U0001F302|                    # CLOSED UMBRELLA
#                 \U0001F303|                    # NIGHT WITH STARS
#                 \U0001F304|                    # SUNRISE OVER MOUNTAINS
#                 \U0001F305|                    # SUNRISE
#                 \U0001F306|                    # CITYSCAPE AT DUSK
#                 \U0001F307|                    # SUNSET OVER BUILDINGS
#                 \U0001F308|                    # RAINBOW
#                 \U0001F309|                    # BRIDGE AT NIGHT
#                 \U0001F30A|                    # WATER WAVE
#                 \U0001F30B|                    # VOLCANO
#                 \U0001F30C|                    # MILKY WAY
#                 \U0001F30F|                    # EARTH GLOBE ASIA-AUSTRALIA
#                 \U0001F311|                    # NEW MOON SYMBOL
#                 \U0001F313|                    # FIRST QUARTER MOON SYMBOL
#                 \U0001F314|                    # WAXING GIBBOUS MOON SYMBOL
#                 \U0001F315|                    # FULL MOON SYMBOL
#                 \U0001F319|                    # CRESCENT MOON
#                 \U0001F31B|                    # FIRST QUARTER MOON WITH FACE
#                 \U0001F31F|                    # GLOWING STAR
#                 \U0001F320|                    # SHOOTING STAR
#                 \U0001F330|                    # CHESTNUT
#                 \U0001F331|                    # SEEDLING
#                 \U0001F334|                    # PALM TREE
#                 \U0001F335|                    # CACTUS
#                 \U0001F337|                    # TULIP
#                 \U0001F338|                    # CHERRY BLOSSOM
#                 \U0001F339|                    # ROSE
#                 \U0001F33A|                    # HIBISCUS
#                 \U0001F33B|                    # SUNFLOWER
#                 \U0001F33C|                    # BLOSSOM
#                 \U0001F33D|                    # EAR OF MAIZE
#                 \U0001F33E|                    # EAR OF RICE
#                 \U0001F33F|                    # HERB
#                 \U0001F340|                    # FOUR LEAF CLOVER
#                 \U0001F341|                    # MAPLE LEAF
#                 \U0001F342|                    # FALLEN LEAF
#                 \U0001F343|                    # LEAF FLUTTERING IN WIND
#                 \U0001F344|                    # MUSHROOM
#                 \U0001F345|                    # TOMATO
#                 \U0001F346|                    # AUBERGINE
#                 \U0001F347|                    # GRAPES
#                 \U0001F348|                    # MELON
#                 \U0001F349|                    # WATERMELON
#                 \U0001F34A|                    # TANGERINE
#                 \U0001F34C|                    # BANANA
#                 \U0001F34D|                    # PINEAPPLE
#                 \U0001F34E|                    # RED APPLE
#                 \U0001F34F|                    # GREEN APPLE
#                 \U0001F351|                    # PEACH
#                 \U0001F352|                    # CHERRIES
#                 \U0001F353|                    # STRAWBERRY
#                 \U0001F354|                    # HAMBURGER
#                 \U0001F355|                    # SLICE OF PIZZA
#                 \U0001F356|                    # MEAT ON BONE
#                 \U0001F357|                    # POULTRY LEG
#                 \U0001F358|                    # RICE CRACKER
#                 \U0001F359|                    # RICE BALL
#                 \U0001F35A|                    # COOKED RICE
#                 \U0001F35B|                    # CURRY AND RICE
#                 \U0001F35C|                    # STEAMING BOWL
#                 \U0001F35D|                    # SPAGHETTI
#                 \U0001F35E|                    # BREAD
#                 \U0001F35F|                    # FRENCH FRIES
#                 \U0001F360|                    # ROASTED SWEET POTATO
#                 \U0001F361|                    # DANGO
#                 \U0001F362|                    # ODEN
#                 \U0001F363|                    # SUSHI
#                 \U0001F364|                    # FRIED SHRIMP
#                 \U0001F365|                    # FISH CAKE WITH SWIRL DESIGN
#                 \U0001F366|                    # SOFT ICE CREAM
#                 \U0001F367|                    # SHAVED ICE
#                 \U0001F368|                    # ICE CREAM
#                 \U0001F369|                    # DOUGHNUT
#                 \U0001F36A|                    # COOKIE
#                 \U0001F36B|                    # CHOCOLATE BAR
#                 \U0001F36C|                    # CANDY
#                 \U0001F36D|                    # LOLLIPOP
#                 \U0001F36E|                    # CUSTARD
#                 \U0001F36F|                    # HONEY POT
#                 \U0001F370|                    # SHORTCAKE
#                 \U0001F371|                    # BENTO BOX
#                 \U0001F372|                    # POT OF FOOD
#                 \U0001F373|                    # COOKING
#                 \U0001F374|                    # FORK AND KNIFE
#                 \U0001F375|                    # TEACUP WITHOUT HANDLE
#                 \U0001F376|                    # SAKE BOTTLE AND CUP
#                 \U0001F377|                    # WINE GLASS
#                 \U0001F378|                    # COCKTAIL GLASS
#                 \U0001F379|                    # TROPICAL DRINK
#                 \U0001F37A|                    # BEER MUG
#                 \U0001F37B|                    # CLINKING BEER MUGS
#                 \U0001F380|                    # RIBBON
#                 \U0001F381|                    # WRAPPED PRESENT
#                 \U0001F382|                    # BIRTHDAY CAKE
#                 \U0001F383|                    # JACK-O-LANTERN
#                 \U0001F384|                    # CHRISTMAS TREE
#                 \U0001F385|                    # FATHER CHRISTMAS
#                 \U0001F386|                    # FIREWORKS
#                 \U0001F387|                    # FIREWORK SPARKLER
#                 \U0001F388|                    # BALLOON
#                 \U0001F389|                    # PARTY POPPER
#                 \U0001F38A|                    # CONFETTI BALL
#                 \U0001F38B|                    # TANABATA TREE
#                 \U0001F38C|                    # CROSSED FLAGS
#                 \U0001F38D|                    # PINE DECORATION
#                 \U0001F38E|                    # JAPANESE DOLLS
#                 \U0001F38F|                    # CARP STREAMER
#                 \U0001F390|                    # WIND CHIME
#                 \U0001F391|                    # MOON VIEWING CEREMONY
#                 \U0001F392|                    # SCHOOL SATCHEL
#                 \U0001F393|                    # GRADUATION CAP
#                 \U0001F3A0|                    # CAROUSEL HORSE
#                 \U0001F3A1|                    # FERRIS WHEEL
#                 \U0001F3A2|                    # ROLLER COASTER
#                 \U0001F3A3|                    # FISHING POLE AND FISH
#                 \U0001F3A4|                    # MICROPHONE
#                 \U0001F3A5|                    # MOVIE CAMERA
#                 \U0001F3A6|                    # CINEMA
#                 \U0001F3A7|                    # HEADPHONE
#                 \U0001F3A8|                    # ARTIST PALETTE
#                 \U0001F3A9|                    # TOP HAT
#                 \U0001F3AA|                    # CIRCUS TENT
#                 \U0001F3AB|                    # TICKET
#                 \U0001F3AC|                    # CLAPPER BOARD
#                 \U0001F3AD|                    # PERFORMING ARTS
#                 \U0001F3AE|                    # VIDEO GAME
#                 \U0001F3AF|                    # DIRECT HIT
#                 \U0001F3B0|                    # SLOT MACHINE
#                 \U0001F3B1|                    # BILLIARDS
#                 \U0001F3B2|                    # GAME DIE
#                 \U0001F3B3|                    # BOWLING
#                 \U0001F3B4|                    # FLOWER PLAYING CARDS
#                 \U0001F3B5|                    # MUSICAL NOTE
#                 \U0001F3B6|                    # MULTIPLE MUSICAL NOTES
#                 \U0001F3B7|                    # SAXOPHONE
#                 \U0001F3B8|                    # GUITAR
#                 \U0001F3B9|                    # MUSICAL KEYBOARD
#                 \U0001F3BA|                    # TRUMPET
#                 \U0001F3BB|                    # VIOLIN
#                 \U0001F3BC|                    # MUSICAL SCORE
#                 \U0001F3BD|                    # RUNNING SHIRT WITH SASH
#                 \U0001F3BE|                    # TENNIS RACQUET AND BALL
#                 \U0001F3BF|                    # SKI AND SKI BOOT
#                 \U0001F3C0|                    # BASKETBALL AND HOOP
#                 \U0001F3C1|                    # CHEQUERED FLAG
#                 \U0001F3C2|                    # SNOWBOARDER
#                 \U0001F3C3|                    # RUNNER
#                 \U0001F3C4|                    # SURFER
#                 \U0001F3C6|                    # TROPHY
#                 \U0001F3C8|                    # AMERICAN FOOTBALL
#                 \U0001F3CA|                    # SWIMMER
#                 \U0001F3E0|                    # HOUSE BUILDING
#                 \U0001F3E1|                    # HOUSE WITH GARDEN
#                 \U0001F3E2|                    # OFFICE BUILDING
#                 \U0001F3E3|                    # JAPANESE POST OFFICE
#                 \U0001F3E5|                    # HOSPITAL
#                 \U0001F3E6|                    # BANK
#                 \U0001F3E7|                    # AUTOMATED TELLER MACHINE
#                 \U0001F3E8|                    # HOTEL
#                 \U0001F3E9|                    # LOVE HOTEL
#                 \U0001F3EA|                    # CONVENIENCE STORE
#                 \U0001F3EB|                    # SCHOOL
#                 \U0001F3EC|                    # DEPARTMENT STORE
#                 \U0001F3ED|                    # FACTORY
#                 \U0001F3EE|                    # IZAKAYA LANTERN
#                 \U0001F3EF|                    # JAPANESE CASTLE
#                 \U0001F3F0|                    # EUROPEAN CASTLE
#                 \U0001F40C|                    # SNAIL
#                 \U0001F40D|                    # SNAKE
#                 \U0001F40E|                    # HORSE
#                 \U0001F411|                    # SHEEP
#                 \U0001F412|                    # MONKEY
#                 \U0001F414|                    # CHICKEN
#                 \U0001F417|                    # BOAR
#                 \U0001F418|                    # ELEPHANT
#                 \U0001F419|                    # OCTOPUS
#                 \U0001F41A|                    # SPIRAL SHELL
#                 \U0001F41B|                    # BUG
#                 \U0001F41C|                    # ANT
#                 \U0001F41D|                    # HONEYBEE
#                 \U0001F41E|                    # LADY BEETLE
#                 \U0001F41F|                    # FISH
#                 \U0001F420|                    # TROPICAL FISH
#                 \U0001F421|                    # BLOWFISH
#                 \U0001F422|                    # TURTLE
#                 \U0001F423|                    # HATCHING CHICK
#                 \U0001F424|                    # BABY CHICK
#                 \U0001F425|                    # FRONT-FACING BABY CHICK
#                 \U0001F426|                    # BIRD
#                 \U0001F427|                    # PENGUIN
#                 \U0001F428|                    # KOALA
#                 \U0001F429|                    # POODLE
#                 \U0001F42B|                    # BACTRIAN CAMEL
#                 \U0001F42C|                    # DOLPHIN
#                 \U0001F42D|                    # MOUSE FACE
#                 \U0001F42E|                    # COW FACE
#                 \U0001F42F|                    # TIGER FACE
#                 \U0001F430|                    # RABBIT FACE
#                 \U0001F431|                    # CAT FACE
#                 \U0001F432|                    # DRAGON FACE
#                 \U0001F433|                    # SPOUTING WHALE
#                 \U0001F434|                    # HORSE FACE
#                 \U0001F435|                    # MONKEY FACE
#                 \U0001F436|                    # DOG FACE
#                 \U0001F437|                    # PIG FACE
#                 \U0001F438|                    # FROG FACE
#                 \U0001F439|                    # HAMSTER FACE
#                 \U0001F43A|                    # WOLF FACE
#                 \U0001F43B|                    # BEAR FACE
#                 \U0001F43C|                    # PANDA FACE
#                 \U0001F43D|                    # PIG NOSE
#                 \U0001F43E|                    # PAW PRINTS
#                 \U0001F440|                    # EYES
#                 \U0001F442|                    # EAR
#                 \U0001F443|                    # NOSE
#                 \U0001F444|                    # MOUTH
#                 \U0001F445|                    # TONGUE
#                 \U0001F446|                    # WHITE UP POINTING BACKHAND INDEX
#                 \U0001F447|                    # WHITE DOWN POINTING BACKHAND INDEX
#                 \U0001F448|                    # WHITE LEFT POINTING BACKHAND INDEX
#                 \U0001F449|                    # WHITE RIGHT POINTING BACKHAND INDEX
#                 \U0001F44A|                    # FISTED HAND SIGN
#                 \U0001F44B|                    # WAVING HAND SIGN
#                 \U0001F44C|                    # OK HAND SIGN
#                 \U0001F44D|                    # THUMBS UP SIGN
#                 \U0001F44E|                    # THUMBS DOWN SIGN
#                 \U0001F44F|                    # CLAPPING HANDS SIGN
#                 \U0001F450|                    # OPEN HANDS SIGN
#                 \U0001F451|                    # CROWN
#                 \U0001F452|                    # WOMANS HAT
#                 \U0001F453|                    # EYEGLASSES
#                 \U0001F454|                    # NECKTIE
#                 \U0001F455|                    # T-SHIRT
#                 \U0001F456|                    # JEANS
#                 \U0001F457|                    # DRESS
#                 \U0001F458|                    # KIMONO
#                 \U0001F459|                    # BIKINI
#                 \U0001F45A|                    # WOMANS CLOTHES
#                 \U0001F45B|                    # PURSE
#                 \U0001F45C|                    # HANDBAG
#                 \U0001F45D|                    # POUCH
#                 \U0001F45E|                    # MANS SHOE
#                 \U0001F45F|                    # ATHLETIC SHOE
#                 \U0001F460|                    # HIGH-HEELED SHOE
#                 \U0001F461|                    # WOMANS SANDAL
#                 \U0001F462|                    # WOMANS BOOTS
#                 \U0001F463|                    # FOOTPRINTS
#                 \U0001F464|                    # BUST IN SILHOUETTE
#                 \U0001F466|                    # BOY
#                 \U0001F467|                    # GIRL
#                 \U0001F468|                    # MAN
#                 \U0001F469|                    # WOMAN
#                 \U0001F46A|                    # FAMILY
#                 \U0001F46B|                    # MAN AND WOMAN HOLDING HANDS
#                 \U0001F46E|                    # POLICE OFFICER
#                 \U0001F46F|                    # WOMAN WITH BUNNY EARS
#                 \U0001F470|                    # BRIDE WITH VEIL
#                 \U0001F471|                    # PERSON WITH BLOND HAIR
#                 \U0001F472|                    # MAN WITH GUA PI MAO
#                 \U0001F473|                    # MAN WITH TURBAN
#                 \U0001F474|                    # OLDER MAN
#                 \U0001F475|                    # OLDER WOMAN
#                 \U0001F476|                    # BABY
#                 \U0001F477|                    # CONSTRUCTION WORKER
#                 \U0001F478|                    # PRINCESS
#                 \U0001F479|                    # JAPANESE OGRE
#                 \U0001F47A|                    # JAPANESE GOBLIN
#                 \U0001F47B|                    # GHOST
#                 \U0001F47C|                    # BABY ANGEL
#                 \U0001F47D|                    # EXTRATERRESTRIAL ALIEN
#                 \U0001F47E|                    # ALIEN MONSTER
#                 \U0001F47F|                    # IMP
#                 \U0001F480|                    # SKULL
#                 \U0001F481|                    # INFORMATION DESK PERSON
#                 \U0001F482|                    # GUARDSMAN
#                 \U0001F483|                    # DANCER
#                 \U0001F484|                    # LIPSTICK
#                 \U0001F485|                    # NAIL POLISH
#                 \U0001F486|                    # FACE MASSAGE
#                 \U0001F487|                    # HAIRCUT
#                 \U0001F488|                    # BARBER POLE
#                 \U0001F489|                    # SYRINGE
#                 \U0001F48A|                    # PILL
#                 \U0001F4A0|                    # DIAMOND SHAPE WITH A DOT INSIDE
#                 \U0001F4A1|                    # ELECTRIC LIGHT BULB
#                 \U0001F4A6|                    # SPLASHING SWEAT SYMBOL
#                 \U0001F4A7|                    # DROPLET
#                 \U0001F4A8|                    # DASH SYMBOL
#                 \U0001F4A9|                    # PILE OF POO
#                 \U0001F4AA|                    # FLEXED BICEPS
#                 \U0001F4AB|                    # DIZZY SYMBOL
#                 \U0001F4AC|                    # SPEECH BALLOON
#                 \U0001F4AE|                    # WHITE FLOWER
#                 \U0001F4AF|                    # HUNDRED POINTS SYMBOL
#                 \U0001F4B0|                    # MONEY BAG
#                 \U0001F4B1|                    # CURRENCY EXCHANGE
#                 \U0001F4B2|                    # HEAVY DOLLAR SIGN
#                 \U0001F4B3|                    # CREDIT CARD
#                 \U0001F4B4|                    # BANKNOTE WITH YEN SIGN
#                 \U0001F4B5|                    # BANKNOTE WITH DOLLAR SIGN
#                 \U0001F4B8|                    # MONEY WITH WINGS
#                 \U0001F4B9|                    # CHART WITH UPWARDS TREND AND YEN SIGN
#                 \U0001F4BA|                    # SEAT
#                 \U0001F4BB|                    # PERSONAL COMPUTER
#                 \U0001F4BC|                    # BRIEFCASE
#                 \U0001F4BD|                    # MINIDISC
#                 \U0001F4BE|                    # FLOPPY DISK
#                 \U0001F4BF|                    # OPTICAL DISC
#                 \U0001F4C0|                    # DVD
#                 \U0001F4C1|                    # FILE FOLDER
#                 \U0001F4C2|                    # OPEN FILE FOLDER
#                 \U0001F4C3|                    # PAGE WITH CURL
#                 \U0001F4C4|                    # PAGE FACING UP
#                 \U0001F4C5|                    # CALENDAR
#                 \U0001F4C6|                    # TEAR-OFF CALENDAR
#                 \U0001F4C7|                    # CARD INDEX
#                 \U0001F4C8|                    # CHART WITH UPWARDS TREND
#                 \U0001F4C9|                    # CHART WITH DOWNWARDS TREND
#                 \U0001F4CA|                    # BAR CHART
#                 \U0001F4CB|                    # CLIPBOARD
#                 \U0001F4CC|                    # PUSHPIN
#                 \U0001F4CD|                    # ROUND PUSHPIN
#                 \U0001F4CE|                    # PAPERCLIP
#                 \U0001F4CF|                    # STRAIGHT RULER
#                 \U0001F4D0|                    # TRIANGULAR RULER
#                 \U0001F4D1|                    # BOOKMARK TABS
#                 \U0001F4D2|                    # LEDGER
#                 \U0001F4D3|                    # NOTEBOOK
#                 \U0001F4D4|                    # NOTEBOOK WITH DECORATIVE COVER
#                 \U0001F4D5|                    # CLOSED BOOK
#                 \U0001F4D6|                    # OPEN BOOK
#                 \U0001F4D7|                    # GREEN BOOK
#                 \U0001F4D8|                    # BLUE BOOK
#                 \U0001F4D9|                    # ORANGE BOOK
#                 \U0001F4DA|                    # BOOKS
#                 \U0001F4DB|                    # NAME BADGE
#                 \U0001F4DC|                    # SCROLL
#                 \U0001F4DD|                    # MEMO
#                 \U0001F4DE|                    # TELEPHONE RECEIVER
#                 \U0001F4DF|                    # PAGER
#                 \U0001F4E0|                    # FAX MACHINE
#                 \U0001F4E1|                    # SATELLITE ANTENNA
#                 \U0001F4E2|                    # PUBLIC ADDRESS LOUDSPEAKER
#                 \U0001F4E3|                    # CHEERING MEGAPHONE
#                 \U0001F4E4|                    # OUTBOX TRAY
#                 \U0001F4E5|                    # INBOX TRAY
#                 \U0001F4E6|                    # PACKAGE
#                 \U0001F4E7|                    # E-MAIL SYMBOL
#                 \U0001F4E8|                    # INCOMING ENVELOPE
#                 \U0001F4E9|                    # ENVELOPE WITH DOWNWARDS ARROW ABOVE
#                 \U0001F4EA|                    # CLOSED MAILBOX WITH LOWERED FLAG
#                 \U0001F4EB|                    # CLOSED MAILBOX WITH RAISED FLAG
#                 \U0001F4EE|                    # POSTBOX
#                 \U0001F4F0|                    # NEWSPAPER
#                 \U0001F4F1|                    # MOBILE PHONE
#                 \U0001F4F2|                    # MOBILE PHONE WITH RIGHTWARDS ARROW AT LEFT
#                 \U0001F4F3|                    # VIBRATION MODE
#                 \U0001F4F4|                    # MOBILE PHONE OFF
#                 \U0001F4F6|                    # ANTENNA WITH BARS
#                 \U0001F4F7|                    # CAMERA
#                 \U0001F4F9|                    # VIDEO CAMERA
#                 \U0001F4FA|                    # TELEVISION
#                 \U0001F4FB|                    # RADIO
#                 \U0001F4FC|                    # VIDEOCASSETTE
#                 \U0001F503|                    # CLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS
#                 \U0001F50A|                    # SPEAKER WITH THREE SOUND WAVES
#                 \U0001F50B|                    # BATTERY
#                 \U0001F50C|                    # ELECTRIC PLUG
#                 \U0001F50D|                    # LEFT-POINTING MAGNIFYING GLASS
#                 \U0001F50E|                    # RIGHT-POINTING MAGNIFYING GLASS
#                 \U0001F50F|                    # LOCK WITH INK PEN
#                 \U0001F510|                    # CLOSED LOCK WITH KEY
#                 \U0001F511|                    # KEY
#                 \U0001F512|                    # LOCK
#                 \U0001F513|                    # OPEN LOCK
#                 \U0001F514|                    # BELL
#                 \U0001F516|                    # BOOKMARK
#                 \U0001F517|                    # LINK SYMBOL
#                 \U0001F518|                    # RADIO BUTTON
#                 \U0001F519|                    # BACK WITH LEFTWARDS ARROW ABOVE
#                 \U0001F51A|                    # END WITH LEFTWARDS ARROW ABOVE
#                 \U0001F51B|                    # ON WITH EXCLAMATION MARK WITH LEFT RIGHT ARROW ABOVE
#                 \U0001F51C|                    # SOON WITH RIGHTWARDS ARROW ABOVE
#                 \U0001F51D|                    # TOP WITH UPWARDS ARROW ABOVE
#                 \U0001F51E|                    # NO ONE UNDER EIGHTEEN SYMBOL
#                 \U0001F51F|                    # KEYCAP TEN
#                 \U0001F520|                    # INPUT SYMBOL FOR LATIN CAPITAL LETTERS
#                 \U0001F521|                    # INPUT SYMBOL FOR LATIN SMALL LETTERS
#                 \U0001F522|                    # INPUT SYMBOL FOR NUMBERS
#                 \U0001F523|                    # INPUT SYMBOL FOR SYMBOLS
#                 \U0001F524|                    # INPUT SYMBOL FOR LATIN LETTERS
#                 \U0001F525|                    # FIRE
#                 \U0001F526|                    # ELECTRIC TORCH
#                 \U0001F527|                    # WRENCH
#                 \U0001F528|                    # HAMMER
#                 \U0001F529|                    # NUT AND BOLT
#                 \U0001F52A|                    # HOCHO
#                 \U0001F52B|                    # PISTOL
#                 \U0001F52E|                    # CRYSTAL BALL
#                 \U0001F52F|                    # SIX POINTED STAR WITH MIDDLE DOT
#                 \U0001F530|                    # JAPANESE SYMBOL FOR BEGINNER
#                 \U0001F531|                    # TRIDENT EMBLEM
#                 \U0001F532|                    # BLACK SQUARE BUTTON
#                 \U0001F533|                    # WHITE SQUARE BUTTON
#                 \U0001F534|                    # LARGE RED CIRCLE
#                 \U0001F535|                    # LARGE BLUE CIRCLE
#                 \U0001F536|                    # LARGE ORANGE DIAMOND
#                 \U0001F537|                    # LARGE BLUE DIAMOND
#                 \U0001F538|                    # SMALL ORANGE DIAMOND
#                 \U0001F539|                    # SMALL BLUE DIAMOND
#                 \U0001F53A|                    # UP-POINTING RED TRIANGLE
#                 \U0001F53B|                    # DOWN-POINTING RED TRIANGLE
#                 \U0001F53C|                    # UP-POINTING SMALL RED TRIANGLE
#                 \U0001F53D|                    # DOWN-POINTING SMALL RED TRIANGLE
#                 \U0001F550|                    # CLOCK FACE ONE OCLOCK
#                 \U0001F551|                    # CLOCK FACE TWO OCLOCK
#                 \U0001F552|                    # CLOCK FACE THREE OCLOCK
#                 \U0001F553|                    # CLOCK FACE FOUR OCLOCK
#                 \U0001F554|                    # CLOCK FACE FIVE OCLOCK
#                 \U0001F555|                    # CLOCK FACE SIX OCLOCK
#                 \U0001F556|                    # CLOCK FACE SEVEN OCLOCK
#                 \U0001F557|                    # CLOCK FACE EIGHT OCLOCK
#                 \U0001F558|                    # CLOCK FACE NINE OCLOCK
#                 \U0001F559|                    # CLOCK FACE TEN OCLOCK
#                 \U0001F55A|                    # CLOCK FACE ELEVEN OCLOCK
#                 \U0001F55B|                    # CLOCK FACE TWELVE OCLOCK
#                 \U0001F5FB|                    # MOUNT FUJI
#                 \U0001F5FC|                    # TOKYO TOWER
#                 \U0001F5FD|                    # STATUE OF LIBERTY
#                 \U0001F5FE|                    # SILHOUETTE OF JAPAN
#                 \U0001F5FF|                    # MOYAI
#                 \U0001F62F|                    # HUSHED FACE
#                 \U0001F681|                    # HELICOPTER
#                 \U0001F682|                    # STEAM LOCOMOTIVE
#                 \U0001F686|                    # TRAIN
#                 \U0001F688|                    # LIGHT RAIL
#                 \U0001F68A|                    # TRAM
#                 \U0001F68D|                    # ONCOMING BUS
#                 \U0001F68E|                    # TROLLEYBUS
#                 \U0001F690|                    # MINIBUS
#                 \U0001F694|                    # ONCOMING POLICE CAR
#                 \U0001F696|                    # ONCOMING TAXI
#                 \U0001F698|                    # ONCOMING AUTOMOBILE
#                 \U0001F69B|                    # ARTICULATED LORRY
#                 \U0001F69C|                    # TRACTOR
#                 \U0001F69D|                    # MONORAIL
#                 \U0001F69E|                    # MOUNTAIN RAILWAY
#                 \U0001F69F|                    # SUSPENSION RAILWAY
#                 \U0001F6A0|                    # MOUNTAIN CABLEWAY
#                 \U0001F6A1|                    # AERIAL TRAMWAY
#                 \U0001F6A3|                    # ROWBOAT
#                 \U0001F6A6|                    # VERTICAL TRAFFIC LIGHT
#                 \U0001F6AE|                    # PUT LITTER IN ITS PLACE SYMBOL
#                 \U0001F6AF|                    # DO NOT LITTER SYMBOL
#                 \U0001F6B0|                    # POTABLE WATER SYMBOL
#                 \U0001F6B1|                    # NON-POTABLE WATER SYMBOL
#                 \U0001F6B3|                    # NO BICYCLES
#                 \U0001F6B4|                    # BICYCLIST
#                 \U0001F6B5|                    # MOUNTAIN BICYCLIST
#                 \U0001F6B7|                    # NO PEDESTRIANS
#                 \U0001F6B8|                    # CHILDREN CROSSING
#                 \U0001F6BF|                    # SHOWER
#                 \U0001F6C1|                    # BATHTUB
#                 \U0001F6C2|                    # PASSPORT CONTROL
#                 \U0001F6C3|                    # CUSTOMS
#                 \U0001F6C4|                    # BAGGAGE CLAIM
#                 \U0001F6C5|                    # LEFT LUGGAGE
#                 \U0001F30D|                    # EARTH GLOBE EUROPE-AFRICA
#                 \U0001F30E|                    # EARTH GLOBE AMERICAS
#                 \U0001F310|                    # GLOBE WITH MERIDIANS
#                 \U0001F312|                    # WAXING CRESCENT MOON SYMBOL
#                 \U0001F316|                    # WANING GIBBOUS MOON SYMBOL
#                 \U0001F317|                    # LAST QUARTER MOON SYMBOL
#                 \U0001F318|                    # WANING CRESCENT MOON SYMBOL
#                 \U0001F31A|                    # NEW MOON WITH FACE
#                 \U0001F31C|                    # LAST QUARTER MOON WITH FACE
#                 \U0001F31D|                    # FULL MOON WITH FACE
#                 \U0001F31E|                    # SUN WITH FACE
#                 \U0001F332|                    # EVERGREEN TREE
#                 \U0001F333|                    # DECIDUOUS TREE
#                 \U0001F34B|                    # LEMON
#                 \U0001F350|                    # PEAR
#                 \U0001F37C|                    # BABY BOTTLE
#                 \U0001F3C7|                    # HORSE RACING
#                 \U0001F3C9|                    # RUGBY FOOTBALL
#                 \U0001F3E4|                    # EUROPEAN POST OFFICE
#                 \U0001F400|                    # RAT
#                 \U0001F401|                    # MOUSE
#                 \U0001F402|                    # OX
#                 \U0001F403|                    # WATER BUFFALO
#                 \U0001F404|                    # COW
#                 \U0001F405|                    # TIGER
#                 \U0001F406|                    # LEOPARD
#                 \U0001F407|                    # RABBIT
#                 \U0001F408|                    # CAT
#                 \U0001F409|                    # DRAGON
#                 \U0001F40A|                    # CROCODILE
#                 \U0001F40B|                    # WHALE
#                 \U0001F40F|                    # RAM
#                 \U0001F410|                    # GOAT
#                 \U0001F413|                    # ROOSTER
#                 \U0001F415|                    # DOG
#                 \U0001F416|                    # PIG
#                 \U0001F42A|                    # DROMEDARY CAMEL
#                 \U0001F465|                    # BUSTS IN SILHOUETTE
#                 \U0001F46C|                    # TWO MEN HOLDING HANDS
#                 \U0001F46D|                    # TWO WOMEN HOLDING HANDS
#                 \U0001F4AD|                    # THOUGHT BALLOON
#                 \U0001F4B6|                    # BANKNOTE WITH EURO SIGN
#                 \U0001F4B7|                    # BANKNOTE WITH POUND SIGN
#                 \U0001F4EC|                    # OPEN MAILBOX WITH RAISED FLAG
#                 \U0001F4ED|                    # OPEN MAILBOX WITH LOWERED FLAG
#                 \U0001F4EF|                    # POSTAL HORN
#                 \U0001F4F5|                    # NO MOBILE PHONES
#                 \U0001F500|                    # TWISTED RIGHTWARDS ARROWS
#                 \U0001F501|                    # CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS
#                 \U0001F502|                    # CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS WITH CIRCLED ONE OVERLAY
#                 \U0001F504|                    # ANTICLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS
#                 \U0001F505|                    # LOW BRIGHTNESS SYMBOL
#                 \U0001F506|                    # HIGH BRIGHTNESS SYMBOL
#                 \U0001F507|                    # SPEAKER WITH CANCELLATION STROKE
#                 \U0001F509|                    # SPEAKER WITH ONE SOUND WAVE
#                 \U0001F515|                    # BELL WITH CANCELLATION STROKE
#                 \U0001F52C|                    # MICROSCOPE
#                 \U0001F52D|                    # TELESCOPE
#                 \U0001F55C|                    # CLOCK FACE ONE-THIRTY
#                 \U0001F55D|                    # CLOCK FACE TWO-THIRTY
#                 \U0001F55E|                    # CLOCK FACE THREE-THIRTY
#                 \U0001F55F|                    # CLOCK FACE FOUR-THIRTY
#                 \U0001F560|                    # CLOCK FACE FIVE-THIRTY
#                 \U0001F561|                    # CLOCK FACE SIX-THIRTY
#                 \U0001F562|                    # CLOCK FACE SEVEN-THIRTY
#                 \U0001F563|                    # CLOCK FACE EIGHT-THIRTY
#                 \U0001F564|                    # CLOCK FACE NINE-THIRTY
#                 \U0001F565|                    # CLOCK FACE TEN-THIRTY
#                 \U0001F566|                    # CLOCK FACE ELEVEN-THIRTY
#                 \U0001F567                     # CLOCK FACE TWELVE-THIRTY
#                 """
#     ),
}

REGEX_FEATURE_CONFIG_ARFF = {
    # ---- Emoticons ----
#     u"Emoticon Happy": (u":-)", 
#                         r"""[:=][o-]?[)}>\]]|               # :-) :o) :)
#                         [({<\[][o-]?[:=]|                   # (-: (o: (:
#                         \^(_*|[-oO]?)\^                     # ^^ ^-^
#                         """
#     ), 
    u"Emoticon Laughing": (u":-D", r"""([:=][-]?|x)[D]"""),   # :-D xD
    u"Emoticon Winking": (u";-)", 
                        r"""[;\*][-o]?[)}>\]]|              # ;-) ;o) ;)
                        [({<\[][-o]?[;\*]                   # (-; (
                        """
    ), 
#     u"Emotion Tongue": (u":-P", 
#                         r"""[:=][-]?[pqP](?!\w)|            # :-P :P
#                         (?<!\w)[pqP][-]?[:=]                # q-: P-:
#                         """
#     ),  
    u"Emoticon Surprise": (u":-O", 
                            r"""(?<!\w|\.)                  # Boundary
                            ([:=]-?[oO0]|                   # :-O
                            [oO0]-?[:=]|                    # O-:
                            [oO](_*|\.)[oO])                # Oo O____o O.o
                            (?!\w)
                            """
    ), 
#     u"Emoticon Dissatisfied": (u":-/", 
#                                 r"""(?<!\w)                 # Boundary
#                                 [:=][-o]?[\/\\|]|           # :-/ :-\ :-| :/
#                                 [\/\\|][-o]?[:=]|           # \-: \:
#                                 -_+-                        # -_- -___-
#                                 """
#     ), 
#     u"Emoticon Sad": (u":-(", 
#                         r"""[:=][o-]?[({<\[]|               # :-( :(
#                         (?<!(\w|%))                         # Boundary
#                         [)}>\[][o-]?[:=]                    # )-: ): )o: 
#                         """
#     ), 
#     u"Emoticon Crying": (u";-(", 
#                         r"""(([:=]')|(;'?))[o-]?[({<\[]|    # ;-( :'(
#                         (?<!(\w|%))                         # Boundary
#                         [)}>\[][o-]?(('[:=])|('?;))         # )-; )-';
#                         """
#     ), 
#      
#     # ---- Punctuation----
#     # u"AllPunctuation": (u"", r"""((\.{2,}|[?!]{2,})1*)"""),
#     u"Question Mark": (u"??", r"""\?{2,}"""),                 # ??
    u"Exclamation Mark": (u"!!", r"""\!{2,}"""),              # !!
    u"Question and Exclamation Mark": (u"?!", r"""[\!\?]*((\?\!)+|              # ?!
                    (\!\?)+)[\!\?]*                         # !?
                    """
    ),                                          # Unicode interrobang: U+203D
#     u"Ellipsis": (u"...", r"""\.{2,}|                         # .. ...
#                 \.(\ \.){2,}                                # . . .
#                 """
#     ),                                          # Unicode Ellipsis: U+2026
#     # ---- Markup----
# #     u"Hashtag": (u"#", r"""\#(irony|ironic|sarcasm|sarcastic)"""),  # #irony
# #     u"Pseudo-Tag": (u"Tag", 
# #                     r"""([<\[][\/\\]
# #                     (irony|ironic|sarcasm|sarcastic)        # </irony>
# #                     [>\]])|                                 #
# #                     ((?<!(\w|[<\[]))[\/\\]                  #
# #                     (irony|ironic|sarcasm|sarcastic)        # /irony
# #                     (?![>\]]))
# #                     """
# #     ),
#  
#     # ---- Acronyms, onomatopoeia ----
#     u"Acroym for Laughter": (u"lol", 
#                     r"""(?<!\w)                             # Boundary
#                     (l(([oua]|aw)l)+([sz]?|wut)|            # lol, lawl, luls
#                     rot?fl(mf?ao)?)|                        # rofl, roflmao
#                     lmf?ao                                  # lmao, lmfao
#                     (?!\w)                                  # Boundary
#                     """
#     ),                                    
    u"Acronym for Grin": (u"*g*", 
                        r"""\*([Gg]{1,2}|                   # *g* *gg*
                        grin)\*                             # *grin*
                        """
    ),
#     u"Onomatopoeia for Laughter": (u"haha", 
#                         r"""(?<!\w)                         # Boundary
#                         (mu|ba)?                            # mu- ba-
#                         (ha|h(e|3)|hi){2,}                  # haha, hehe, hihi
#                         (?!\w)                              # Boundary
#                         """
#     ),
    u"Interjection": (u"ITJ", 
                        r"""(?<!\w)((a+h+a?)|               # ah, aha
                        (e+h+)|                             # eh
                        (u+g?h+)|                           # ugh
                        (huh)|                              # huh
                        ([uo]h( |-)h?[uo]h)|                # uh huh, 
                        (m*hm+)                             # hmm, mhm
                        |(h(u|r)?mp(h|f))|                  # hmpf
                        (ar+gh+)|                           # argh
                        (wow+))(?!\w)                       # wow
                        """
    ),
    u"GRINNING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x81", ur"""\U0001F601"""
    ),
    u"FACE WITH TEARS OF JOY": (u"\xF0\x9F\x98\x82", ur"""\U0001F602"""
    ),
    u"SMILING FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\x83", ur"""\U0001F603"""
    ),
    u"SMILING FACE WITH OPEN MOUTH AND SMILING EYES": (u"\xF0\x9F\x98\x84", ur"""\U0001F604"""
    ),
#     u"SMILING FACE WITH OPEN MOUTH AND COLD SWEAT": (u"\xF0\x9F\x98\x85", ur"""\U0001F605"""
#     ),
#     u"SMILING FACE WITH OPEN MOUTH AND TIGHTLY-CLOSED EYES": (u"\xF0\x9F\x98\x86", ur"""\U0001F606"""
#     ),
    u"WINKING FACE": (u"\xF0\x9F\x98\x89", ur"""\U0001F609"""
    ),
    u"SMILING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x8A", ur"""\U0001F60A"""
    ),
#     u"FACE SAVOURING DELICIOUS FOOD": (u"\xF0\x9F\x98\x8B", ur"""\U0001F60B"""
#     ),
#     u"RELIEVED FACE": (u"\xF0\x9F\x98\x8C", ur"""\U0001F60C"""
#     ),
    u"SMILING FACE WITH HEART-SHAPED EYES": (u"\xF0\x9F\x98\x8D", ur"""\U0001F60D"""
    ),
    u"SMIRKING FACE": (u"\xF0\x9F\x98\x8F", ur"""\U0001F60F"""
    ),
    u"UNAMUSED FACE": (u"\xF0\x9F\x98\x92", ur"""\U0001F612"""
    ),
#     u"FACE WITH COLD SWEAT": (u"\xF0\x9F\x98\x93", ur"""\U0001F613"""
#     ),
#     u"PENSIVE FACE": (u"\xF0\x9F\x98\x94", ur"""\U0001F614"""
#     ),
    u"CONFOUNDED FACE": (u"\xF0\x9F\x98\x96", ur"""\U0001F616"""
    ),
#     u"FACE THROWING A KISS": (u"\xF0\x9F\x98\x98", ur"""\U0001F618"""
#     ),
#     u"KISSING FACE WITH CLOSED EYES": (u"\xF0\x9F\x98\x9A", ur"""\U0001F61A"""
#     ),
    u"FACE WITH STUCK-OUT TONGUE AND WINKING EYE": (u"\xF0\x9F\x98\x9C", ur"""\U0001F61C"""
    ),
#     u"FACE WITH STUCK-OUT TONGUE AND TIGHTLY-CLOSED EYES": (u"\xF0\x9F\x98\x9D", ur"""\U0001F61D"""
#     ),
    u"DISAPPOINTED FACE": (u"\xF0\x9F\x98\x9E", ur"""\U0001F61E"""
    ),
#     u"ANGRY FACE": (u"\xF0\x9F\x98\xA0", ur"""\U0001F620"""
#     ),
#     u"POUTING FACE": (u"\xF0\x9F\x98\xA1", ur"""\U0001F621"""
#     ),
#     u"CRYING FACE": (u"\xF0\x9F\x98\xA2", ur"""\U0001F622"""
#     ),
#     u"PERSEVERING FACE": (u"\xF0\x9F\x98\xA3", ur"""\U0001F623"""
#     ),
    u"FACE WITH LOOK OF TRIUMPH": (u"\xF0\x9F\x98\xA4", ur"""\U0001F624"""
    ),
    u"DISAPPOINTED BUT RELIEVED FACE": (u"\xF0\x9F\x98\xA5", ur"""\U0001F625"""
    ),
#     u"FEARFUL FACE": (u"\xF0\x9F\x98\xA8", ur"""\U0001F628"""
#     ),
    u"WEARY FACE": (u"\xF0\x9F\x98\xA9", ur"""\U0001F629"""
    ),
#     u"SLEEPY FACE": (u"\xF0\x9F\x98\xAA", ur"""\U0001F62A"""
#     ),
#     u"TIRED FACE": (u"\xF0\x9F\x98\xAB", ur"""\U0001F62B"""
#     ),
    u"LOUDLY CRYING FACE": (u"\xF0\x9F\x98\xAD", ur"""\U0001F62D"""
    ),
#     u"FACE WITH OPEN MOUTH AND COLD SWEAT": (u"\xF0\x9F\x98\xB0", ur"""\U0001F630"""
#     ),
#     u"FACE SCREAMING IN FEAR": (u"\xF0\x9F\x98\xB1", ur"""\U0001F631"""
#     ),
#     u"ASTONISHED FACE": (u"\xF0\x9F\x98\xB2", ur"""\U0001F632"""
#     ),
    u"FLUSHED FACE": (u"\xF0\x9F\x98\xB3", ur"""\U0001F633"""
    ),
#     u"DIZZY FACE": (u"\xF0\x9F\x98\xB5", ur"""\U0001F635"""
#     ),
    u"FACE WITH MEDICAL MASK": (u"\xF0\x9F\x98\xB7", ur"""\U0001F637"""
    ),
#     u"GRINNING CAT FACE WITH SMILING EYES": (u"\xF0\x9F\x98\xB8", ur"""\U0001F638"""
#     ),
    u"CAT FACE WITH TEARS OF JOY": (u"\xF0\x9F\x98\xB9", ur"""\U0001F639"""
    ),
#     u"SMILING CAT FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xBA", ur"""\U0001F63A"""
#     ),
#     u"SMILING CAT FACE WITH HEART-SHAPED EYES": (u"\xF0\x9F\x98\xBB", ur"""\U0001F63B"""
#     ),
#     u"CAT FACE WITH WRY SMILE": (u"\xF0\x9F\x98\xBC", ur"""\U0001F63C"""
#     ),
#     u"KISSING CAT FACE WITH CLOSED EYES": (u"\xF0\x9F\x98\xBD", ur"""\U0001F63D"""
#     ),
#     u"POUTING CAT FACE": (u"\xF0\x9F\x98\xBE", ur"""\U0001F63E"""
#     ),
#     u"CRYING CAT FACE": (u"\xF0\x9F\x98\xBF", ur"""\U0001F63F"""
#     ),
#     u"WEARY CAT FACE": (u"\xF0\x9F\x99\x80", ur"""\U0001F640"""
#     ),
#     u"FACE WITH NO GOOD GESTURE": (u"\xF0\x9F\x99\x85", ur"""\U0001F645"""
#     ),
#     u"FACE WITH OK GESTURE": (u"\xF0\x9F\x99\x86", ur"""\U0001F646"""
#     ),
    u"PERSON BOWING DEEPLY": (u"\xF0\x9F\x99\x87", ur"""\U0001F647"""
    ),
#     u"SEE-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x88", ur"""\U0001F648"""
#     ),
#     u"HEAR-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x89", ur"""\U0001F649"""
#     ),
#     u"SPEAK-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x8A", ur"""\U0001F64A"""
#     ),
#     u"HAPPY PERSON RAISING ONE HAND": (u"\xF0\x9F\x99\x8B", ur"""\U0001F64B"""
#     ),
    u"PERSON RAISING BOTH HANDS IN CELEBRATION": (u"\xF0\x9F\x99\x8C", ur"""\U0001F64C"""
    ),
#     u"PERSON FROWNING": (u"\xF0\x9F\x99\x8D", ur"""\U0001F64D"""
#     ),
#     u"PERSON WITH POUTING FACE": (u"\xF0\x9F\x99\x8E", ur"""\U0001F64E"""
#     ),
    u"PERSON WITH FOLDED HANDS": (u"\xF0\x9F\x99\x8F", ur"""\U0001F64F"""
    ),
#     u"BLACK SCISSORS": (u"\xE2\x9C\x82", ur"""\U00002702"""
#     ),
#     u"WHITE HEAVY CHECK MARK": (u"\xE2\x9C\x85", ur"""\U00002705"""
#     ),
#     u"AIRPLANE": (u"\xE2\x9C\x88", ur"""\U00002708"""
#     ),
#     u"ENVELOPE": (u"\xE2\x9C\x89", ur"""\U00002709"""
#     ),
#     u"RAISED FIST": (u"\xE2\x9C\x8A", ur"""\U0000270A"""
#     ),
#     u"RAISED HAND": (u"\xE2\x9C\x8B", ur"""\U0000270B"""
#     ),
#     u"VICTORY HAND": (u"\xE2\x9C\x8C", ur"""\U0000270C"""
#     ),
#     u"PENCIL": (u"\xE2\x9C\x8F", ur"""\U0000270F"""
#     ),
#     u"BLACK NIB": (u"\xE2\x9C\x92", ur"""\U00002712"""
#     ),
#     u"HEAVY CHECK MARK": (u"\xE2\x9C\x94", ur"""\U00002714"""
#     ),
#     u"HEAVY MULTIPLICATION X": (u"\xE2\x9C\x96", ur"""\U00002716"""
#     ),
#     u"SPARKLES": (u"\xE2\x9C\xA8", ur"""\U00002728"""
#     ),
#     u"EIGHT SPOKED ASTERISK": (u"\xE2\x9C\xB3", ur"""\U00002733"""
#     ),
#     u"EIGHT POINTED BLACK STAR": (u"\xE2\x9C\xB4", ur"""\U00002734"""
#     ),
#     u"SNOWFLAKE": (u"\xE2\x9D\x84", ur"""\U00002744"""
#     ),
#     u"SPARKLE": (u"\xE2\x9D\x87", ur"""\U00002747"""
#     ),
#     u"CROSS MARK": (u"\xE2\x9D\x8C", ur"""\U0000274C"""
#     ),
#     u"NEGATIVE SQUARED CROSS MARK": (u"\xE2\x9D\x8E", ur"""\U0000274E"""
#     ),
#     u"BLACK QUESTION MARK ORNAMENT": (u"\xE2\x9D\x93", ur"""\U00002753"""
#     ),
#     u"WHITE QUESTION MARK ORNAMENT": (u"\xE2\x9D\x94", ur"""\U00002754"""
#     ),
#     u"WHITE EXCLAMATION MARK ORNAMENT": (u"\xE2\x9D\x95", ur"""\U00002755"""
#     ),
#     u"HEAVY EXCLAMATION MARK SYMBOL": (u"\xE2\x9D\x97", ur"""\U00002757"""
#     ),
    u"HEAVY BLACK HEART": (u"\xE2\x9D\xA4", ur"""\U00002764"""
    ),
#     u"HEAVY PLUS SIGN": (u"\xE2\x9E\x95", ur"""\U00002795"""
#     ),
#     u"HEAVY MINUS SIGN": (u"\xE2\x9E\x96", ur"""\U00002796"""
#     ),
#     u"HEAVY DIVISION SIGN": (u"\xE2\x9E\x97", ur"""\U00002797"""
#     ),
#     u"BLACK RIGHTWARDS ARROW": (u"\xE2\x9E\xA1", ur"""\U000027A1"""
#     ),
#     u"CURLY LOOP": (u"\xE2\x9E\xB0", ur"""\U000027B0"""
#     ),
#     u"ROCKET": (u"\xF0\x9F\x9A\x80", ur"""\U0001F680"""
#     ),
#     u"RAILWAY CAR": (u"\xF0\x9F\x9A\x83", ur"""\U0001F683"""
#     ),
#     u"HIGH-SPEED TRAIN": (u"\xF0\x9F\x9A\x84", ur"""\U0001F684"""
#     ),
#     u"HIGH-SPEED TRAIN WITH BULLET NOSE": (u"\xF0\x9F\x9A\x85", ur"""\U0001F685"""
#     ),
#     u"METRO": (u"\xF0\x9F\x9A\x87", ur"""\U0001F687"""
#     ),
#     u"STATION": (u"\xF0\x9F\x9A\x89", ur"""\U0001F689"""
#     ),
#     u"BUS": (u"\xF0\x9F\x9A\x8C", ur"""\U0001F68C"""
#     ),
#     u"BUS STOP": (u"\xF0\x9F\x9A\x8F", ur"""\U0001F68F"""
#     ),
#     u"AMBULANCE": (u"\xF0\x9F\x9A\x91", ur"""\U0001F691"""
#     ),
#     u"FIRE ENGINE": (u"\xF0\x9F\x9A\x92", ur"""\U0001F692"""
#     ),
#     u"POLICE CAR": (u"\xF0\x9F\x9A\x93", ur"""\U0001F693"""
#     ),
#     u"TAXI": (u"\xF0\x9F\x9A\x95", ur"""\U0001F695"""
#     ),
#     u"AUTOMOBILE": (u"\xF0\x9F\x9A\x97", ur"""\U0001F697"""
#     ),
    u"RECREATIONAL VEHICLE": (u"\xF0\x9F\x9A\x99", ur"""\U0001F699"""
    ),
#     u"DELIVERY TRUCK": (u"\xF0\x9F\x9A\x9A", ur"""\U0001F69A"""
#     ),
#     u"SHIP": (u"\xF0\x9F\x9A\xA2", ur"""\U0001F6A2"""
#     ),
#     u"SPEEDBOAT": (u"\xF0\x9F\x9A\xA4", ur"""\U0001F6A4"""
#     ),
#     u"HORIZONTAL TRAFFIC LIGHT": (u"\xF0\x9F\x9A\xA5", ur"""\U0001F6A5"""
#     ),
#     u"CONSTRUCTION SIGN": (u"\xF0\x9F\x9A\xA7", ur"""\U0001F6A7"""
#     ),
#     u"POLICE CARS REVOLVING LIGHT": (u"\xF0\x9F\x9A\xA8", ur"""\U0001F6A8"""
#     ),
#     u"TRIANGULAR FLAG ON POST": (u"\xF0\x9F\x9A\xA9", ur"""\U0001F6A9"""
#     ),
#     u"DOOR": (u"\xF0\x9F\x9A\xAA", ur"""\U0001F6AA"""
#     ),
#     u"NO ENTRY SIGN": (u"\xF0\x9F\x9A\xAB", ur"""\U0001F6AB"""
#     ),
#     u"SMOKING SYMBOL": (u"\xF0\x9F\x9A\xAC", ur"""\U0001F6AC"""
#     ),
#     u"NO SMOKING SYMBOL": (u"\xF0\x9F\x9A\xAD", ur"""\U0001F6AD"""
#     ),
#     u"BICYCLE": (u"\xF0\x9F\x9A\xB2", ur"""\U0001F6B2"""
#     ),
#     u"PEDESTRIAN": (u"\xF0\x9F\x9A\xB6", ur"""\U0001F6B6"""
#     ),
#     u"MENS SYMBOL": (u"\xF0\x9F\x9A\xB9", ur"""\U0001F6B9"""
#     ),
#     u"WOMENS SYMBOL": (u"\xF0\x9F\x9A\xBA", ur"""\U0001F6BA"""
#     ),
#     u"RESTROOM": (u"\xF0\x9F\x9A\xBB", ur"""\U0001F6BB"""
#     ),
#     u"BABY SYMBOL": (u"\xF0\x9F\x9A\xBC", ur"""\U0001F6BC"""
#     ),
#     u"TOILET": (u"\xF0\x9F\x9A\xBD", ur"""\U0001F6BD"""
#     ),
#     u"WATER CLOSET": (u"\xF0\x9F\x9A\xBE", ur"""\U0001F6BE"""
#     ),
#     u"BATH": (u"\xF0\x9F\x9B\x80", ur"""\U0001F6C0"""
#     ),
#     u"CIRCLED LATIN CAPITAL LETTER M": (u"\xE2\x93\x82", ur"""\U000024C2"""
#     ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER A": (u"\xF0\x9F\x85\xB0", ur"""\U0001F170"""
#     ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER B": (u"\xF0\x9F\x85\xB1", ur"""\U0001F171"""
#     ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER O": (u"\xF0\x9F\x85\xBE", ur"""\U0001F17E"""
#     ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER P": (u"\xF0\x9F\x85\xBF", ur"""\U0001F17F"""
#     ),
#     u"NEGATIVE SQUARED AB": (u"\xF0\x9F\x86\x8E", ur"""\U0001F18E"""
#     ),
#     u"SQUARED CL": (u"\xF0\x9F\x86\x91", ur"""\U0001F191"""
#     ),
#     u"SQUARED COOL": (u"\xF0\x9F\x86\x92", ur"""\U0001F192"""
#     ),
#     u"SQUARED FREE": (u"\xF0\x9F\x86\x93", ur"""\U0001F193"""
#     ),
#     u"SQUARED ID": (u"\xF0\x9F\x86\x94", ur"""\U0001F194"""
#     ),
#     u"SQUARED NEW": (u"\xF0\x9F\x86\x95", ur"""\U0001F195"""
#     ),
#     u"SQUARED NG": (u"\xF0\x9F\x86\x96", ur"""\U0001F196"""
#     ),
#     u"SQUARED OK": (u"\xF0\x9F\x86\x97", ur"""\U0001F197"""
#     ),
#     u"SQUARED SOS": (u"\xF0\x9F\x86\x98", ur"""\U0001F198"""
#     ),
#     u"SQUARED UP WITH EXCLAMATION MARK": (u"\xF0\x9F\x86\x99", ur"""\U0001F199"""
#     ),
#     u"SQUARED VS": (u"\xF0\x9F\x86\x9A", ur"""\U0001F19A"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER D + REGIONAL INDICATOR SYMBOL LETTER E": (u"\xF0\x9F\x87\xA9\xF0\x9F\x87\xAA", ur"""\U0001F1E9 \U0001F1EA"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER G + REGIONAL INDICATOR SYMBOL LETTER B": (u"\xF0\x9F\x87\xAC\xF0\x9F\x87\xA7", ur"""\U0001F1EC \U0001F1E7"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER C + REGIONAL INDICATOR SYMBOL LETTER N": (u"\xF0\x9F\x87\xA8\xF0\x9F\x87\xB3", ur"""\U0001F1E8 \U0001F1F3"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER J + REGIONAL INDICATOR SYMBOL LETTER P": (u"\xF0\x9F\x87\xAF\xF0\x9F\x87\xB5", ur"""\U0001F1EF \U0001F1F5"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER K + REGIONAL INDICATOR SYMBOL LETTER R": (u"\xF0\x9F\x87\xB0\xF0\x9F\x87\xB7", ur"""\U0001F1F0 \U0001F1F7"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER F + REGIONAL INDICATOR SYMBOL LETTER R": (u"\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7", ur"""\U0001F1EB \U0001F1F7"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER E + REGIONAL INDICATOR SYMBOL LETTER S": (u"\xF0\x9F\x87\xAA\xF0\x9F\x87\xB8", ur"""\U0001F1EA \U0001F1F8"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER I + REGIONAL INDICATOR SYMBOL LETTER T": (u"\xF0\x9F\x87\xAE\xF0\x9F\x87\xB9", ur"""\U0001F1EE \U0001F1F9"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER U + REGIONAL INDICATOR SYMBOL LETTER S": (u"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8", ur"""\U0001F1FA \U0001F1F8"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER R + REGIONAL INDICATOR SYMBOL LETTER U": (u"\xF0\x9F\x87\xB7\xF0\x9F\x87\xBA", ur"""\U0001F1F7 \U0001F1FA"""
#     ),
#     u"SQUARED KATAKANA KOKO": (u"\xF0\x9F\x88\x81", ur"""\U0001F201"""
#     ),
#     u"SQUARED KATAKANA SA": (u"\xF0\x9F\x88\x82", ur"""\U0001F202"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7121": (u"\xF0\x9F\x88\x9A", ur"""\U0001F21A"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6307": (u"\xF0\x9F\x88\xAF", ur"""\U0001F22F"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7981": (u"\xF0\x9F\x88\xB2", ur"""\U0001F232"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7A7A": (u"\xF0\x9F\x88\xB3", ur"""\U0001F233"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-5408": (u"\xF0\x9F\x88\xB4", ur"""\U0001F234"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6E80": (u"\xF0\x9F\x88\xB5", ur"""\U0001F235"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6709": (u"\xF0\x9F\x88\xB6", ur"""\U0001F236"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6708": (u"\xF0\x9F\x88\xB7", ur"""\U0001F237"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7533": (u"\xF0\x9F\x88\xB8", ur"""\U0001F238"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-5272": (u"\xF0\x9F\x88\xB9", ur"""\U0001F239"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-55B6": (u"\xF0\x9F\x88\xBA", ur"""\U0001F23A"""
#     ),
#     u"CIRCLED IDEOGRAPH ADVANTAGE": (u"\xF0\x9F\x89\x90", ur"""\U0001F250"""
#     ),
#     u"CIRCLED IDEOGRAPH ACCEPT": (u"\xF0\x9F\x89\x91", ur"""\U0001F251"""
#     ),
    u"COPYRIGHT SIGN": (u"\xC2\xA9", ur"""\U000000A9"""
    ),
#     u"REGISTERED SIGN": (u"\xC2\xAE", ur"""\U000000AE"""
#     ),
#     u"DOUBLE EXCLAMATION MARK": (u"\xE2\x80\xBC", ur"""\U0000203C"""
#     ),
#     u"EXCLAMATION QUESTION MARK": (u"\xE2\x81\x89", ur"""\U00002049"""
#     ),
#     u"DIGIT EIGHT + COMBINING ENCLOSING KEYCAP": (u"\x38\xE2\x83\xA3", ur"""\U00000038 \U000020E3"""
#     ),
#     u"DIGIT NINE + COMBINING ENCLOSING KEYCAP": (u"\x39\xE2\x83\xA3", ur"""\U00000039 \U000020E3"""
#     ),
#     u"DIGIT SEVEN + COMBINING ENCLOSING KEYCAP": (u"\x37\xE2\x83\xA3", ur"""\U00000037 \U000020E3"""
#     ),
#     u"DIGIT SIX + COMBINING ENCLOSING KEYCAP": (u"\x36\xE2\x83\xA3", ur"""\U00000036 \U000020E3"""
#     ),
#     u"DIGIT ONE + COMBINING ENCLOSING KEYCAP": (u"\x31\xE2\x83\xA3", ur"""\U00000031 \U000020E3"""
#     ),
#     u"DIGIT ZERO + COMBINING ENCLOSING KEYCAP": (u"\x30\xE2\x83\xA3", ur"""\U00000030 \U000020E3"""
#     ),
#     u"DIGIT TWO + COMBINING ENCLOSING KEYCAP": (u"\x32\xE2\x83\xA3", ur"""\U00000032 \U000020E3"""
#     ),
#     u"DIGIT THREE + COMBINING ENCLOSING KEYCAP": (u"\x33\xE2\x83\xA3", ur"""\U00000033 \U000020E3"""
#     ),
#     u"DIGIT FIVE + COMBINING ENCLOSING KEYCAP": (u"\x35\xE2\x83\xA3", ur"""\U00000035 \U000020E3"""
#     ),
#     u"DIGIT FOUR + COMBINING ENCLOSING KEYCAP": (u"\x34\xE2\x83\xA3", ur"""\U00000034 \U000020E3"""
#     ),
#     u"NUMBER SIGN + COMBINING ENCLOSING KEYCAP": (u"\x23\xE2\x83\xA3", ur"""\U00000023 \U000020E3"""
#     ),
#     u"TRADE MARK SIGN": (u"\xE2\x84\xA2", ur"""\U00002122"""
#     ),
#     u"INFORMATION SOURCE": (u"\xE2\x84\xB9", ur"""\U00002139"""
#     ),
#     u"LEFT RIGHT ARROW": (u"\xE2\x86\x94", ur"""\U00002194"""
#     ),
#     u"UP DOWN ARROW": (u"\xE2\x86\x95", ur"""\U00002195"""
#     ),
#     u"NORTH WEST ARROW": (u"\xE2\x86\x96", ur"""\U00002196"""
#     ),
#     u"NORTH EAST ARROW": (u"\xE2\x86\x97", ur"""\U00002197"""
#     ),
#     u"SOUTH EAST ARROW": (u"\xE2\x86\x98", ur"""\U00002198"""
#     ),
#     u"SOUTH WEST ARROW": (u"\xE2\x86\x99", ur"""\U00002199"""
#     ),
#     u"LEFTWARDS ARROW WITH HOOK": (u"\xE2\x86\xA9", ur"""\U000021A9"""
#     ),
#     u"RIGHTWARDS ARROW WITH HOOK": (u"\xE2\x86\xAA", ur"""\U000021AA"""
#     ),
#     u"WATCH": (u"\xE2\x8C\x9A", ur"""\U0000231A"""
#     ),
#     u"HOURGLASS": (u"\xE2\x8C\x9B", ur"""\U0000231B"""
#     ),
#     u"BLACK RIGHT-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xA9", ur"""\U000023E9"""
#     ),
#     u"BLACK LEFT-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAA", ur"""\U000023EA"""
#     ),
#     u"BLACK UP-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAB", ur"""\U000023EB"""
#     ),
#     u"BLACK DOWN-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAC", ur"""\U000023EC"""
#     ),
#     u"ALARM CLOCK": (u"\xE2\x8F\xB0", ur"""\U000023F0"""
#     ),
#     u"HOURGLASS WITH FLOWING SAND": (u"\xE2\x8F\xB3", ur"""\U000023F3"""
#     ),
#     u"BLACK SMALL SQUARE": (u"\xE2\x96\xAA", ur"""\U000025AA"""
#     ),
#     u"WHITE SMALL SQUARE": (u"\xE2\x96\xAB", ur"""\U000025AB"""
#     ),
#     u"BLACK RIGHT-POINTING TRIANGLE": (u"\xE2\x96\xB6", ur"""\U000025B6"""
#     ),
#     u"BLACK LEFT-POINTING TRIANGLE": (u"\xE2\x97\x80", ur"""\U000025C0"""
#     ),
#     u"WHITE MEDIUM SQUARE": (u"\xE2\x97\xBB", ur"""\U000025FB"""
#     ),
#     u"BLACK MEDIUM SQUARE": (u"\xE2\x97\xBC", ur"""\U000025FC"""
#     ),
#     u"WHITE MEDIUM SMALL SQUARE": (u"\xE2\x97\xBD", ur"""\U000025FD"""
#     ),
#     u"BLACK MEDIUM SMALL SQUARE": (u"\xE2\x97\xBE", ur"""\U000025FE"""
#     ),
#     u"BLACK SUN WITH RAYS": (u"\xE2\x98\x80", ur"""\U00002600"""
#     ),
#     u"CLOUD": (u"\xE2\x98\x81", ur"""\U00002601"""
#     ),
#     u"BLACK TELEPHONE": (u"\xE2\x98\x8E", ur"""\U0000260E"""
#     ),
#     u"BALLOT BOX WITH CHECK": (u"\xE2\x98\x91", ur"""\U00002611"""
#     ),
    u"UMBRELLA WITH RAIN DROPS": (u"\xE2\x98\x94", ur"""\U00002614"""
    ),
#     u"HOT BEVERAGE": (u"\xE2\x98\x95", ur"""\U00002615"""
#     ),
#     u"WHITE UP POINTING INDEX": (u"\xE2\x98\x9D", ur"""\U0000261D"""
#     ),
#     u"WHITE SMILING FACE": (u"\xE2\x98\xBA", ur"""\U0000263A"""
#     ),
#     u"ARIES": (u"\xE2\x99\x88", ur"""\U00002648"""
#     ),
#     u"TAURUS": (u"\xE2\x99\x89", ur"""\U00002649"""
#     ),
#     u"GEMINI": (u"\xE2\x99\x8A", ur"""\U0000264A"""
#     ),
#     u"CANCER": (u"\xE2\x99\x8B", ur"""\U0000264B"""
#     ),
#     u"LEO": (u"\xE2\x99\x8C", ur"""\U0000264C"""
#     ),
#     u"VIRGO": (u"\xE2\x99\x8D", ur"""\U0000264D"""
#     ),
#     u"LIBRA": (u"\xE2\x99\x8E", ur"""\U0000264E"""
#     ),
#     u"SCORPIUS": (u"\xE2\x99\x8F", ur"""\U0000264F"""
#     ),
#     u"SAGITTARIUS": (u"\xE2\x99\x90", ur"""\U00002650"""
#     ),
#     u"CAPRICORN": (u"\xE2\x99\x91", ur"""\U00002651"""
#     ),
#     u"AQUARIUS": (u"\xE2\x99\x92", ur"""\U00002652"""
#     ),
#     u"PISCES": (u"\xE2\x99\x93", ur"""\U00002653"""
#     ),
#     u"BLACK SPADE SUIT": (u"\xE2\x99\xA0", ur"""\U00002660"""
#     ),
#     u"BLACK CLUB SUIT": (u"\xE2\x99\xA3", ur"""\U00002663"""
#     ),
    u"BLACK HEART SUIT": (u"\xE2\x99\xA5", ur"""\U00002665"""
    ),
#     u"BLACK DIAMOND SUIT": (u"\xE2\x99\xA6", ur"""\U00002666"""
#     ),
#     u"HOT SPRINGS": (u"\xE2\x99\xA8", ur"""\U00002668"""
#     ),
#     u"BLACK UNIVERSAL RECYCLING SYMBOL": (u"\xE2\x99\xBB", ur"""\U0000267B"""
#     ),
#     u"WHEELCHAIR SYMBOL": (u"\xE2\x99\xBF", ur"""\U0000267F"""
#     ),
#     u"ANCHOR": (u"\xE2\x9A\x93", ur"""\U00002693"""
#     ),
#     u"WARNING SIGN": (u"\xE2\x9A\xA0", ur"""\U000026A0"""
#     ),
#     u"HIGH VOLTAGE SIGN": (u"\xE2\x9A\xA1", ur"""\U000026A1"""
#     ),
#     u"MEDIUM WHITE CIRCLE": (u"\xE2\x9A\xAA", ur"""\U000026AA"""
#     ),
#     u"MEDIUM BLACK CIRCLE": (u"\xE2\x9A\xAB", ur"""\U000026AB"""
#     ),
#     u"SOCCER BALL": (u"\xE2\x9A\xBD", ur"""\U000026BD"""
#     ),
    u"BASEBALL": (u"\xE2\x9A\xBE", ur"""\U000026BE"""
    ),
#     u"SNOWMAN WITHOUT SNOW": (u"\xE2\x9B\x84", ur"""\U000026C4"""
#     ),
#     u"SUN BEHIND CLOUD": (u"\xE2\x9B\x85", ur"""\U000026C5"""
#     ),
#     u"OPHIUCHUS": (u"\xE2\x9B\x8E", ur"""\U000026CE"""
#     ),
#     u"NO ENTRY": (u"\xE2\x9B\x94", ur"""\U000026D4"""
#     ),
#     u"CHURCH": (u"\xE2\x9B\xAA", ur"""\U000026EA"""
#     ),
#     u"FOUNTAIN": (u"\xE2\x9B\xB2", ur"""\U000026F2"""
#     ),
#     u"FLAG IN HOLE": (u"\xE2\x9B\xB3", ur"""\U000026F3"""
#     ),
#     u"SAILBOAT": (u"\xE2\x9B\xB5", ur"""\U000026F5"""
#     ),
#     u"TENT": (u"\xE2\x9B\xBA", ur"""\U000026FA"""
#     ),
#     u"FUEL PUMP": (u"\xE2\x9B\xBD", ur"""\U000026FD"""
#     ),
#     u"ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS": (u"\xE2\xA4\xB4", ur"""\U00002934"""
#     ),
#     u"ARROW POINTING RIGHTWARDS THEN CURVING DOWNWARDS": (u"\xE2\xA4\xB5", ur"""\U00002935"""
#     ),
#     u"LEFTWARDS BLACK ARROW": (u"\xE2\xAC\x85", ur"""\U00002B05"""
#     ),
#     u"UPWARDS BLACK ARROW": (u"\xE2\xAC\x86", ur"""\U00002B06"""
#     ),
#     u"DOWNWARDS BLACK ARROW": (u"\xE2\xAC\x87", ur"""\U00002B07"""
#     ),
#     u"BLACK LARGE SQUARE": (u"\xE2\xAC\x9B", ur"""\U00002B1B"""
#     ),
#     u"WHITE LARGE SQUARE": (u"\xE2\xAC\x9C", ur"""\U00002B1C"""
#     ),
#     u"WHITE MEDIUM STAR": (u"\xE2\xAD\x90", ur"""\U00002B50"""
#     ),
#     u"HEAVY LARGE CIRCLE": (u"\xE2\xAD\x95", ur"""\U00002B55"""
#     ),
#     u"WAVY DASH": (u"\xE3\x80\xB0", ur"""\U00003030"""
#     ),
#     u"PART ALTERNATION MARK": (u"\xE3\x80\xBD", ur"""\U0000303D"""
#     ),
#     u"CIRCLED IDEOGRAPH CONGRATULATION": (u"\xE3\x8A\x97", ur"""\U00003297"""
#     ),
#     u"CIRCLED IDEOGRAPH SECRET": (u"\xE3\x8A\x99", ur"""\U00003299"""
#     ),
#     u"MAHJONG TILE RED DRAGON": (u"\xF0\x9F\x80\x84", ur"""\U0001F004"""
#     ),
#     u"PLAYING CARD BLACK JOKER": (u"\xF0\x9F\x83\x8F", ur"""\U0001F0CF"""
#     ),
#     u"CYCLONE": (u"\xF0\x9F\x8C\x80", ur"""\U0001F300"""
#     ),
#     u"FOGGY": (u"\xF0\x9F\x8C\x81", ur"""\U0001F301"""
#     ),
#     u"CLOSED UMBRELLA": (u"\xF0\x9F\x8C\x82", ur"""\U0001F302"""
#     ),
#     u"NIGHT WITH STARS": (u"\xF0\x9F\x8C\x83", ur"""\U0001F303"""
#     ),
#     u"SUNRISE OVER MOUNTAINS": (u"\xF0\x9F\x8C\x84", ur"""\U0001F304"""
#     ),
#     u"SUNRISE": (u"\xF0\x9F\x8C\x85", ur"""\U0001F305"""
#     ),
#     u"CITYSCAPE AT DUSK": (u"\xF0\x9F\x8C\x86", ur"""\U0001F306"""
#     ),
#     u"SUNSET OVER BUILDINGS": (u"\xF0\x9F\x8C\x87", ur"""\U0001F307"""
#     ),
#     u"RAINBOW": (u"\xF0\x9F\x8C\x88", ur"""\U0001F308"""
#     ),
#     u"BRIDGE AT NIGHT": (u"\xF0\x9F\x8C\x89", ur"""\U0001F309"""
#     ),
#     u"WATER WAVE": (u"\xF0\x9F\x8C\x8A", ur"""\U0001F30A"""
#     ),
#     u"VOLCANO": (u"\xF0\x9F\x8C\x8B", ur"""\U0001F30B"""
#     ),
#     u"MILKY WAY": (u"\xF0\x9F\x8C\x8C", ur"""\U0001F30C"""
#     ),
#     u"EARTH GLOBE ASIA-AUSTRALIA": (u"\xF0\x9F\x8C\x8F", ur"""\U0001F30F"""
#     ),
#     u"NEW MOON SYMBOL": (u"\xF0\x9F\x8C\x91", ur"""\U0001F311"""
#     ),
#     u"FIRST QUARTER MOON SYMBOL": (u"\xF0\x9F\x8C\x93", ur"""\U0001F313"""
#     ),
#     u"WAXING GIBBOUS MOON SYMBOL": (u"\xF0\x9F\x8C\x94", ur"""\U0001F314"""
#     ),
#     u"FULL MOON SYMBOL": (u"\xF0\x9F\x8C\x95", ur"""\U0001F315"""
#     ),
#     u"CRESCENT MOON": (u"\xF0\x9F\x8C\x99", ur"""\U0001F319"""
#     ),
#     u"FIRST QUARTER MOON WITH FACE": (u"\xF0\x9F\x8C\x9B", ur"""\U0001F31B"""
#     ),
#     u"GLOWING STAR": (u"\xF0\x9F\x8C\x9F", ur"""\U0001F31F"""
#     ),
#     u"SHOOTING STAR": (u"\xF0\x9F\x8C\xA0", ur"""\U0001F320"""
#     ),
#     u"CHESTNUT": (u"\xF0\x9F\x8C\xB0", ur"""\U0001F330"""
#     ),
#     u"SEEDLING": (u"\xF0\x9F\x8C\xB1", ur"""\U0001F331"""
#     ),
#     u"PALM TREE": (u"\xF0\x9F\x8C\xB4", ur"""\U0001F334"""
#     ),
#     u"CACTUS": (u"\xF0\x9F\x8C\xB5", ur"""\U0001F335"""
#     ),
#     u"TULIP": (u"\xF0\x9F\x8C\xB7", ur"""\U0001F337"""
#     ),
#     u"CHERRY BLOSSOM": (u"\xF0\x9F\x8C\xB8", ur"""\U0001F338"""
#     ),
#     u"ROSE": (u"\xF0\x9F\x8C\xB9", ur"""\U0001F339"""
#     ),
    u"HIBISCUS": (u"\xF0\x9F\x8C\xBA", ur"""\U0001F33A"""
    ),
#     u"SUNFLOWER": (u"\xF0\x9F\x8C\xBB", ur"""\U0001F33B"""
#     ),
#     u"BLOSSOM": (u"\xF0\x9F\x8C\xBC", ur"""\U0001F33C"""
#     ),
#     u"EAR OF MAIZE": (u"\xF0\x9F\x8C\xBD", ur"""\U0001F33D"""
#     ),
#     u"EAR OF RICE": (u"\xF0\x9F\x8C\xBE", ur"""\U0001F33E"""
#     ),
#     u"HERB": (u"\xF0\x9F\x8C\xBF", ur"""\U0001F33F"""
#     ),
#     u"FOUR LEAF CLOVER": (u"\xF0\x9F\x8D\x80", ur"""\U0001F340"""
#     ),
#     u"MAPLE LEAF": (u"\xF0\x9F\x8D\x81", ur"""\U0001F341"""
#     ),
#     u"FALLEN LEAF": (u"\xF0\x9F\x8D\x82", ur"""\U0001F342"""
#     ),
#     u"LEAF FLUTTERING IN WIND": (u"\xF0\x9F\x8D\x83", ur"""\U0001F343"""
#     ),
#     u"MUSHROOM": (u"\xF0\x9F\x8D\x84", ur"""\U0001F344"""
#     ),
#     u"TOMATO": (u"\xF0\x9F\x8D\x85", ur"""\U0001F345"""
#     ),
#     u"AUBERGINE": (u"\xF0\x9F\x8D\x86", ur"""\U0001F346"""
#     ),
#     u"GRAPES": (u"\xF0\x9F\x8D\x87", ur"""\U0001F347"""
#     ),
#     u"MELON": (u"\xF0\x9F\x8D\x88", ur"""\U0001F348"""
#     ),
#     u"WATERMELON": (u"\xF0\x9F\x8D\x89", ur"""\U0001F349"""
#     ),
#     u"TANGERINE": (u"\xF0\x9F\x8D\x8A", ur"""\U0001F34A"""
#     ),
#     u"BANANA": (u"\xF0\x9F\x8D\x8C", ur"""\U0001F34C"""
#     ),
#     u"PINEAPPLE": (u"\xF0\x9F\x8D\x8D", ur"""\U0001F34D"""
#     ),
#     u"RED APPLE": (u"\xF0\x9F\x8D\x8E", ur"""\U0001F34E"""
#     ),
#     u"GREEN APPLE": (u"\xF0\x9F\x8D\x8F", ur"""\U0001F34F"""
#     ),
#     u"PEACH": (u"\xF0\x9F\x8D\x91", ur"""\U0001F351"""
#     ),
#     u"CHERRIES": (u"\xF0\x9F\x8D\x92", ur"""\U0001F352"""
#     ),
#     u"STRAWBERRY": (u"\xF0\x9F\x8D\x93", ur"""\U0001F353"""
#     ),
#     u"HAMBURGER": (u"\xF0\x9F\x8D\x94", ur"""\U0001F354"""
#     ),
#     u"SLICE OF PIZZA": (u"\xF0\x9F\x8D\x95", ur"""\U0001F355"""
#     ),
#     u"MEAT ON BONE": (u"\xF0\x9F\x8D\x96", ur"""\U0001F356"""
#     ),
#     u"POULTRY LEG": (u"\xF0\x9F\x8D\x97", ur"""\U0001F357"""
#     ),
#     u"RICE CRACKER": (u"\xF0\x9F\x8D\x98", ur"""\U0001F358"""
#     ),
#     u"RICE BALL": (u"\xF0\x9F\x8D\x99", ur"""\U0001F359"""
#     ),
#     u"COOKED RICE": (u"\xF0\x9F\x8D\x9A", ur"""\U0001F35A"""
#     ),
#     u"CURRY AND RICE": (u"\xF0\x9F\x8D\x9B", ur"""\U0001F35B"""
#     ),
#     u"STEAMING BOWL": (u"\xF0\x9F\x8D\x9C", ur"""\U0001F35C"""
#     ),
#     u"SPAGHETTI": (u"\xF0\x9F\x8D\x9D", ur"""\U0001F35D"""
#     ),
#     u"BREAD": (u"\xF0\x9F\x8D\x9E", ur"""\U0001F35E"""
#     ),
#     u"FRENCH FRIES": (u"\xF0\x9F\x8D\x9F", ur"""\U0001F35F"""
#     ),
#     u"ROASTED SWEET POTATO": (u"\xF0\x9F\x8D\xA0", ur"""\U0001F360"""
#     ),
#     u"DANGO": (u"\xF0\x9F\x8D\xA1", ur"""\U0001F361"""
#     ),
#     u"ODEN": (u"\xF0\x9F\x8D\xA2", ur"""\U0001F362"""
#     ),
#     u"SUSHI": (u"\xF0\x9F\x8D\xA3", ur"""\U0001F363"""
#     ),
#     u"FRIED SHRIMP": (u"\xF0\x9F\x8D\xA4", ur"""\U0001F364"""
#     ),
#     u"FISH CAKE WITH SWIRL DESIGN": (u"\xF0\x9F\x8D\xA5", ur"""\U0001F365"""
#     ),
#     u"SOFT ICE CREAM": (u"\xF0\x9F\x8D\xA6", ur"""\U0001F366"""
#     ),
#     u"SHAVED ICE": (u"\xF0\x9F\x8D\xA7", ur"""\U0001F367"""
#     ),
#     u"ICE CREAM": (u"\xF0\x9F\x8D\xA8", ur"""\U0001F368"""
#     ),
#     u"DOUGHNUT": (u"\xF0\x9F\x8D\xA9", ur"""\U0001F369"""
#     ),
#     u"COOKIE": (u"\xF0\x9F\x8D\xAA", ur"""\U0001F36A"""
#     ),
#     u"CHOCOLATE BAR": (u"\xF0\x9F\x8D\xAB", ur"""\U0001F36B"""
#     ),
#     u"CANDY": (u"\xF0\x9F\x8D\xAC", ur"""\U0001F36C"""
#     ),
#     u"LOLLIPOP": (u"\xF0\x9F\x8D\xAD", ur"""\U0001F36D"""
#     ),
#     u"CUSTARD": (u"\xF0\x9F\x8D\xAE", ur"""\U0001F36E"""
#     ),
#     u"HONEY POT": (u"\xF0\x9F\x8D\xAF", ur"""\U0001F36F"""
#     ),
#     u"SHORTCAKE": (u"\xF0\x9F\x8D\xB0", ur"""\U0001F370"""
#     ),
#     u"BENTO BOX": (u"\xF0\x9F\x8D\xB1", ur"""\U0001F371"""
#     ),
#     u"POT OF FOOD": (u"\xF0\x9F\x8D\xB2", ur"""\U0001F372"""
#     ),
#     u"COOKING": (u"\xF0\x9F\x8D\xB3", ur"""\U0001F373"""
#     ),
#     u"FORK AND KNIFE": (u"\xF0\x9F\x8D\xB4", ur"""\U0001F374"""
#     ),
#     u"TEACUP WITHOUT HANDLE": (u"\xF0\x9F\x8D\xB5", ur"""\U0001F375"""
#     ),
#     u"SAKE BOTTLE AND CUP": (u"\xF0\x9F\x8D\xB6", ur"""\U0001F376"""
#     ),
#     u"WINE GLASS": (u"\xF0\x9F\x8D\xB7", ur"""\U0001F377"""
#     ),
#     u"COCKTAIL GLASS": (u"\xF0\x9F\x8D\xB8", ur"""\U0001F378"""
#     ),
#     u"TROPICAL DRINK": (u"\xF0\x9F\x8D\xB9", ur"""\U0001F379"""
#     ),
#     u"BEER MUG": (u"\xF0\x9F\x8D\xBA", ur"""\U0001F37A"""
#     ),
#     u"CLINKING BEER MUGS": (u"\xF0\x9F\x8D\xBB", ur"""\U0001F37B"""
#     ),
#     u"RIBBON": (u"\xF0\x9F\x8E\x80", ur"""\U0001F380"""
#     ),
#     u"WRAPPED PRESENT": (u"\xF0\x9F\x8E\x81", ur"""\U0001F381"""
#     ),
#     u"BIRTHDAY CAKE": (u"\xF0\x9F\x8E\x82", ur"""\U0001F382"""
#     ),
#     u"JACK-O-LANTERN": (u"\xF0\x9F\x8E\x83", ur"""\U0001F383"""
#     ),
#     u"CHRISTMAS TREE": (u"\xF0\x9F\x8E\x84", ur"""\U0001F384"""
#     ),
#     u"FATHER CHRISTMAS": (u"\xF0\x9F\x8E\x85", ur"""\U0001F385"""
#     ),
#     u"FIREWORKS": (u"\xF0\x9F\x8E\x86", ur"""\U0001F386"""
#     ),
#     u"FIREWORK SPARKLER": (u"\xF0\x9F\x8E\x87", ur"""\U0001F387"""
#     ),
#     u"BALLOON": (u"\xF0\x9F\x8E\x88", ur"""\U0001F388"""
#     ),
    u"PARTY POPPER": (u"\xF0\x9F\x8E\x89", ur"""\U0001F389"""
    ),
#     u"CONFETTI BALL": (u"\xF0\x9F\x8E\x8A", ur"""\U0001F38A"""
#     ),
#     u"TANABATA TREE": (u"\xF0\x9F\x8E\x8B", ur"""\U0001F38B"""
#     ),
#     u"CROSSED FLAGS": (u"\xF0\x9F\x8E\x8C", ur"""\U0001F38C"""
#     ),
#     u"PINE DECORATION": (u"\xF0\x9F\x8E\x8D", ur"""\U0001F38D"""
#     ),
#     u"JAPANESE DOLLS": (u"\xF0\x9F\x8E\x8E", ur"""\U0001F38E"""
#     ),
#     u"CARP STREAMER": (u"\xF0\x9F\x8E\x8F", ur"""\U0001F38F"""
#     ),
#     u"WIND CHIME": (u"\xF0\x9F\x8E\x90", ur"""\U0001F390"""
#     ),
#     u"MOON VIEWING CEREMONY": (u"\xF0\x9F\x8E\x91", ur"""\U0001F391"""
#     ),
#     u"SCHOOL SATCHEL": (u"\xF0\x9F\x8E\x92", ur"""\U0001F392"""
#     ),
#     u"GRADUATION CAP": (u"\xF0\x9F\x8E\x93", ur"""\U0001F393"""
#     ),
#     u"CAROUSEL HORSE": (u"\xF0\x9F\x8E\xA0", ur"""\U0001F3A0"""
#     ),
#     u"FERRIS WHEEL": (u"\xF0\x9F\x8E\xA1", ur"""\U0001F3A1"""
#     ),
#     u"ROLLER COASTER": (u"\xF0\x9F\x8E\xA2", ur"""\U0001F3A2"""
#     ),
#     u"FISHING POLE AND FISH": (u"\xF0\x9F\x8E\xA3", ur"""\U0001F3A3"""
#     ),
#     u"MICROPHONE": (u"\xF0\x9F\x8E\xA4", ur"""\U0001F3A4"""
#     ),
#     u"MOVIE CAMERA": (u"\xF0\x9F\x8E\xA5", ur"""\U0001F3A5"""
#     ),
#     u"CINEMA": (u"\xF0\x9F\x8E\xA6", ur"""\U0001F3A6"""
#     ),
#     u"HEADPHONE": (u"\xF0\x9F\x8E\xA7", ur"""\U0001F3A7"""
#     ),
#     u"ARTIST PALETTE": (u"\xF0\x9F\x8E\xA8", ur"""\U0001F3A8"""
#     ),
#     u"TOP HAT": (u"\xF0\x9F\x8E\xA9", ur"""\U0001F3A9"""
#     ),
#     u"CIRCUS TENT": (u"\xF0\x9F\x8E\xAA", ur"""\U0001F3AA"""
#     ),
#     u"TICKET": (u"\xF0\x9F\x8E\xAB", ur"""\U0001F3AB"""
#     ),
#     u"CLAPPER BOARD": (u"\xF0\x9F\x8E\xAC", ur"""\U0001F3AC"""
#     ),
#     u"PERFORMING ARTS": (u"\xF0\x9F\x8E\xAD", ur"""\U0001F3AD"""
#     ),
#     u"VIDEO GAME": (u"\xF0\x9F\x8E\xAE", ur"""\U0001F3AE"""
#     ),
#     u"DIRECT HIT": (u"\xF0\x9F\x8E\xAF", ur"""\U0001F3AF"""
#     ),
#     u"SLOT MACHINE": (u"\xF0\x9F\x8E\xB0", ur"""\U0001F3B0"""
#     ),
#     u"BILLIARDS": (u"\xF0\x9F\x8E\xB1", ur"""\U0001F3B1"""
#     ),
#     u"GAME DIE": (u"\xF0\x9F\x8E\xB2", ur"""\U0001F3B2"""
#     ),
#     u"BOWLING": (u"\xF0\x9F\x8E\xB3", ur"""\U0001F3B3"""
#     ),
#     u"FLOWER PLAYING CARDS": (u"\xF0\x9F\x8E\xB4", ur"""\U0001F3B4"""
#     ),
#     u"MUSICAL NOTE": (u"\xF0\x9F\x8E\xB5", ur"""\U0001F3B5"""
#     ),
    u"MULTIPLE MUSICAL NOTES": (u"\xF0\x9F\x8E\xB6", ur"""\U0001F3B6"""
    ),
#     u"SAXOPHONE": (u"\xF0\x9F\x8E\xB7", ur"""\U0001F3B7"""
#     ),
#     u"GUITAR": (u"\xF0\x9F\x8E\xB8", ur"""\U0001F3B8"""
#     ),
#     u"MUSICAL KEYBOARD": (u"\xF0\x9F\x8E\xB9", ur"""\U0001F3B9"""
#     ),
#     u"TRUMPET": (u"\xF0\x9F\x8E\xBA", ur"""\U0001F3BA"""
#     ),
#     u"VIOLIN": (u"\xF0\x9F\x8E\xBB", ur"""\U0001F3BB"""
#     ),
#     u"MUSICAL SCORE": (u"\xF0\x9F\x8E\xBC", ur"""\U0001F3BC"""
#     ),
#     u"RUNNING SHIRT WITH SASH": (u"\xF0\x9F\x8E\xBD", ur"""\U0001F3BD"""
#     ),
#     u"TENNIS RACQUET AND BALL": (u"\xF0\x9F\x8E\xBE", ur"""\U0001F3BE"""
#     ),
#     u"SKI AND SKI BOOT": (u"\xF0\x9F\x8E\xBF", ur"""\U0001F3BF"""
#     ),
#     u"BASKETBALL AND HOOP": (u"\xF0\x9F\x8F\x80", ur"""\U0001F3C0"""
#     ),
#     u"CHEQUERED FLAG": (u"\xF0\x9F\x8F\x81", ur"""\U0001F3C1"""
#     ),
#     u"SNOWBOARDER": (u"\xF0\x9F\x8F\x82", ur"""\U0001F3C2"""
#     ),
    u"RUNNER": (u"\xF0\x9F\x8F\x83", ur"""\U0001F3C3"""
    ),
#     u"SURFER": (u"\xF0\x9F\x8F\x84", ur"""\U0001F3C4"""
#     ),
#     u"TROPHY": (u"\xF0\x9F\x8F\x86", ur"""\U0001F3C6"""
#     ),
#     u"AMERICAN FOOTBALL": (u"\xF0\x9F\x8F\x88", ur"""\U0001F3C8"""
#     ),
#     u"SWIMMER": (u"\xF0\x9F\x8F\x8A", ur"""\U0001F3CA"""
#     ),
#     u"HOUSE BUILDING": (u"\xF0\x9F\x8F\xA0", ur"""\U0001F3E0"""
#     ),
#     u"HOUSE WITH GARDEN": (u"\xF0\x9F\x8F\xA1", ur"""\U0001F3E1"""
#     ),
#     u"OFFICE BUILDING": (u"\xF0\x9F\x8F\xA2", ur"""\U0001F3E2"""
#     ),
#     u"JAPANESE POST OFFICE": (u"\xF0\x9F\x8F\xA3", ur"""\U0001F3E3"""
#     ),
#     u"HOSPITAL": (u"\xF0\x9F\x8F\xA5", ur"""\U0001F3E5"""
#     ),
#     u"BANK": (u"\xF0\x9F\x8F\xA6", ur"""\U0001F3E6"""
#     ),
#     u"AUTOMATED TELLER MACHINE": (u"\xF0\x9F\x8F\xA7", ur"""\U0001F3E7"""
#     ),
#     u"HOTEL": (u"\xF0\x9F\x8F\xA8", ur"""\U0001F3E8"""
#     ),
#     u"LOVE HOTEL": (u"\xF0\x9F\x8F\xA9", ur"""\U0001F3E9"""
#     ),
#     u"CONVENIENCE STORE": (u"\xF0\x9F\x8F\xAA", ur"""\U0001F3EA"""
#     ),
#     u"SCHOOL": (u"\xF0\x9F\x8F\xAB", ur"""\U0001F3EB"""
#     ),
#     u"DEPARTMENT STORE": (u"\xF0\x9F\x8F\xAC", ur"""\U0001F3EC"""
#     ),
#     u"FACTORY": (u"\xF0\x9F\x8F\xAD", ur"""\U0001F3ED"""
#     ),
#     u"IZAKAYA LANTERN": (u"\xF0\x9F\x8F\xAE", ur"""\U0001F3EE"""
#     ),
#     u"JAPANESE CASTLE": (u"\xF0\x9F\x8F\xAF", ur"""\U0001F3EF"""
#     ),
#     u"EUROPEAN CASTLE": (u"\xF0\x9F\x8F\xB0", ur"""\U0001F3F0"""
#     ),
#     u"SNAIL": (u"\xF0\x9F\x90\x8C", ur"""\U0001F40C"""
#     ),
#     u"SNAKE": (u"\xF0\x9F\x90\x8D", ur"""\U0001F40D"""
#     ),
#     u"HORSE": (u"\xF0\x9F\x90\x8E", ur"""\U0001F40E"""
#     ),
#     u"SHEEP": (u"\xF0\x9F\x90\x91", ur"""\U0001F411"""
#     ),
#     u"MONKEY": (u"\xF0\x9F\x90\x92", ur"""\U0001F412"""
#     ),
#     u"CHICKEN": (u"\xF0\x9F\x90\x94", ur"""\U0001F414"""
#     ),
#     u"BOAR": (u"\xF0\x9F\x90\x97", ur"""\U0001F417"""
#     ),
#     u"ELEPHANT": (u"\xF0\x9F\x90\x98", ur"""\U0001F418"""
#     ),
#     u"OCTOPUS": (u"\xF0\x9F\x90\x99", ur"""\U0001F419"""
#     ),
#     u"SPIRAL SHELL": (u"\xF0\x9F\x90\x9A", ur"""\U0001F41A"""
#     ),
#     u"BUG": (u"\xF0\x9F\x90\x9B", ur"""\U0001F41B"""
#     ),
#     u"ANT": (u"\xF0\x9F\x90\x9C", ur"""\U0001F41C"""
#     ),
#     u"HONEYBEE": (u"\xF0\x9F\x90\x9D", ur"""\U0001F41D"""
#     ),
#     u"LADY BEETLE": (u"\xF0\x9F\x90\x9E", ur"""\U0001F41E"""
#     ),
#     u"FISH": (u"\xF0\x9F\x90\x9F", ur"""\U0001F41F"""
#     ),
#     u"TROPICAL FISH": (u"\xF0\x9F\x90\xA0", ur"""\U0001F420"""
#     ),
#     u"BLOWFISH": (u"\xF0\x9F\x90\xA1", ur"""\U0001F421"""
#     ),
#     u"TURTLE": (u"\xF0\x9F\x90\xA2", ur"""\U0001F422"""
#     ),
#     u"HATCHING CHICK": (u"\xF0\x9F\x90\xA3", ur"""\U0001F423"""
#     ),
#     u"BABY CHICK": (u"\xF0\x9F\x90\xA4", ur"""\U0001F424"""
#     ),
#     u"FRONT-FACING BABY CHICK": (u"\xF0\x9F\x90\xA5", ur"""\U0001F425"""
#     ),
#     u"BIRD": (u"\xF0\x9F\x90\xA6", ur"""\U0001F426"""
#     ),
#     u"PENGUIN": (u"\xF0\x9F\x90\xA7", ur"""\U0001F427"""
#     ),
#     u"KOALA": (u"\xF0\x9F\x90\xA8", ur"""\U0001F428"""
#     ),
#     u"POODLE": (u"\xF0\x9F\x90\xA9", ur"""\U0001F429"""
#     ),
#     u"BACTRIAN CAMEL": (u"\xF0\x9F\x90\xAB", ur"""\U0001F42B"""
#     ),
#     u"DOLPHIN": (u"\xF0\x9F\x90\xAC", ur"""\U0001F42C"""
#     ),
#     u"MOUSE FACE": (u"\xF0\x9F\x90\xAD", ur"""\U0001F42D"""
#     ),
#     u"COW FACE": (u"\xF0\x9F\x90\xAE", ur"""\U0001F42E"""
#     ),
#     u"TIGER FACE": (u"\xF0\x9F\x90\xAF", ur"""\U0001F42F"""
#     ),
#     u"RABBIT FACE": (u"\xF0\x9F\x90\xB0", ur"""\U0001F430"""
#     ),
#     u"CAT FACE": (u"\xF0\x9F\x90\xB1", ur"""\U0001F431"""
#     ),
#     u"DRAGON FACE": (u"\xF0\x9F\x90\xB2", ur"""\U0001F432"""
#     ),
#     u"SPOUTING WHALE": (u"\xF0\x9F\x90\xB3", ur"""\U0001F433"""
#     ),
#     u"HORSE FACE": (u"\xF0\x9F\x90\xB4", ur"""\U0001F434"""
#     ),
#     u"MONKEY FACE": (u"\xF0\x9F\x90\xB5", ur"""\U0001F435"""
#     ),
    u"DOG FACE": (u"\xF0\x9F\x90\xB6", ur"""\U0001F436"""
    ),
#     u"PIG FACE": (u"\xF0\x9F\x90\xB7", ur"""\U0001F437"""
#     ),
#     u"FROG FACE": (u"\xF0\x9F\x90\xB8", ur"""\U0001F438"""
#     ),
#     u"HAMSTER FACE": (u"\xF0\x9F\x90\xB9", ur"""\U0001F439"""
#     ),
#     u"WOLF FACE": (u"\xF0\x9F\x90\xBA", ur"""\U0001F43A"""
#     ),
#     u"BEAR FACE": (u"\xF0\x9F\x90\xBB", ur"""\U0001F43B"""
#     ),
#     u"PANDA FACE": (u"\xF0\x9F\x90\xBC", ur"""\U0001F43C"""
#     ),
#     u"PIG NOSE": (u"\xF0\x9F\x90\xBD", ur"""\U0001F43D"""
#     ),
#     u"PAW PRINTS": (u"\xF0\x9F\x90\xBE", ur"""\U0001F43E"""
#     ),
    u"EYES": (u"\xF0\x9F\x91\x80", ur"""\U0001F440"""
    ),
#     u"EAR": (u"\xF0\x9F\x91\x82", ur"""\U0001F442"""
#     ),
#     u"NOSE": (u"\xF0\x9F\x91\x83", ur"""\U0001F443"""
#     ),
#     u"MOUTH": (u"\xF0\x9F\x91\x84", ur"""\U0001F444"""
#     ),
#     u"TONGUE": (u"\xF0\x9F\x91\x85", ur"""\U0001F445"""
#     ),
#     u"WHITE UP POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x86", ur"""\U0001F446"""
#     ),
#     u"WHITE DOWN POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x87", ur"""\U0001F447"""
#     ),
#     u"WHITE LEFT POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x88", ur"""\U0001F448"""
#     ),
#     u"WHITE RIGHT POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x89", ur"""\U0001F449"""
#     ),
#     u"FISTED HAND SIGN": (u"\xF0\x9F\x91\x8A", ur"""\U0001F44A"""
#     ),
#     u"WAVING HAND SIGN": (u"\xF0\x9F\x91\x8B", ur"""\U0001F44B"""
#     ),
    u"OK HAND SIGN": (u"\xF0\x9F\x91\x8C", ur"""\U0001F44C"""
    ),
    u"THUMBS UP SIGN": (u"\xF0\x9F\x91\x8D", ur"""\U0001F44D"""
    ),
    u"THUMBS DOWN SIGN": (u"\xF0\x9F\x91\x8E", ur"""\U0001F44E"""
    ),
    u"CLAPPING HANDS SIGN": (u"\xF0\x9F\x91\x8F", ur"""\U0001F44F"""
    ),
#     u"OPEN HANDS SIGN": (u"\xF0\x9F\x91\x90", ur"""\U0001F450"""
#     ),
    u"CROWN": (u"\xF0\x9F\x91\x91", ur"""\U0001F451"""
    ),
#     u"WOMANS HAT": (u"\xF0\x9F\x91\x92", ur"""\U0001F452"""
#     ),
#     u"EYEGLASSES": (u"\xF0\x9F\x91\x93", ur"""\U0001F453"""
#     ),
#     u"NECKTIE": (u"\xF0\x9F\x91\x94", ur"""\U0001F454"""
#     ),
#     u"T-SHIRT": (u"\xF0\x9F\x91\x95", ur"""\U0001F455"""
#     ),
#     u"JEANS": (u"\xF0\x9F\x91\x96", ur"""\U0001F456"""
#     ),
#     u"DRESS": (u"\xF0\x9F\x91\x97", ur"""\U0001F457"""
#     ),
#     u"KIMONO": (u"\xF0\x9F\x91\x98", ur"""\U0001F458"""
#     ),
#     u"BIKINI": (u"\xF0\x9F\x91\x99", ur"""\U0001F459"""
#     ),
#     u"WOMANS CLOTHES": (u"\xF0\x9F\x91\x9A", ur"""\U0001F45A"""
#     ),
#     u"PURSE": (u"\xF0\x9F\x91\x9B", ur"""\U0001F45B"""
#     ),
#     u"HANDBAG": (u"\xF0\x9F\x91\x9C", ur"""\U0001F45C"""
#     ),
#     u"POUCH": (u"\xF0\x9F\x91\x9D", ur"""\U0001F45D"""
#     ),
#     u"MANS SHOE": (u"\xF0\x9F\x91\x9E", ur"""\U0001F45E"""
#     ),
#     u"ATHLETIC SHOE": (u"\xF0\x9F\x91\x9F", ur"""\U0001F45F"""
#     ),
#     u"HIGH-HEELED SHOE": (u"\xF0\x9F\x91\xA0", ur"""\U0001F460"""
#     ),
#     u"WOMANS SANDAL": (u"\xF0\x9F\x91\xA1", ur"""\U0001F461"""
#     ),
#     u"WOMANS BOOTS": (u"\xF0\x9F\x91\xA2", ur"""\U0001F462"""
#     ),
#     u"FOOTPRINTS": (u"\xF0\x9F\x91\xA3", ur"""\U0001F463"""
#     ),
#     u"BUST IN SILHOUETTE": (u"\xF0\x9F\x91\xA4", ur"""\U0001F464"""
#     ),
#     u"BOY": (u"\xF0\x9F\x91\xA6", ur"""\U0001F466"""
#     ),
#     u"GIRL": (u"\xF0\x9F\x91\xA7", ur"""\U0001F467"""
#     ),
#     u"MAN": (u"\xF0\x9F\x91\xA8", ur"""\U0001F468"""
#     ),
#     u"WOMAN": (u"\xF0\x9F\x91\xA9", ur"""\U0001F469"""
#     ),
#     u"FAMILY": (u"\xF0\x9F\x91\xAA", ur"""\U0001F46A"""
#     ),
#     u"MAN AND WOMAN HOLDING HANDS": (u"\xF0\x9F\x91\xAB", ur"""\U0001F46B"""
#     ),
#     u"POLICE OFFICER": (u"\xF0\x9F\x91\xAE", ur"""\U0001F46E"""
#     ),
#     u"WOMAN WITH BUNNY EARS": (u"\xF0\x9F\x91\xAF", ur"""\U0001F46F"""
#     ),
#     u"BRIDE WITH VEIL": (u"\xF0\x9F\x91\xB0", ur"""\U0001F470"""
#     ),
#     u"PERSON WITH BLOND HAIR": (u"\xF0\x9F\x91\xB1", ur"""\U0001F471"""
#     ),
#     u"MAN WITH GUA PI MAO": (u"\xF0\x9F\x91\xB2", ur"""\U0001F472"""
#     ),
#     u"MAN WITH TURBAN": (u"\xF0\x9F\x91\xB3", ur"""\U0001F473"""
#     ),
#     u"OLDER MAN": (u"\xF0\x9F\x91\xB4", ur"""\U0001F474"""
#     ),
#     u"OLDER WOMAN": (u"\xF0\x9F\x91\xB5", ur"""\U0001F475"""
#     ),
#     u"BABY": (u"\xF0\x9F\x91\xB6", ur"""\U0001F476"""
#     ),
    u"CONSTRUCTION WORKER": (u"\xF0\x9F\x91\xB7", ur"""\U0001F477"""
    ),
#     u"PRINCESS": (u"\xF0\x9F\x91\xB8", ur"""\U0001F478"""
#     ),
#     u"JAPANESE OGRE": (u"\xF0\x9F\x91\xB9", ur"""\U0001F479"""
#     ),
#     u"JAPANESE GOBLIN": (u"\xF0\x9F\x91\xBA", ur"""\U0001F47A"""
#     ),
#     u"GHOST": (u"\xF0\x9F\x91\xBB", ur"""\U0001F47B"""
#     ),
#     u"BABY ANGEL": (u"\xF0\x9F\x91\xBC", ur"""\U0001F47C"""
#     ),
#     u"EXTRATERRESTRIAL ALIEN": (u"\xF0\x9F\x91\xBD", ur"""\U0001F47D"""
#     ),
#     u"ALIEN MONSTER": (u"\xF0\x9F\x91\xBE", ur"""\U0001F47E"""
#     ),
#     u"IMP": (u"\xF0\x9F\x91\xBF", ur"""\U0001F47F"""
#     ),
    u"SKULL": (u"\xF0\x9F\x92\x80", ur"""\U0001F480"""
    ),
    u"INFORMATION DESK PERSON": (u"\xF0\x9F\x92\x81", ur"""\U0001F481"""
    ),
#     u"GUARDSMAN": (u"\xF0\x9F\x92\x82", ur"""\U0001F482"""
#     ),
#     u"DANCER": (u"\xF0\x9F\x92\x83", ur"""\U0001F483"""
#     ),
#     u"LIPSTICK": (u"\xF0\x9F\x92\x84", ur"""\U0001F484"""
#     ),
#     u"NAIL POLISH": (u"\xF0\x9F\x92\x85", ur"""\U0001F485"""
#     ),
#     u"FACE MASSAGE": (u"\xF0\x9F\x92\x86", ur"""\U0001F486"""
#     ),
#     u"HAIRCUT": (u"\xF0\x9F\x92\x87", ur"""\U0001F487"""
#     ),
#     u"BARBER POLE": (u"\xF0\x9F\x92\x88", ur"""\U0001F488"""
#     ),
#     u"SYRINGE": (u"\xF0\x9F\x92\x89", ur"""\U0001F489"""
#     ),
#     u"PILL": (u"\xF0\x9F\x92\x8A", ur"""\U0001F48A"""
#     ),
#     u"KISS MARK": (u"\xF0\x9F\x92\x8B", ur"""\U0001F48B"""
#     ),
#     u"LOVE LETTER": (u"\xF0\x9F\x92\x8C", ur"""\U0001F48C"""
#     ),
#     u"RING": (u"\xF0\x9F\x92\x8D", ur"""\U0001F48D"""
#     ),
#     u"GEM STONE": (u"\xF0\x9F\x92\x8E", ur"""\U0001F48E"""
#     ),
#     u"KISS": (u"\xF0\x9F\x92\x8F", ur"""\U0001F48F"""
#     ),
#     u"BOUQUET": (u"\xF0\x9F\x92\x90", ur"""\U0001F490"""
#     ),
#     u"COUPLE WITH HEART": (u"\xF0\x9F\x92\x91", ur"""\U0001F491"""
#     ),
#     u"WEDDING": (u"\xF0\x9F\x92\x92", ur"""\U0001F492"""
#     ),
#     u"BEATING HEART": (u"\xF0\x9F\x92\x93", ur"""\U0001F493"""
#     ),
#     u"BROKEN HEART": (u"\xF0\x9F\x92\x94", ur"""\U0001F494"""
#     ),
#     u"TWO HEARTS": (u"\xF0\x9F\x92\x95", ur"""\U0001F495"""
#     ),
#     u"SPARKLING HEART": (u"\xF0\x9F\x92\x96", ur"""\U0001F496"""
#     ),
#     u"GROWING HEART": (u"\xF0\x9F\x92\x97", ur"""\U0001F497"""
#     ),
#     u"HEART WITH ARROW": (u"\xF0\x9F\x92\x98", ur"""\U0001F498"""
#     ),
#     u"BLUE HEART": (u"\xF0\x9F\x92\x99", ur"""\U0001F499"""
#     ),
#     u"GREEN HEART": (u"\xF0\x9F\x92\x9A", ur"""\U0001F49A"""
#     ),
#     u"YELLOW HEART": (u"\xF0\x9F\x92\x9B", ur"""\U0001F49B"""
#     ),
#     u"PURPLE HEART": (u"\xF0\x9F\x92\x9C", ur"""\U0001F49C"""
#     ),
#     u"HEART WITH RIBBON": (u"\xF0\x9F\x92\x9D", ur"""\U0001F49D"""
#     ),
#     u"REVOLVING HEARTS": (u"\xF0\x9F\x92\x9E", ur"""\U0001F49E"""
#     ),
#     u"HEART DECORATION": (u"\xF0\x9F\x92\x9F", ur"""\U0001F49F"""
#     ),
#     u"DIAMOND SHAPE WITH A DOT INSIDE": (u"\xF0\x9F\x92\xA0", ur"""\U0001F4A0"""
#     ),
#     u"ELECTRIC LIGHT BULB": (u"\xF0\x9F\x92\xA1", ur"""\U0001F4A1"""
#     ),
#     u"ANGER SYMBOL": (u"\xF0\x9F\x92\xA2", ur"""\U0001F4A2"""
#     ),
#     u"BOMB": (u"\xF0\x9F\x92\xA3", ur"""\U0001F4A3"""
#     ),
#     u"SLEEPING SYMBOL": (u"\xF0\x9F\x92\xA4", ur"""\U0001F4A4"""
#     ),
#     u"COLLISION SYMBOL": (u"\xF0\x9F\x92\xA5", ur"""\U0001F4A5"""
#     ),
#     u"SPLASHING SWEAT SYMBOL": (u"\xF0\x9F\x92\xA6", ur"""\U0001F4A6"""
#     ),
#     u"DROPLET": (u"\xF0\x9F\x92\xA7", ur"""\U0001F4A7"""
#     ),
#     u"DASH SYMBOL": (u"\xF0\x9F\x92\xA8", ur"""\U0001F4A8"""
#     ),
#     u"PILE OF POO": (u"\xF0\x9F\x92\xA9", ur"""\U0001F4A9"""
#     ),
#     u"FLEXED BICEPS": (u"\xF0\x9F\x92\xAA", ur"""\U0001F4AA"""
#     ),
#     u"DIZZY SYMBOL": (u"\xF0\x9F\x92\xAB", ur"""\U0001F4AB"""
#     ),
#     u"SPEECH BALLOON": (u"\xF0\x9F\x92\xAC", ur"""\U0001F4AC"""
#     ),
#     u"WHITE FLOWER": (u"\xF0\x9F\x92\xAE", ur"""\U0001F4AE"""
#     ),
    u"HUNDRED POINTS SYMBOL": (u"\xF0\x9F\x92\xAF", ur"""\U0001F4AF"""
    ),
#     u"MONEY BAG": (u"\xF0\x9F\x92\xB0", ur"""\U0001F4B0"""
#     ),
#     u"CURRENCY EXCHANGE": (u"\xF0\x9F\x92\xB1", ur"""\U0001F4B1"""
#     ),
#     u"HEAVY DOLLAR SIGN": (u"\xF0\x9F\x92\xB2", ur"""\U0001F4B2"""
#     ),
#     u"CREDIT CARD": (u"\xF0\x9F\x92\xB3", ur"""\U0001F4B3"""
#     ),
#     u"BANKNOTE WITH YEN SIGN": (u"\xF0\x9F\x92\xB4", ur"""\U0001F4B4"""
#     ),
#     u"BANKNOTE WITH DOLLAR SIGN": (u"\xF0\x9F\x92\xB5", ur"""\U0001F4B5"""
#     ),
#     u"MONEY WITH WINGS": (u"\xF0\x9F\x92\xB8", ur"""\U0001F4B8"""
#     ),
#     u"CHART WITH UPWARDS TREND AND YEN SIGN": (u"\xF0\x9F\x92\xB9", ur"""\U0001F4B9"""
#     ),
#     u"SEAT": (u"\xF0\x9F\x92\xBA", ur"""\U0001F4BA"""
#     ),
#     u"PERSONAL COMPUTER": (u"\xF0\x9F\x92\xBB", ur"""\U0001F4BB"""
#     ),
#     u"BRIEFCASE": (u"\xF0\x9F\x92\xBC", ur"""\U0001F4BC"""
#     ),
#     u"MINIDISC": (u"\xF0\x9F\x92\xBD", ur"""\U0001F4BD"""
#     ),
#     u"FLOPPY DISK": (u"\xF0\x9F\x92\xBE", ur"""\U0001F4BE"""
#     ),
#     u"OPTICAL DISC": (u"\xF0\x9F\x92\xBF", ur"""\U0001F4BF"""
#     ),
#     u"DVD": (u"\xF0\x9F\x93\x80", ur"""\U0001F4C0"""
#     ),
#     u"FILE FOLDER": (u"\xF0\x9F\x93\x81", ur"""\U0001F4C1"""
#     ),
#     u"OPEN FILE FOLDER": (u"\xF0\x9F\x93\x82", ur"""\U0001F4C2"""
#     ),
#     u"PAGE WITH CURL": (u"\xF0\x9F\x93\x83", ur"""\U0001F4C3"""
#     ),
#     u"PAGE FACING UP": (u"\xF0\x9F\x93\x84", ur"""\U0001F4C4"""
#     ),
#     u"CALENDAR": (u"\xF0\x9F\x93\x85", ur"""\U0001F4C5"""
#     ),
#     u"TEAR-OFF CALENDAR": (u"\xF0\x9F\x93\x86", ur"""\U0001F4C6"""
#     ),
#     u"CARD INDEX": (u"\xF0\x9F\x93\x87", ur"""\U0001F4C7"""
#     ),
#     u"CHART WITH UPWARDS TREND": (u"\xF0\x9F\x93\x88", ur"""\U0001F4C8"""
#     ),
#     u"CHART WITH DOWNWARDS TREND": (u"\xF0\x9F\x93\x89", ur"""\U0001F4C9"""
#     ),
#     u"BAR CHART": (u"\xF0\x9F\x93\x8A", ur"""\U0001F4CA"""
#     ),
#     u"CLIPBOARD": (u"\xF0\x9F\x93\x8B", ur"""\U0001F4CB"""
#     ),
#     u"PUSHPIN": (u"\xF0\x9F\x93\x8C", ur"""\U0001F4CC"""
#     ),
#     u"ROUND PUSHPIN": (u"\xF0\x9F\x93\x8D", ur"""\U0001F4CD"""
#     ),
#     u"PAPERCLIP": (u"\xF0\x9F\x93\x8E", ur"""\U0001F4CE"""
#     ),
#     u"STRAIGHT RULER": (u"\xF0\x9F\x93\x8F", ur"""\U0001F4CF"""
#     ),
#     u"TRIANGULAR RULER": (u"\xF0\x9F\x93\x90", ur"""\U0001F4D0"""
#     ),
#     u"BOOKMARK TABS": (u"\xF0\x9F\x93\x91", ur"""\U0001F4D1"""
#     ),
#     u"LEDGER": (u"\xF0\x9F\x93\x92", ur"""\U0001F4D2"""
#     ),
#     u"NOTEBOOK": (u"\xF0\x9F\x93\x93", ur"""\U0001F4D3"""
#     ),
#     u"NOTEBOOK WITH DECORATIVE COVER": (u"\xF0\x9F\x93\x94", ur"""\U0001F4D4"""
#     ),
#     u"CLOSED BOOK": (u"\xF0\x9F\x93\x95", ur"""\U0001F4D5"""
#     ),
#     u"OPEN BOOK": (u"\xF0\x9F\x93\x96", ur"""\U0001F4D6"""
#     ),
#     u"GREEN BOOK": (u"\xF0\x9F\x93\x97", ur"""\U0001F4D7"""
#     ),
#     u"BLUE BOOK": (u"\xF0\x9F\x93\x98", ur"""\U0001F4D8"""
#     ),
#     u"ORANGE BOOK": (u"\xF0\x9F\x93\x99", ur"""\U0001F4D9"""
#     ),
#     u"BOOKS": (u"\xF0\x9F\x93\x9A", ur"""\U0001F4DA"""
#     ),
#     u"NAME BADGE": (u"\xF0\x9F\x93\x9B", ur"""\U0001F4DB"""
#     ),
#     u"SCROLL": (u"\xF0\x9F\x93\x9C", ur"""\U0001F4DC"""
#     ),
#     u"MEMO": (u"\xF0\x9F\x93\x9D", ur"""\U0001F4DD"""
#     ),
#     u"TELEPHONE RECEIVER": (u"\xF0\x9F\x93\x9E", ur"""\U0001F4DE"""
#     ),
#     u"PAGER": (u"\xF0\x9F\x93\x9F", ur"""\U0001F4DF"""
#     ),
#     u"FAX MACHINE": (u"\xF0\x9F\x93\xA0", ur"""\U0001F4E0"""
#     ),
#     u"SATELLITE ANTENNA": (u"\xF0\x9F\x93\xA1", ur"""\U0001F4E1"""
#     ),
#     u"PUBLIC ADDRESS LOUDSPEAKER": (u"\xF0\x9F\x93\xA2", ur"""\U0001F4E2"""
#     ),
#     u"CHEERING MEGAPHONE": (u"\xF0\x9F\x93\xA3", ur"""\U0001F4E3"""
#     ),
#     u"OUTBOX TRAY": (u"\xF0\x9F\x93\xA4", ur"""\U0001F4E4"""
#     ),
#     u"INBOX TRAY": (u"\xF0\x9F\x93\xA5", ur"""\U0001F4E5"""
#     ),
#     u"PACKAGE": (u"\xF0\x9F\x93\xA6", ur"""\U0001F4E6"""
#     ),
#     u"E-MAIL SYMBOL": (u"\xF0\x9F\x93\xA7", ur"""\U0001F4E7"""
#     ),
#     u"INCOMING ENVELOPE": (u"\xF0\x9F\x93\xA8", ur"""\U0001F4E8"""
#     ),
#     u"ENVELOPE WITH DOWNWARDS ARROW ABOVE": (u"\xF0\x9F\x93\xA9", ur"""\U0001F4E9"""
#     ),
#     u"CLOSED MAILBOX WITH LOWERED FLAG": (u"\xF0\x9F\x93\xAA", ur"""\U0001F4EA"""
#     ),
#     u"CLOSED MAILBOX WITH RAISED FLAG": (u"\xF0\x9F\x93\xAB", ur"""\U0001F4EB"""
#     ),
#     u"POSTBOX": (u"\xF0\x9F\x93\xAE", ur"""\U0001F4EE"""
#     ),
#     u"NEWSPAPER": (u"\xF0\x9F\x93\xB0", ur"""\U0001F4F0"""
#     ),
#     u"MOBILE PHONE": (u"\xF0\x9F\x93\xB1", ur"""\U0001F4F1"""
#     ),
#     u"MOBILE PHONE WITH RIGHTWARDS ARROW AT LEFT": (u"\xF0\x9F\x93\xB2", ur"""\U0001F4F2"""
#     ),
#     u"VIBRATION MODE": (u"\xF0\x9F\x93\xB3", ur"""\U0001F4F3"""
#     ),
#     u"MOBILE PHONE OFF": (u"\xF0\x9F\x93\xB4", ur"""\U0001F4F4"""
#     ),
#     u"ANTENNA WITH BARS": (u"\xF0\x9F\x93\xB6", ur"""\U0001F4F6"""
#     ),
#     u"CAMERA": (u"\xF0\x9F\x93\xB7", ur"""\U0001F4F7"""
#     ),
#     u"VIDEO CAMERA": (u"\xF0\x9F\x93\xB9", ur"""\U0001F4F9"""
#     ),
#     u"TELEVISION": (u"\xF0\x9F\x93\xBA", ur"""\U0001F4FA"""
#     ),
#     u"RADIO": (u"\xF0\x9F\x93\xBB", ur"""\U0001F4FB"""
#     ),
#     u"VIDEOCASSETTE": (u"\xF0\x9F\x93\xBC", ur"""\U0001F4FC"""
#     ),
#     u"CLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x83", ur"""\U0001F503"""
#     ),
#     u"SPEAKER WITH THREE SOUND WAVES": (u"\xF0\x9F\x94\x8A", ur"""\U0001F50A"""
#     ),
#     u"BATTERY": (u"\xF0\x9F\x94\x8B", ur"""\U0001F50B"""
#     ),
#     u"ELECTRIC PLUG": (u"\xF0\x9F\x94\x8C", ur"""\U0001F50C"""
#     ),
#     u"LEFT-POINTING MAGNIFYING GLASS": (u"\xF0\x9F\x94\x8D", ur"""\U0001F50D"""
#     ),
#     u"RIGHT-POINTING MAGNIFYING GLASS": (u"\xF0\x9F\x94\x8E", ur"""\U0001F50E"""
#     ),
#     u"LOCK WITH INK PEN": (u"\xF0\x9F\x94\x8F", ur"""\U0001F50F"""
#     ),
#     u"CLOSED LOCK WITH KEY": (u"\xF0\x9F\x94\x90", ur"""\U0001F510"""
#     ),
#     u"KEY": (u"\xF0\x9F\x94\x91", ur"""\U0001F511"""
#     ),
#     u"LOCK": (u"\xF0\x9F\x94\x92", ur"""\U0001F512"""
#     ),
#     u"OPEN LOCK": (u"\xF0\x9F\x94\x93", ur"""\U0001F513"""
#     ),
#     u"BELL": (u"\xF0\x9F\x94\x94", ur"""\U0001F514"""
#     ),
#     u"BOOKMARK": (u"\xF0\x9F\x94\x96", ur"""\U0001F516"""
#     ),
#     u"LINK SYMBOL": (u"\xF0\x9F\x94\x97", ur"""\U0001F517"""
#     ),
#     u"RADIO BUTTON": (u"\xF0\x9F\x94\x98", ur"""\U0001F518"""
#     ),
#     u"BACK WITH LEFTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x99", ur"""\U0001F519"""
#     ),
#     u"END WITH LEFTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9A", ur"""\U0001F51A"""
#     ),
#     u"ON WITH EXCLAMATION MARK WITH LEFT RIGHT ARROW ABOVE": (u"\xF0\x9F\x94\x9B", ur"""\U0001F51B"""
#     ),
#     u"SOON WITH RIGHTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9C", ur"""\U0001F51C"""
#     ),
#     u"TOP WITH UPWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9D", ur"""\U0001F51D"""
#     ),
#     u"NO ONE UNDER EIGHTEEN SYMBOL": (u"\xF0\x9F\x94\x9E", ur"""\U0001F51E"""
#     ),
#     u"KEYCAP TEN": (u"\xF0\x9F\x94\x9F", ur"""\U0001F51F"""
#     ),
#     u"INPUT SYMBOL FOR LATIN CAPITAL LETTERS": (u"\xF0\x9F\x94\xA0", ur"""\U0001F520"""
#     ),
#     u"INPUT SYMBOL FOR LATIN SMALL LETTERS": (u"\xF0\x9F\x94\xA1", ur"""\U0001F521"""
#     ),
#     u"INPUT SYMBOL FOR NUMBERS": (u"\xF0\x9F\x94\xA2", ur"""\U0001F522"""
#     ),
#     u"INPUT SYMBOL FOR SYMBOLS": (u"\xF0\x9F\x94\xA3", ur"""\U0001F523"""
#     ),
#     u"INPUT SYMBOL FOR LATIN LETTERS": (u"\xF0\x9F\x94\xA4", ur"""\U0001F524"""
#     ),
#     u"FIRE": (u"\xF0\x9F\x94\xA5", ur"""\U0001F525"""
#     ),
#     u"ELECTRIC TORCH": (u"\xF0\x9F\x94\xA6", ur"""\U0001F526"""
#     ),
#     u"WRENCH": (u"\xF0\x9F\x94\xA7", ur"""\U0001F527"""
#     ),
#     u"HAMMER": (u"\xF0\x9F\x94\xA8", ur"""\U0001F528"""
#     ),
#     u"NUT AND BOLT": (u"\xF0\x9F\x94\xA9", ur"""\U0001F529"""
#     ),
#     u"HOCHO": (u"\xF0\x9F\x94\xAA", ur"""\U0001F52A"""
#     ),
#     u"PISTOL": (u"\xF0\x9F\x94\xAB", ur"""\U0001F52B"""
#     ),
#     u"CRYSTAL BALL": (u"\xF0\x9F\x94\xAE", ur"""\U0001F52E"""
#     ),
#     u"SIX POINTED STAR WITH MIDDLE DOT": (u"\xF0\x9F\x94\xAF", ur"""\U0001F52F"""
#     ),
#     u"JAPANESE SYMBOL FOR BEGINNER": (u"\xF0\x9F\x94\xB0", ur"""\U0001F530"""
#     ),
#     u"TRIDENT EMBLEM": (u"\xF0\x9F\x94\xB1", ur"""\U0001F531"""
#     ),
#     u"BLACK SQUARE BUTTON": (u"\xF0\x9F\x94\xB2", ur"""\U0001F532"""
#     ),
#     u"WHITE SQUARE BUTTON": (u"\xF0\x9F\x94\xB3", ur"""\U0001F533"""
#     ),
#     u"LARGE RED CIRCLE": (u"\xF0\x9F\x94\xB4", ur"""\U0001F534"""
#     ),
#     u"LARGE BLUE CIRCLE": (u"\xF0\x9F\x94\xB5", ur"""\U0001F535"""
#     ),
#     u"LARGE ORANGE DIAMOND": (u"\xF0\x9F\x94\xB6", ur"""\U0001F536"""
#     ),
#     u"LARGE BLUE DIAMOND": (u"\xF0\x9F\x94\xB7", ur"""\U0001F537"""
#     ),
#     u"SMALL ORANGE DIAMOND": (u"\xF0\x9F\x94\xB8", ur"""\U0001F538"""
#     ),
#     u"SMALL BLUE DIAMOND": (u"\xF0\x9F\x94\xB9", ur"""\U0001F539"""
#     ),
#     u"UP-POINTING RED TRIANGLE": (u"\xF0\x9F\x94\xBA", ur"""\U0001F53A"""
#     ),
#     u"DOWN-POINTING RED TRIANGLE": (u"\xF0\x9F\x94\xBB", ur"""\U0001F53B"""
#     ),
#     u"UP-POINTING SMALL RED TRIANGLE": (u"\xF0\x9F\x94\xBC", ur"""\U0001F53C"""
#     ),
#     u"DOWN-POINTING SMALL RED TRIANGLE": (u"\xF0\x9F\x94\xBD", ur"""\U0001F53D"""
#     ),
#     u"CLOCK FACE ONE OCLOCK": (u"\xF0\x9F\x95\x90", ur"""\U0001F550"""
#     ),
#     u"CLOCK FACE TWO OCLOCK": (u"\xF0\x9F\x95\x91", ur"""\U0001F551"""
#     ),
#     u"CLOCK FACE THREE OCLOCK": (u"\xF0\x9F\x95\x92", ur"""\U0001F552"""
#     ),
#     u"CLOCK FACE FOUR OCLOCK": (u"\xF0\x9F\x95\x93", ur"""\U0001F553"""
#     ),
#     u"CLOCK FACE FIVE OCLOCK": (u"\xF0\x9F\x95\x94", ur"""\U0001F554"""
#     ),
#     u"CLOCK FACE SIX OCLOCK": (u"\xF0\x9F\x95\x95", ur"""\U0001F555"""
#     ),
#     u"CLOCK FACE SEVEN OCLOCK": (u"\xF0\x9F\x95\x96", ur"""\U0001F556"""
#     ),
#     u"CLOCK FACE EIGHT OCLOCK": (u"\xF0\x9F\x95\x97", ur"""\U0001F557"""
#     ),
#     u"CLOCK FACE NINE OCLOCK": (u"\xF0\x9F\x95\x98", ur"""\U0001F558"""
#     ),
#     u"CLOCK FACE TEN OCLOCK": (u"\xF0\x9F\x95\x99", ur"""\U0001F559"""
#     ),
#     u"CLOCK FACE ELEVEN OCLOCK": (u"\xF0\x9F\x95\x9A", ur"""\U0001F55A"""
#     ),
#     u"CLOCK FACE TWELVE OCLOCK": (u"\xF0\x9F\x95\x9B", ur"""\U0001F55B"""
#     ),
#     u"MOUNT FUJI": (u"\xF0\x9F\x97\xBB", ur"""\U0001F5FB"""
#     ),
#     u"TOKYO TOWER": (u"\xF0\x9F\x97\xBC", ur"""\U0001F5FC"""
#     ),
#     u"STATUE OF LIBERTY": (u"\xF0\x9F\x97\xBD", ur"""\U0001F5FD"""
#     ),
#     u"SILHOUETTE OF JAPAN": (u"\xF0\x9F\x97\xBE", ur"""\U0001F5FE"""
#     ),
#     u"MOYAI": (u"\xF0\x9F\x97\xBF", ur"""\U0001F5FF"""
#     ),
#     u"GRINNING FACE": (u"\xF0\x9F\x98\x80", ur"""\U0001F600"""
#     ),
#     u"SMILING FACE WITH HALO": (u"\xF0\x9F\x98\x87", ur"""\U0001F607"""
#     ),
#     u"SMILING FACE WITH HORNS": (u"\xF0\x9F\x98\x88", ur"""\U0001F608"""
#     ),
    u"SMILING FACE WITH SUNGLASSES": (u"\xF0\x9F\x98\x8E", ur"""\U0001F60E"""
    ),
    u"NEUTRAL FACE": (u"\xF0\x9F\x98\x90", ur"""\U0001F610"""
    ),
    u"EXPRESSIONLESS FACE": (u"\xF0\x9F\x98\x91", ur"""\U0001F611"""
    ),
    u"CONFUSED FACE": (u"\xF0\x9F\x98\x95", ur"""\U0001F615"""
    ),
#     u"KISSING FACE": (u"\xF0\x9F\x98\x97", ur"""\U0001F617"""
#     ),
#     u"KISSING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x99", ur"""\U0001F619"""
#     ),
#     u"FACE WITH STUCK-OUT TONGUE": (u"\xF0\x9F\x98\x9B", ur"""\U0001F61B"""
#     ),
#     u"WORRIED FACE": (u"\xF0\x9F\x98\x9F", ur"""\U0001F61F"""
#     ),
    u"FROWNING FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xA6", ur"""\U0001F626"""
    ),
    u"ANGUISHED FACE": (u"\xF0\x9F\x98\xA7", ur"""\U0001F627"""
    ),
    u"GRIMACING FACE": (u"\xF0\x9F\x98\xAC", ur"""\U0001F62C"""
    ),
#     u"FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xAE", ur"""\U0001F62E"""
#     ),
#     u"HUSHED FACE": (u"\xF0\x9F\x98\xAF", ur"""\U0001F62F"""
#     ),
    u"SLEEPING FACE": (u"\xF0\x9F\x98\xB4", ur"""\U0001F634"""
    ),
#     u"FACE WITHOUT MOUTH": (u"\xF0\x9F\x98\xB6", ur"""\U0001F636"""
#     ),
#     u"HELICOPTER": (u"\xF0\x9F\x9A\x81", ur"""\U0001F681"""
#     ),
#     u"STEAM LOCOMOTIVE": (u"\xF0\x9F\x9A\x82", ur"""\U0001F682"""
#     ),
#     u"TRAIN": (u"\xF0\x9F\x9A\x86", ur"""\U0001F686"""
#     ),
#     u"LIGHT RAIL": (u"\xF0\x9F\x9A\x88", ur"""\U0001F688"""
#     ),
#     u"TRAM": (u"\xF0\x9F\x9A\x8A", ur"""\U0001F68A"""
#     ),
#     u"ONCOMING BUS": (u"\xF0\x9F\x9A\x8D", ur"""\U0001F68D"""
#     ),
#     u"TROLLEYBUS": (u"\xF0\x9F\x9A\x8E", ur"""\U0001F68E"""
#     ),
#     u"MINIBUS": (u"\xF0\x9F\x9A\x90", ur"""\U0001F690"""
#     ),
#     u"ONCOMING POLICE CAR": (u"\xF0\x9F\x9A\x94", ur"""\U0001F694"""
#     ),
#     u"ONCOMING TAXI": (u"\xF0\x9F\x9A\x96", ur"""\U0001F696"""
#     ),
#     u"ONCOMING AUTOMOBILE": (u"\xF0\x9F\x9A\x98", ur"""\U0001F698"""
#     ),
#     u"ARTICULATED LORRY": (u"\xF0\x9F\x9A\x9B", ur"""\U0001F69B"""
#     ),
#     u"TRACTOR": (u"\xF0\x9F\x9A\x9C", ur"""\U0001F69C"""
#     ),
#     u"MONORAIL": (u"\xF0\x9F\x9A\x9D", ur"""\U0001F69D"""
#     ),
#     u"MOUNTAIN RAILWAY": (u"\xF0\x9F\x9A\x9E", ur"""\U0001F69E"""
#     ),
#     u"SUSPENSION RAILWAY": (u"\xF0\x9F\x9A\x9F", ur"""\U0001F69F"""
#     ),
#     u"MOUNTAIN CABLEWAY": (u"\xF0\x9F\x9A\xA0", ur"""\U0001F6A0"""
#     ),
#     u"AERIAL TRAMWAY": (u"\xF0\x9F\x9A\xA1", ur"""\U0001F6A1"""
#     ),
#     u"ROWBOAT": (u"\xF0\x9F\x9A\xA3", ur"""\U0001F6A3"""
#     ),
#     u"VERTICAL TRAFFIC LIGHT": (u"\xF0\x9F\x9A\xA6", ur"""\U0001F6A6"""
#     ),
#     u"PUT LITTER IN ITS PLACE SYMBOL": (u"\xF0\x9F\x9A\xAE", ur"""\U0001F6AE"""
#     ),
#     u"DO NOT LITTER SYMBOL": (u"\xF0\x9F\x9A\xAF", ur"""\U0001F6AF"""
#     ),
#     u"POTABLE WATER SYMBOL": (u"\xF0\x9F\x9A\xB0", ur"""\U0001F6B0"""
#     ),
#     u"NON-POTABLE WATER SYMBOL": (u"\xF0\x9F\x9A\xB1", ur"""\U0001F6B1"""
#     ),
#     u"NO BICYCLES": (u"\xF0\x9F\x9A\xB3", ur"""\U0001F6B3"""
#     ),
#     u"BICYCLIST": (u"\xF0\x9F\x9A\xB4", ur"""\U0001F6B4"""
#     ),
#     u"MOUNTAIN BICYCLIST": (u"\xF0\x9F\x9A\xB5", ur"""\U0001F6B5"""
#     ),
#     u"NO PEDESTRIANS": (u"\xF0\x9F\x9A\xB7", ur"""\U0001F6B7"""
#     ),
#     u"CHILDREN CROSSING": (u"\xF0\x9F\x9A\xB8", ur"""\U0001F6B8"""
#     ),
#     u"SHOWER": (u"\xF0\x9F\x9A\xBF", ur"""\U0001F6BF"""
#     ),
#     u"BATHTUB": (u"\xF0\x9F\x9B\x81", ur"""\U0001F6C1"""
#     ),
#     u"PASSPORT CONTROL": (u"\xF0\x9F\x9B\x82", ur"""\U0001F6C2"""
#     ),
#     u"CUSTOMS": (u"\xF0\x9F\x9B\x83", ur"""\U0001F6C3"""
#     ),
#     u"BAGGAGE CLAIM": (u"\xF0\x9F\x9B\x84", ur"""\U0001F6C4"""
#     ),
#     u"LEFT LUGGAGE": (u"\xF0\x9F\x9B\x85", ur"""\U0001F6C5"""
#     ),
#     u"EARTH GLOBE EUROPE-AFRICA": (u"\xF0\x9F\x8C\x8D", ur"""\U0001F30D"""
#     ),
    u"EARTH GLOBE AMERICAS": (u"\xF0\x9F\x8C\x8E", ur"""\U0001F30E"""
    ),
#     u"GLOBE WITH MERIDIANS": (u"\xF0\x9F\x8C\x90", ur"""\U0001F310"""
#     ),
#     u"WAXING CRESCENT MOON SYMBOL": (u"\xF0\x9F\x8C\x92", ur"""\U0001F312"""
#     ),
#     u"WANING GIBBOUS MOON SYMBOL": (u"\xF0\x9F\x8C\x96", ur"""\U0001F316"""
#     ),
#     u"LAST QUARTER MOON SYMBOL": (u"\xF0\x9F\x8C\x97", ur"""\U0001F317"""
#     ),
#     u"WANING CRESCENT MOON SYMBOL": (u"\xF0\x9F\x8C\x98", ur"""\U0001F318"""
#     ),
#     u"NEW MOON WITH FACE": (u"\xF0\x9F\x8C\x9A", ur"""\U0001F31A"""
#     ),
#     u"LAST QUARTER MOON WITH FACE": (u"\xF0\x9F\x8C\x9C", ur"""\U0001F31C"""
#     ),
#     u"FULL MOON WITH FACE": (u"\xF0\x9F\x8C\x9D", ur"""\U0001F31D"""
#     ),
#     u"SUN WITH FACE": (u"\xF0\x9F\x8C\x9E", ur"""\U0001F31E"""
#     ),
#     u"EVERGREEN TREE": (u"\xF0\x9F\x8C\xB2", ur"""\U0001F332"""
#     ),
#     u"DECIDUOUS TREE": (u"\xF0\x9F\x8C\xB3", ur"""\U0001F333"""
#     ),
#     u"LEMON": (u"\xF0\x9F\x8D\x8B", ur"""\U0001F34B"""
#     ),
#     u"PEAR": (u"\xF0\x9F\x8D\x90", ur"""\U0001F350"""
#     ),
#     u"BABY BOTTLE": (u"\xF0\x9F\x8D\xBC", ur"""\U0001F37C"""
#     ),
#     u"HORSE RACING": (u"\xF0\x9F\x8F\x87", ur"""\U0001F3C7"""
#     ),
#     u"RUGBY FOOTBALL": (u"\xF0\x9F\x8F\x89", ur"""\U0001F3C9"""
#     ),
#     u"EUROPEAN POST OFFICE": (u"\xF0\x9F\x8F\xA4", ur"""\U0001F3E4"""
#     ),
#     u"RAT": (u"\xF0\x9F\x90\x80", ur"""\U0001F400"""
#     ),
#     u"MOUSE": (u"\xF0\x9F\x90\x81", ur"""\U0001F401"""
#     ),
#     u"OX": (u"\xF0\x9F\x90\x82", ur"""\U0001F402"""
#     ),
#     u"WATER BUFFALO": (u"\xF0\x9F\x90\x83", ur"""\U0001F403"""
#     ),
#     u"COW": (u"\xF0\x9F\x90\x84", ur"""\U0001F404"""
#     ),
#     u"TIGER": (u"\xF0\x9F\x90\x85", ur"""\U0001F405"""
#     ),
#     u"LEOPARD": (u"\xF0\x9F\x90\x86", ur"""\U0001F406"""
#     ),
#     u"RABBIT": (u"\xF0\x9F\x90\x87", ur"""\U0001F407"""
#     ),
#     u"CAT": (u"\xF0\x9F\x90\x88", ur"""\U0001F408"""
#     ),
#     u"DRAGON": (u"\xF0\x9F\x90\x89", ur"""\U0001F409"""
#     ),
#     u"CROCODILE": (u"\xF0\x9F\x90\x8A", ur"""\U0001F40A"""
#     ),
#     u"WHALE": (u"\xF0\x9F\x90\x8B", ur"""\U0001F40B"""
#     ),
#     u"RAM": (u"\xF0\x9F\x90\x8F", ur"""\U0001F40F"""
#     ),
#     u"GOAT": (u"\xF0\x9F\x90\x90", ur"""\U0001F410"""
#     ),
#     u"ROOSTER": (u"\xF0\x9F\x90\x93", ur"""\U0001F413"""
#     ),
#     u"DOG": (u"\xF0\x9F\x90\x95", ur"""\U0001F415"""
#     ),
#     u"PIG": (u"\xF0\x9F\x90\x96", ur"""\U0001F416"""
#     ),
#     u"DROMEDARY CAMEL": (u"\xF0\x9F\x90\xAA", ur"""\U0001F42A"""
#     ),
#     u"BUSTS IN SILHOUETTE": (u"\xF0\x9F\x91\xA5", ur"""\U0001F465"""
#     ),
#     u"TWO MEN HOLDING HANDS": (u"\xF0\x9F\x91\xAC", ur"""\U0001F46C"""
#     ),
#     u"TWO WOMEN HOLDING HANDS": (u"\xF0\x9F\x91\xAD", ur"""\U0001F46D"""
#     ),
#     u"THOUGHT BALLOON": (u"\xF0\x9F\x92\xAD", ur"""\U0001F4AD"""
#     ),
#     u"BANKNOTE WITH EURO SIGN": (u"\xF0\x9F\x92\xB6", ur"""\U0001F4B6"""
#     ),
#     u"BANKNOTE WITH POUND SIGN": (u"\xF0\x9F\x92\xB7", ur"""\U0001F4B7"""
#     ),
#     u"OPEN MAILBOX WITH RAISED FLAG": (u"\xF0\x9F\x93\xAC", ur"""\U0001F4EC"""
#     ),
#     u"OPEN MAILBOX WITH LOWERED FLAG": (u"\xF0\x9F\x93\xAD", ur"""\U0001F4ED"""
#     ),
#     u"POSTAL HORN": (u"\xF0\x9F\x93\xAF", ur"""\U0001F4EF"""
#     ),
#     u"NO MOBILE PHONES": (u"\xF0\x9F\x93\xB5", ur"""\U0001F4F5"""
#     ),
#     u"TWISTED RIGHTWARDS ARROWS": (u"\xF0\x9F\x94\x80", ur"""\U0001F500"""
#     ),
#     u"CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x81", ur"""\U0001F501"""
#     ),
#     u"CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS WITH CIRCLED ONE OVERLAY": (u"\xF0\x9F\x94\x82", ur"""\U0001F502"""
#     ),
#     u"ANTICLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x84", ur"""\U0001F504"""
#     ),
#     u"LOW BRIGHTNESS SYMBOL": (u"\xF0\x9F\x94\x85", ur"""\U0001F505"""
#     ),
#     u"HIGH BRIGHTNESS SYMBOL": (u"\xF0\x9F\x94\x86", ur"""\U0001F506"""
#     ),
#     u"SPEAKER WITH CANCELLATION STROKE": (u"\xF0\x9F\x94\x87", ur"""\U0001F507"""
#     ),
#     u"SPEAKER WITH ONE SOUND WAVE": (u"\xF0\x9F\x94\x89", ur"""\U0001F509"""
#     ),
#     u"BELL WITH CANCELLATION STROKE": (u"\xF0\x9F\x94\x95", ur"""\U0001F515"""
#     ),
#     u"MICROSCOPE": (u"\xF0\x9F\x94\xAC", ur"""\U0001F52C"""
#     ),
#     u"TELESCOPE": (u"\xF0\x9F\x94\xAD", ur"""\U0001F52D"""
#     ),
#     u"CLOCK FACE ONE-THIRTY": (u"\xF0\x9F\x95\x9C", ur"""\U0001F55C"""
#     ),
#     u"CLOCK FACE TWO-THIRTY": (u"\xF0\x9F\x95\x9D", ur"""\U0001F55D"""
#     ),
#     u"CLOCK FACE THREE-THIRTY": (u"\xF0\x9F\x95\x9E", ur"""\U0001F55E"""
#     ),
#     u"CLOCK FACE FOUR-THIRTY": (u"\xF0\x9F\x95\x9F", ur"""\U0001F55F"""
#     ),
#     u"CLOCK FACE FIVE-THIRTY": (u"\xF0\x9F\x95\xA0", ur"""\U0001F560"""
#     ),
#     u"CLOCK FACE SIX-THIRTY": (u"\xF0\x9F\x95\xA1", ur"""\U0001F561"""
#     ),
#     u"CLOCK FACE SEVEN-THIRTY": (u"\xF0\x9F\x95\xA2", ur"""\U0001F562"""
#     ),
#     u"CLOCK FACE EIGHT-THIRTY": (u"\xF0\x9F\x95\xA3", ur"""\U0001F563"""
#     ),
#     u"CLOCK FACE NINE-THIRTY": (u"\xF0\x9F\x95\xA4", ur"""\U0001F564"""
#     ),
#     u"CLOCK FACE TEN-THIRTY": (u"\xF0\x9F\x95\xA5", ur"""\U0001F565"""
#     ),
#     u"CLOCK FACE ELEVEN-THIRTY": (u"\xF0\x9F\x95\xA6", ur"""\U0001F566"""
#     ),
#     u"CLOCK FACE TWELVE-THIRTY": (u"\xF0\x9F\x95\xA7", ur"""\U0001F567"""
#     ),

}


REGEX_FEATURE_CONFIG_VERY_SMALL = {
    # ---- Emoticons ----
#     u"Emoticon Happy": (u":-)", 
#                         r"""[:=][o-]?[)}>\]]|               # :-) :o) :)
#                         [({<\[][o-]?[:=]|                   # (-: (o: (:
#                         \^(_*|[-oO]?)\^                     # ^^ ^-^
#                         """
#     ), 
#     u"Emoticon Laughing": (u":-D", r"""([:=][-]?|x)[D]"""),   # :-D xD
#     u"Emoticon Winking": (u";-)", 
#                         r"""[;\*][-o]?[)}>\]]|              # ;-) ;o) ;)
#                         [({<\[][-o]?[;\*]                   # (-; (
#                         """
#     ), 
#     u"Emotion Tongue": (u":-P", 
#                         r"""[:=][-]?[pqP](?!\w)|            # :-P :P
#                         (?<!\w)[pqP][-]?[:=]                # q-: P-:
#                         """
#     ),  
#     u"Emoticon Surprise": (u":-O", 
#                             r"""(?<!\w|\.)                  # Boundary
#                             ([:=]-?[oO0]|                   # :-O
#                             [oO0]-?[:=]|                    # O-:
#                             [oO](_*|\.)[oO])                # Oo O____o O.o
#                             (?!\w)
#                             """
#     ), 
#     u"Emoticon Dissatisfied": (u":-/", 
#                                 r"""(?<!\w)                 # Boundary
#                                 [:=][-o]?[\/\\|]|           # :-/ :-\ :-| :/
#                                 [\/\\|][-o]?[:=]|           # \-: \:
#                                 -_+-                        # -_- -___-
#                                 """
#     ), 
    u"Emoticon Sad": (u":-(", 
                        r"""[:=][o-]?[({<\[]|               # :-( :(
                        (?<!(\w|%))                         # Boundary
                        [)}>\[][o-]?[:=]                    # )-: ): )o: 
                        """
    ), 
#     u"Emoticon Crying": (u";-(", 
#                         r"""(([:=]')|(;'?))[o-]?[({<\[]|    # ;-( :'(
#                         (?<!(\w|%))                         # Boundary
#                         [)}>\[][o-]?(('[:=])|('?;))         # )-; )-';
#                         """
#     ), 
#      
#     # ---- Punctuation----
#     # u"AllPunctuation": (u"", r"""((\.{2,}|[?!]{2,})1*)"""),
#     u"Question Mark": (u"??", r"""\?{2,}"""),                 # ??
    u"Exclamation Mark": (u"!!", r"""\!{2,}"""),              # !!
    u"Question and Exclamation Mark": (u"?!", r"""[\!\?]*((\?\!)+|              # ?!
                    (\!\?)+)[\!\?]*                         # !?
                    """
    ),                                          # Unicode interrobang: U+203D
#     u"Ellipsis": (u"...", r"""\.{2,}|                         # .. ...
#                 \.(\ \.){2,}                                # . . .
#                 """
#     ),                                          # Unicode Ellipsis: U+2026
#     # ---- Markup----
# #     u"Hashtag": (u"#", r"""\#(irony|ironic|sarcasm|sarcastic)"""),  # #irony
# #     u"Pseudo-Tag": (u"Tag", 
# #                     r"""([<\[][\/\\]
# #                     (irony|ironic|sarcasm|sarcastic)        # </irony>
# #                     [>\]])|                                 #
# #                     ((?<!(\w|[<\[]))[\/\\]                  #
# #                     (irony|ironic|sarcasm|sarcastic)        # /irony
# #                     (?![>\]]))
# #                     """
# #     ),
#  
#     # ---- Acronyms, onomatopoeia ----
    u"Acroym for Laughter": (u"lol", 
                    r"""(?<!\w)                             # Boundary
                    (l(([oua]|aw)l)+([sz]?|wut)|            # lol, lawl, luls
                    rot?fl(mf?ao)?)|                        # rofl, roflmao
                    lmf?ao                                  # lmao, lmfao
                    (?!\w)                                  # Boundary
                    """
    ),                                    
#     u"Acronym for Grin": (u"*g*", 
#                         r"""\*([Gg]{1,2}|                   # *g* *gg*
#                         grin)\*                             # *grin*
#                         """
#     ),
#     u"Onomatopoeia for Laughter": (u"haha", 
#                         r"""(?<!\w)                         # Boundary
#                         (mu|ba)?                            # mu- ba-
#                         (ha|h(e|3)|hi){2,}                  # haha, hehe, hihi
#                         (?!\w)                              # Boundary
#                         """
#     ),
    u"Interjection": (u"ITJ", 
                        r"""(?<!\w)((a+h+a?)|               # ah, aha
                        (e+h+)|                             # eh
                        (u+g?h+)|                           # ugh
                        (huh)|                              # huh
                        ([uo]h( |-)h?[uo]h)|                # uh huh, 
                        (m*hm+)                             # hmm, mhm
                        |(h(u|r)?mp(h|f))|                  # hmpf
                        (ar+gh+)|                           # argh
                        (wow+))(?!\w)                       # wow
                        """
    ),
#     u"GRINNING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x81", ur"""\U0001F601"""
#     ),
#     u"FACE WITH TEARS OF JOY": (u"\xF0\x9F\x98\x82", ur"""\U0001F602"""
#     ),
#     u"SMILING FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\x83", ur"""\U0001F603"""
#     ),
#     u"SMILING FACE WITH OPEN MOUTH AND SMILING EYES": (u"\xF0\x9F\x98\x84", ur"""\U0001F604"""
#     ),
#     u"SMILING FACE WITH OPEN MOUTH AND COLD SWEAT": (u"\xF0\x9F\x98\x85", ur"""\U0001F605"""
#     ),
    u"SMILING FACE WITH OPEN MOUTH AND TIGHTLY-CLOSED EYES": (u"\xF0\x9F\x98\x86", ur"""\U0001F606"""
    ),
#     u"WINKING FACE": (u"\xF0\x9F\x98\x89", ur"""\U0001F609"""
#     ),
#     u"SMILING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x8A", ur"""\U0001F60A"""
#     ),
#     u"FACE SAVOURING DELICIOUS FOOD": (u"\xF0\x9F\x98\x8B", ur"""\U0001F60B"""
#     ),
#     u"RELIEVED FACE": (u"\xF0\x9F\x98\x8C", ur"""\U0001F60C"""
#     ),
#     u"SMILING FACE WITH HEART-SHAPED EYES": (u"\xF0\x9F\x98\x8D", ur"""\U0001F60D"""
#     ),
#     u"SMIRKING FACE": (u"\xF0\x9F\x98\x8F", ur"""\U0001F60F"""
#     ),
#     u"UNAMUSED FACE": (u"\xF0\x9F\x98\x92", ur"""\U0001F612"""
#     ),
#     u"FACE WITH COLD SWEAT": (u"\xF0\x9F\x98\x93", ur"""\U0001F613"""
#     ),
#     u"PENSIVE FACE": (u"\xF0\x9F\x98\x94", ur"""\U0001F614"""
#     ),
#     u"CONFOUNDED FACE": (u"\xF0\x9F\x98\x96", ur"""\U0001F616"""
#     ),
#     u"FACE THROWING A KISS": (u"\xF0\x9F\x98\x98", ur"""\U0001F618"""
#     ),
#     u"KISSING FACE WITH CLOSED EYES": (u"\xF0\x9F\x98\x9A", ur"""\U0001F61A"""
#     ),
#     u"FACE WITH STUCK-OUT TONGUE AND WINKING EYE": (u"\xF0\x9F\x98\x9C", ur"""\U0001F61C"""
#     ),
#     u"FACE WITH STUCK-OUT TONGUE AND TIGHTLY-CLOSED EYES": (u"\xF0\x9F\x98\x9D", ur"""\U0001F61D"""
#     ),
#     u"DISAPPOINTED FACE": (u"\xF0\x9F\x98\x9E", ur"""\U0001F61E"""
#     ),
#     u"ANGRY FACE": (u"\xF0\x9F\x98\xA0", ur"""\U0001F620"""
#     ),
#     u"POUTING FACE": (u"\xF0\x9F\x98\xA1", ur"""\U0001F621"""
#     ),
#     u"CRYING FACE": (u"\xF0\x9F\x98\xA2", ur"""\U0001F622"""
#     ),
#     u"PERSEVERING FACE": (u"\xF0\x9F\x98\xA3", ur"""\U0001F623"""
#     ),
#     u"FACE WITH LOOK OF TRIUMPH": (u"\xF0\x9F\x98\xA4", ur"""\U0001F624"""
#     ),
#     u"DISAPPOINTED BUT RELIEVED FACE": (u"\xF0\x9F\x98\xA5", ur"""\U0001F625"""
#     ),
#     u"FEARFUL FACE": (u"\xF0\x9F\x98\xA8", ur"""\U0001F628"""
#     ),
#     u"WEARY FACE": (u"\xF0\x9F\x98\xA9", ur"""\U0001F629"""
#     ),
#     u"SLEEPY FACE": (u"\xF0\x9F\x98\xAA", ur"""\U0001F62A"""
#     ),
#     u"TIRED FACE": (u"\xF0\x9F\x98\xAB", ur"""\U0001F62B"""
#     ),
#     u"LOUDLY CRYING FACE": (u"\xF0\x9F\x98\xAD", ur"""\U0001F62D"""
#     ),
#     u"FACE WITH OPEN MOUTH AND COLD SWEAT": (u"\xF0\x9F\x98\xB0", ur"""\U0001F630"""
#     ),
#     u"FACE SCREAMING IN FEAR": (u"\xF0\x9F\x98\xB1", ur"""\U0001F631"""
#     ),
#     u"ASTONISHED FACE": (u"\xF0\x9F\x98\xB2", ur"""\U0001F632"""
#     ),
#     u"FLUSHED FACE": (u"\xF0\x9F\x98\xB3", ur"""\U0001F633"""
#     ),
#     u"DIZZY FACE": (u"\xF0\x9F\x98\xB5", ur"""\U0001F635"""
#     ),
#     u"FACE WITH MEDICAL MASK": (u"\xF0\x9F\x98\xB7", ur"""\U0001F637"""
#     ),
#     u"GRINNING CAT FACE WITH SMILING EYES": (u"\xF0\x9F\x98\xB8", ur"""\U0001F638"""
#     ),
#     u"CAT FACE WITH TEARS OF JOY": (u"\xF0\x9F\x98\xB9", ur"""\U0001F639"""
#     ),
#     u"SMILING CAT FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xBA", ur"""\U0001F63A"""
#     ),
#     u"SMILING CAT FACE WITH HEART-SHAPED EYES": (u"\xF0\x9F\x98\xBB", ur"""\U0001F63B"""
#     ),
#     u"CAT FACE WITH WRY SMILE": (u"\xF0\x9F\x98\xBC", ur"""\U0001F63C"""
#     ),
#     u"KISSING CAT FACE WITH CLOSED EYES": (u"\xF0\x9F\x98\xBD", ur"""\U0001F63D"""
#     ),
#     u"POUTING CAT FACE": (u"\xF0\x9F\x98\xBE", ur"""\U0001F63E"""
#     ),
#     u"CRYING CAT FACE": (u"\xF0\x9F\x98\xBF", ur"""\U0001F63F"""
#     ),
#     u"WEARY CAT FACE": (u"\xF0\x9F\x99\x80", ur"""\U0001F640"""
#     ),
#     u"FACE WITH NO GOOD GESTURE": (u"\xF0\x9F\x99\x85", ur"""\U0001F645"""
#     ),
#     u"FACE WITH OK GESTURE": (u"\xF0\x9F\x99\x86", ur"""\U0001F646"""
#     ),
#     u"PERSON BOWING DEEPLY": (u"\xF0\x9F\x99\x87", ur"""\U0001F647"""
#     ),
#     u"SEE-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x88", ur"""\U0001F648"""
#     ),
#     u"HEAR-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x89", ur"""\U0001F649"""
#     ),
#     u"SPEAK-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x8A", ur"""\U0001F64A"""
#     ),
#     u"HAPPY PERSON RAISING ONE HAND": (u"\xF0\x9F\x99\x8B", ur"""\U0001F64B"""
#     ),
#     u"PERSON RAISING BOTH HANDS IN CELEBRATION": (u"\xF0\x9F\x99\x8C", ur"""\U0001F64C"""
#     ),
#     u"PERSON FROWNING": (u"\xF0\x9F\x99\x8D", ur"""\U0001F64D"""
#     ),
#     u"PERSON WITH POUTING FACE": (u"\xF0\x9F\x99\x8E", ur"""\U0001F64E"""
#     ),
#     u"PERSON WITH FOLDED HANDS": (u"\xF0\x9F\x99\x8F", ur"""\U0001F64F"""
#     ),
#     u"BLACK SCISSORS": (u"\xE2\x9C\x82", ur"""\U00002702"""
#     ),
#     u"WHITE HEAVY CHECK MARK": (u"\xE2\x9C\x85", ur"""\U00002705"""
#     ),
#     u"AIRPLANE": (u"\xE2\x9C\x88", ur"""\U00002708"""
#     ),
#     u"ENVELOPE": (u"\xE2\x9C\x89", ur"""\U00002709"""
#     ),
#     u"RAISED FIST": (u"\xE2\x9C\x8A", ur"""\U0000270A"""
#     ),
#     u"RAISED HAND": (u"\xE2\x9C\x8B", ur"""\U0000270B"""
#     ),
#     u"VICTORY HAND": (u"\xE2\x9C\x8C", ur"""\U0000270C"""
#     ),
#     u"PENCIL": (u"\xE2\x9C\x8F", ur"""\U0000270F"""
#     ),
#     u"BLACK NIB": (u"\xE2\x9C\x92", ur"""\U00002712"""
#     ),
#     u"HEAVY CHECK MARK": (u"\xE2\x9C\x94", ur"""\U00002714"""
#     ),
#     u"HEAVY MULTIPLICATION X": (u"\xE2\x9C\x96", ur"""\U00002716"""
#     ),
#     u"SPARKLES": (u"\xE2\x9C\xA8", ur"""\U00002728"""
#     ),
#     u"EIGHT SPOKED ASTERISK": (u"\xE2\x9C\xB3", ur"""\U00002733"""
#     ),
#     u"EIGHT POINTED BLACK STAR": (u"\xE2\x9C\xB4", ur"""\U00002734"""
#     ),
#     u"SNOWFLAKE": (u"\xE2\x9D\x84", ur"""\U00002744"""
#     ),
#     u"SPARKLE": (u"\xE2\x9D\x87", ur"""\U00002747"""
#     ),
#     u"CROSS MARK": (u"\xE2\x9D\x8C", ur"""\U0000274C"""
#     ),
#     u"NEGATIVE SQUARED CROSS MARK": (u"\xE2\x9D\x8E", ur"""\U0000274E"""
#     ),
#     u"BLACK QUESTION MARK ORNAMENT": (u"\xE2\x9D\x93", ur"""\U00002753"""
#     ),
#     u"WHITE QUESTION MARK ORNAMENT": (u"\xE2\x9D\x94", ur"""\U00002754"""
#     ),
#     u"WHITE EXCLAMATION MARK ORNAMENT": (u"\xE2\x9D\x95", ur"""\U00002755"""
#     ),
#     u"HEAVY EXCLAMATION MARK SYMBOL": (u"\xE2\x9D\x97", ur"""\U00002757"""
#     ),
#     u"HEAVY BLACK HEART": (u"\xE2\x9D\xA4", ur"""\U00002764"""
#     ),
#     u"HEAVY PLUS SIGN": (u"\xE2\x9E\x95", ur"""\U00002795"""
#     ),
#     u"HEAVY MINUS SIGN": (u"\xE2\x9E\x96", ur"""\U00002796"""
#     ),
#     u"HEAVY DIVISION SIGN": (u"\xE2\x9E\x97", ur"""\U00002797"""
#     ),
#     u"BLACK RIGHTWARDS ARROW": (u"\xE2\x9E\xA1", ur"""\U000027A1"""
#     ),
#     u"CURLY LOOP": (u"\xE2\x9E\xB0", ur"""\U000027B0"""
#     ),
#     u"ROCKET": (u"\xF0\x9F\x9A\x80", ur"""\U0001F680"""
#     ),
#     u"RAILWAY CAR": (u"\xF0\x9F\x9A\x83", ur"""\U0001F683"""
#     ),
#     u"HIGH-SPEED TRAIN": (u"\xF0\x9F\x9A\x84", ur"""\U0001F684"""
#     ),
#     u"HIGH-SPEED TRAIN WITH BULLET NOSE": (u"\xF0\x9F\x9A\x85", ur"""\U0001F685"""
#     ),
#     u"METRO": (u"\xF0\x9F\x9A\x87", ur"""\U0001F687"""
#     ),
#     u"STATION": (u"\xF0\x9F\x9A\x89", ur"""\U0001F689"""
#     ),
#     u"BUS": (u"\xF0\x9F\x9A\x8C", ur"""\U0001F68C"""
#     ),
#     u"BUS STOP": (u"\xF0\x9F\x9A\x8F", ur"""\U0001F68F"""
#     ),
#     u"AMBULANCE": (u"\xF0\x9F\x9A\x91", ur"""\U0001F691"""
#     ),
#     u"FIRE ENGINE": (u"\xF0\x9F\x9A\x92", ur"""\U0001F692"""
#     ),
#     u"POLICE CAR": (u"\xF0\x9F\x9A\x93", ur"""\U0001F693"""
#     ),
#     u"TAXI": (u"\xF0\x9F\x9A\x95", ur"""\U0001F695"""
#     ),
#     u"AUTOMOBILE": (u"\xF0\x9F\x9A\x97", ur"""\U0001F697"""
#     ),
#     u"RECREATIONAL VEHICLE": (u"\xF0\x9F\x9A\x99", ur"""\U0001F699"""
#     ),
#     u"DELIVERY TRUCK": (u"\xF0\x9F\x9A\x9A", ur"""\U0001F69A"""
#     ),
#     u"SHIP": (u"\xF0\x9F\x9A\xA2", ur"""\U0001F6A2"""
#     ),
#     u"SPEEDBOAT": (u"\xF0\x9F\x9A\xA4", ur"""\U0001F6A4"""
#     ),
#     u"HORIZONTAL TRAFFIC LIGHT": (u"\xF0\x9F\x9A\xA5", ur"""\U0001F6A5"""
#     ),
#     u"CONSTRUCTION SIGN": (u"\xF0\x9F\x9A\xA7", ur"""\U0001F6A7"""
#     ),
#     u"POLICE CARS REVOLVING LIGHT": (u"\xF0\x9F\x9A\xA8", ur"""\U0001F6A8"""
#     ),
#     u"TRIANGULAR FLAG ON POST": (u"\xF0\x9F\x9A\xA9", ur"""\U0001F6A9"""
#     ),
#     u"DOOR": (u"\xF0\x9F\x9A\xAA", ur"""\U0001F6AA"""
#     ),
#     u"NO ENTRY SIGN": (u"\xF0\x9F\x9A\xAB", ur"""\U0001F6AB"""
#     ),
#     u"SMOKING SYMBOL": (u"\xF0\x9F\x9A\xAC", ur"""\U0001F6AC"""
#     ),
#     u"NO SMOKING SYMBOL": (u"\xF0\x9F\x9A\xAD", ur"""\U0001F6AD"""
#     ),
#     u"BICYCLE": (u"\xF0\x9F\x9A\xB2", ur"""\U0001F6B2"""
#     ),
#     u"PEDESTRIAN": (u"\xF0\x9F\x9A\xB6", ur"""\U0001F6B6"""
#     ),
#     u"MENS SYMBOL": (u"\xF0\x9F\x9A\xB9", ur"""\U0001F6B9"""
#     ),
#     u"WOMENS SYMBOL": (u"\xF0\x9F\x9A\xBA", ur"""\U0001F6BA"""
#     ),
#     u"RESTROOM": (u"\xF0\x9F\x9A\xBB", ur"""\U0001F6BB"""
#     ),
#     u"BABY SYMBOL": (u"\xF0\x9F\x9A\xBC", ur"""\U0001F6BC"""
#     ),
#     u"TOILET": (u"\xF0\x9F\x9A\xBD", ur"""\U0001F6BD"""
#     ),
#     u"WATER CLOSET": (u"\xF0\x9F\x9A\xBE", ur"""\U0001F6BE"""
#     ),
#     u"BATH": (u"\xF0\x9F\x9B\x80", ur"""\U0001F6C0"""
#     ),
#     u"CIRCLED LATIN CAPITAL LETTER M": (u"\xE2\x93\x82", ur"""\U000024C2"""
#     ),
    u"NEGATIVE SQUARED LATIN CAPITAL LETTER A": (u"\xF0\x9F\x85\xB0", ur"""\U0001F170"""
    ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER B": (u"\xF0\x9F\x85\xB1", ur"""\U0001F171"""
#     ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER O": (u"\xF0\x9F\x85\xBE", ur"""\U0001F17E"""
#     ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER P": (u"\xF0\x9F\x85\xBF", ur"""\U0001F17F"""
#     ),
#     u"NEGATIVE SQUARED AB": (u"\xF0\x9F\x86\x8E", ur"""\U0001F18E"""
#     ),
#     u"SQUARED CL": (u"\xF0\x9F\x86\x91", ur"""\U0001F191"""
#     ),
#     u"SQUARED COOL": (u"\xF0\x9F\x86\x92", ur"""\U0001F192"""
#     ),
#     u"SQUARED FREE": (u"\xF0\x9F\x86\x93", ur"""\U0001F193"""
#     ),
#     u"SQUARED ID": (u"\xF0\x9F\x86\x94", ur"""\U0001F194"""
#     ),
#     u"SQUARED NEW": (u"\xF0\x9F\x86\x95", ur"""\U0001F195"""
#     ),
#     u"SQUARED NG": (u"\xF0\x9F\x86\x96", ur"""\U0001F196"""
#     ),
#     u"SQUARED OK": (u"\xF0\x9F\x86\x97", ur"""\U0001F197"""
#     ),
#     u"SQUARED SOS": (u"\xF0\x9F\x86\x98", ur"""\U0001F198"""
#     ),
#     u"SQUARED UP WITH EXCLAMATION MARK": (u"\xF0\x9F\x86\x99", ur"""\U0001F199"""
#     ),
#     u"SQUARED VS": (u"\xF0\x9F\x86\x9A", ur"""\U0001F19A"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER D + REGIONAL INDICATOR SYMBOL LETTER E": (u"\xF0\x9F\x87\xA9\xF0\x9F\x87\xAA", ur"""\U0001F1E9 \U0001F1EA"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER G + REGIONAL INDICATOR SYMBOL LETTER B": (u"\xF0\x9F\x87\xAC\xF0\x9F\x87\xA7", ur"""\U0001F1EC \U0001F1E7"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER C + REGIONAL INDICATOR SYMBOL LETTER N": (u"\xF0\x9F\x87\xA8\xF0\x9F\x87\xB3", ur"""\U0001F1E8 \U0001F1F3"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER J + REGIONAL INDICATOR SYMBOL LETTER P": (u"\xF0\x9F\x87\xAF\xF0\x9F\x87\xB5", ur"""\U0001F1EF \U0001F1F5"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER K + REGIONAL INDICATOR SYMBOL LETTER R": (u"\xF0\x9F\x87\xB0\xF0\x9F\x87\xB7", ur"""\U0001F1F0 \U0001F1F7"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER F + REGIONAL INDICATOR SYMBOL LETTER R": (u"\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7", ur"""\U0001F1EB \U0001F1F7"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER E + REGIONAL INDICATOR SYMBOL LETTER S": (u"\xF0\x9F\x87\xAA\xF0\x9F\x87\xB8", ur"""\U0001F1EA \U0001F1F8"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER I + REGIONAL INDICATOR SYMBOL LETTER T": (u"\xF0\x9F\x87\xAE\xF0\x9F\x87\xB9", ur"""\U0001F1EE \U0001F1F9"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER U + REGIONAL INDICATOR SYMBOL LETTER S": (u"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8", ur"""\U0001F1FA \U0001F1F8"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER R + REGIONAL INDICATOR SYMBOL LETTER U": (u"\xF0\x9F\x87\xB7\xF0\x9F\x87\xBA", ur"""\U0001F1F7 \U0001F1FA"""
#     ),
#     u"SQUARED KATAKANA KOKO": (u"\xF0\x9F\x88\x81", ur"""\U0001F201"""
#     ),
#     u"SQUARED KATAKANA SA": (u"\xF0\x9F\x88\x82", ur"""\U0001F202"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7121": (u"\xF0\x9F\x88\x9A", ur"""\U0001F21A"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6307": (u"\xF0\x9F\x88\xAF", ur"""\U0001F22F"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7981": (u"\xF0\x9F\x88\xB2", ur"""\U0001F232"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7A7A": (u"\xF0\x9F\x88\xB3", ur"""\U0001F233"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-5408": (u"\xF0\x9F\x88\xB4", ur"""\U0001F234"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6E80": (u"\xF0\x9F\x88\xB5", ur"""\U0001F235"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6709": (u"\xF0\x9F\x88\xB6", ur"""\U0001F236"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6708": (u"\xF0\x9F\x88\xB7", ur"""\U0001F237"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7533": (u"\xF0\x9F\x88\xB8", ur"""\U0001F238"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-5272": (u"\xF0\x9F\x88\xB9", ur"""\U0001F239"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-55B6": (u"\xF0\x9F\x88\xBA", ur"""\U0001F23A"""
#     ),
#     u"CIRCLED IDEOGRAPH ADVANTAGE": (u"\xF0\x9F\x89\x90", ur"""\U0001F250"""
#     ),
#     u"CIRCLED IDEOGRAPH ACCEPT": (u"\xF0\x9F\x89\x91", ur"""\U0001F251"""
#     ),
#     u"COPYRIGHT SIGN": (u"\xC2\xA9", ur"""\U000000A9"""
#     ),
#     u"REGISTERED SIGN": (u"\xC2\xAE", ur"""\U000000AE"""
#     ),
#     u"DOUBLE EXCLAMATION MARK": (u"\xE2\x80\xBC", ur"""\U0000203C"""
#     ),
#     u"EXCLAMATION QUESTION MARK": (u"\xE2\x81\x89", ur"""\U00002049"""
#     ),
#     u"DIGIT EIGHT + COMBINING ENCLOSING KEYCAP": (u"\x38\xE2\x83\xA3", ur"""\U00000038 \U000020E3"""
#     ),
#     u"DIGIT NINE + COMBINING ENCLOSING KEYCAP": (u"\x39\xE2\x83\xA3", ur"""\U00000039 \U000020E3"""
#     ),
#     u"DIGIT SEVEN + COMBINING ENCLOSING KEYCAP": (u"\x37\xE2\x83\xA3", ur"""\U00000037 \U000020E3"""
#     ),
#     u"DIGIT SIX + COMBINING ENCLOSING KEYCAP": (u"\x36\xE2\x83\xA3", ur"""\U00000036 \U000020E3"""
#     ),
#     u"DIGIT ONE + COMBINING ENCLOSING KEYCAP": (u"\x31\xE2\x83\xA3", ur"""\U00000031 \U000020E3"""
#     ),
#     u"DIGIT ZERO + COMBINING ENCLOSING KEYCAP": (u"\x30\xE2\x83\xA3", ur"""\U00000030 \U000020E3"""
#     ),
#     u"DIGIT TWO + COMBINING ENCLOSING KEYCAP": (u"\x32\xE2\x83\xA3", ur"""\U00000032 \U000020E3"""
#     ),
#     u"DIGIT THREE + COMBINING ENCLOSING KEYCAP": (u"\x33\xE2\x83\xA3", ur"""\U00000033 \U000020E3"""
#     ),
#     u"DIGIT FIVE + COMBINING ENCLOSING KEYCAP": (u"\x35\xE2\x83\xA3", ur"""\U00000035 \U000020E3"""
#     ),
#     u"DIGIT FOUR + COMBINING ENCLOSING KEYCAP": (u"\x34\xE2\x83\xA3", ur"""\U00000034 \U000020E3"""
#     ),
#     u"NUMBER SIGN + COMBINING ENCLOSING KEYCAP": (u"\x23\xE2\x83\xA3", ur"""\U00000023 \U000020E3"""
#     ),
#     u"TRADE MARK SIGN": (u"\xE2\x84\xA2", ur"""\U00002122"""
#     ),
#     u"INFORMATION SOURCE": (u"\xE2\x84\xB9", ur"""\U00002139"""
#     ),
#     u"LEFT RIGHT ARROW": (u"\xE2\x86\x94", ur"""\U00002194"""
#     ),
#     u"UP DOWN ARROW": (u"\xE2\x86\x95", ur"""\U00002195"""
#     ),
#     u"NORTH WEST ARROW": (u"\xE2\x86\x96", ur"""\U00002196"""
#     ),
#     u"NORTH EAST ARROW": (u"\xE2\x86\x97", ur"""\U00002197"""
#     ),
#     u"SOUTH EAST ARROW": (u"\xE2\x86\x98", ur"""\U00002198"""
#     ),
#     u"SOUTH WEST ARROW": (u"\xE2\x86\x99", ur"""\U00002199"""
#     ),
#     u"LEFTWARDS ARROW WITH HOOK": (u"\xE2\x86\xA9", ur"""\U000021A9"""
#     ),
#     u"RIGHTWARDS ARROW WITH HOOK": (u"\xE2\x86\xAA", ur"""\U000021AA"""
#     ),
#     u"WATCH": (u"\xE2\x8C\x9A", ur"""\U0000231A"""
#     ),
#     u"HOURGLASS": (u"\xE2\x8C\x9B", ur"""\U0000231B"""
#     ),
#     u"BLACK RIGHT-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xA9", ur"""\U000023E9"""
#     ),
#     u"BLACK LEFT-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAA", ur"""\U000023EA"""
#     ),
#     u"BLACK UP-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAB", ur"""\U000023EB"""
#     ),
#     u"BLACK DOWN-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAC", ur"""\U000023EC"""
#     ),
#     u"ALARM CLOCK": (u"\xE2\x8F\xB0", ur"""\U000023F0"""
#     ),
#     u"HOURGLASS WITH FLOWING SAND": (u"\xE2\x8F\xB3", ur"""\U000023F3"""
#     ),
#     u"BLACK SMALL SQUARE": (u"\xE2\x96\xAA", ur"""\U000025AA"""
#     ),
#     u"WHITE SMALL SQUARE": (u"\xE2\x96\xAB", ur"""\U000025AB"""
#     ),
#     u"BLACK RIGHT-POINTING TRIANGLE": (u"\xE2\x96\xB6", ur"""\U000025B6"""
#     ),
#     u"BLACK LEFT-POINTING TRIANGLE": (u"\xE2\x97\x80", ur"""\U000025C0"""
#     ),
#     u"WHITE MEDIUM SQUARE": (u"\xE2\x97\xBB", ur"""\U000025FB"""
#     ),
#     u"BLACK MEDIUM SQUARE": (u"\xE2\x97\xBC", ur"""\U000025FC"""
#     ),
#     u"WHITE MEDIUM SMALL SQUARE": (u"\xE2\x97\xBD", ur"""\U000025FD"""
#     ),
#     u"BLACK MEDIUM SMALL SQUARE": (u"\xE2\x97\xBE", ur"""\U000025FE"""
#     ),
#     u"BLACK SUN WITH RAYS": (u"\xE2\x98\x80", ur"""\U00002600"""
#     ),
#     u"CLOUD": (u"\xE2\x98\x81", ur"""\U00002601"""
#     ),
#     u"BLACK TELEPHONE": (u"\xE2\x98\x8E", ur"""\U0000260E"""
#     ),
#     u"BALLOT BOX WITH CHECK": (u"\xE2\x98\x91", ur"""\U00002611"""
#     ),
#     u"UMBRELLA WITH RAIN DROPS": (u"\xE2\x98\x94", ur"""\U00002614"""
#     ),
#     u"HOT BEVERAGE": (u"\xE2\x98\x95", ur"""\U00002615"""
#     ),
#     u"WHITE UP POINTING INDEX": (u"\xE2\x98\x9D", ur"""\U0000261D"""
#     ),
#     u"WHITE SMILING FACE": (u"\xE2\x98\xBA", ur"""\U0000263A"""
#     ),
#     u"ARIES": (u"\xE2\x99\x88", ur"""\U00002648"""
#     ),
#     u"TAURUS": (u"\xE2\x99\x89", ur"""\U00002649"""
#     ),
#     u"GEMINI": (u"\xE2\x99\x8A", ur"""\U0000264A"""
#     ),
#     u"CANCER": (u"\xE2\x99\x8B", ur"""\U0000264B"""
#     ),
#     u"LEO": (u"\xE2\x99\x8C", ur"""\U0000264C"""
#     ),
#     u"VIRGO": (u"\xE2\x99\x8D", ur"""\U0000264D"""
#     ),
#     u"LIBRA": (u"\xE2\x99\x8E", ur"""\U0000264E"""
#     ),
#     u"SCORPIUS": (u"\xE2\x99\x8F", ur"""\U0000264F"""
#     ),
#     u"SAGITTARIUS": (u"\xE2\x99\x90", ur"""\U00002650"""
#     ),
#     u"CAPRICORN": (u"\xE2\x99\x91", ur"""\U00002651"""
#     ),
#     u"AQUARIUS": (u"\xE2\x99\x92", ur"""\U00002652"""
#     ),
#     u"PISCES": (u"\xE2\x99\x93", ur"""\U00002653"""
#     ),
#     u"BLACK SPADE SUIT": (u"\xE2\x99\xA0", ur"""\U00002660"""
#     ),
#     u"BLACK CLUB SUIT": (u"\xE2\x99\xA3", ur"""\U00002663"""
#     ),
#     u"BLACK HEART SUIT": (u"\xE2\x99\xA5", ur"""\U00002665"""
#     ),
#     u"BLACK DIAMOND SUIT": (u"\xE2\x99\xA6", ur"""\U00002666"""
#     ),
#     u"HOT SPRINGS": (u"\xE2\x99\xA8", ur"""\U00002668"""
#     ),
#     u"BLACK UNIVERSAL RECYCLING SYMBOL": (u"\xE2\x99\xBB", ur"""\U0000267B"""
#     ),
#     u"WHEELCHAIR SYMBOL": (u"\xE2\x99\xBF", ur"""\U0000267F"""
#     ),
#     u"ANCHOR": (u"\xE2\x9A\x93", ur"""\U00002693"""
#     ),
#     u"WARNING SIGN": (u"\xE2\x9A\xA0", ur"""\U000026A0"""
#     ),
#     u"HIGH VOLTAGE SIGN": (u"\xE2\x9A\xA1", ur"""\U000026A1"""
#     ),
#     u"MEDIUM WHITE CIRCLE": (u"\xE2\x9A\xAA", ur"""\U000026AA"""
#     ),
#     u"MEDIUM BLACK CIRCLE": (u"\xE2\x9A\xAB", ur"""\U000026AB"""
#     ),
#     u"SOCCER BALL": (u"\xE2\x9A\xBD", ur"""\U000026BD"""
#     ),
#     u"BASEBALL": (u"\xE2\x9A\xBE", ur"""\U000026BE"""
#     ),
#     u"SNOWMAN WITHOUT SNOW": (u"\xE2\x9B\x84", ur"""\U000026C4"""
#     ),
#     u"SUN BEHIND CLOUD": (u"\xE2\x9B\x85", ur"""\U000026C5"""
#     ),
#     u"OPHIUCHUS": (u"\xE2\x9B\x8E", ur"""\U000026CE"""
#     ),
#     u"NO ENTRY": (u"\xE2\x9B\x94", ur"""\U000026D4"""
#     ),
#     u"CHURCH": (u"\xE2\x9B\xAA", ur"""\U000026EA"""
#     ),
#     u"FOUNTAIN": (u"\xE2\x9B\xB2", ur"""\U000026F2"""
#     ),
#     u"FLAG IN HOLE": (u"\xE2\x9B\xB3", ur"""\U000026F3"""
#     ),
#     u"SAILBOAT": (u"\xE2\x9B\xB5", ur"""\U000026F5"""
#     ),
#     u"TENT": (u"\xE2\x9B\xBA", ur"""\U000026FA"""
#     ),
#     u"FUEL PUMP": (u"\xE2\x9B\xBD", ur"""\U000026FD"""
#     ),
#     u"ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS": (u"\xE2\xA4\xB4", ur"""\U00002934"""
#     ),
#     u"ARROW POINTING RIGHTWARDS THEN CURVING DOWNWARDS": (u"\xE2\xA4\xB5", ur"""\U00002935"""
#     ),
#     u"LEFTWARDS BLACK ARROW": (u"\xE2\xAC\x85", ur"""\U00002B05"""
#     ),
#     u"UPWARDS BLACK ARROW": (u"\xE2\xAC\x86", ur"""\U00002B06"""
#     ),
#     u"DOWNWARDS BLACK ARROW": (u"\xE2\xAC\x87", ur"""\U00002B07"""
#     ),
#     u"BLACK LARGE SQUARE": (u"\xE2\xAC\x9B", ur"""\U00002B1B"""
#     ),
#     u"WHITE LARGE SQUARE": (u"\xE2\xAC\x9C", ur"""\U00002B1C"""
#     ),
#     u"WHITE MEDIUM STAR": (u"\xE2\xAD\x90", ur"""\U00002B50"""
#     ),
#     u"HEAVY LARGE CIRCLE": (u"\xE2\xAD\x95", ur"""\U00002B55"""
#     ),
#     u"WAVY DASH": (u"\xE3\x80\xB0", ur"""\U00003030"""
#     ),
#     u"PART ALTERNATION MARK": (u"\xE3\x80\xBD", ur"""\U0000303D"""
#     ),
#     u"CIRCLED IDEOGRAPH CONGRATULATION": (u"\xE3\x8A\x97", ur"""\U00003297"""
#     ),
#     u"CIRCLED IDEOGRAPH SECRET": (u"\xE3\x8A\x99", ur"""\U00003299"""
#     ),
#     u"MAHJONG TILE RED DRAGON": (u"\xF0\x9F\x80\x84", ur"""\U0001F004"""
#     ),
#     u"PLAYING CARD BLACK JOKER": (u"\xF0\x9F\x83\x8F", ur"""\U0001F0CF"""
#     ),
#     u"CYCLONE": (u"\xF0\x9F\x8C\x80", ur"""\U0001F300"""
#     ),
#     u"FOGGY": (u"\xF0\x9F\x8C\x81", ur"""\U0001F301"""
#     ),
#     u"CLOSED UMBRELLA": (u"\xF0\x9F\x8C\x82", ur"""\U0001F302"""
#     ),
#     u"NIGHT WITH STARS": (u"\xF0\x9F\x8C\x83", ur"""\U0001F303"""
#     ),
#     u"SUNRISE OVER MOUNTAINS": (u"\xF0\x9F\x8C\x84", ur"""\U0001F304"""
#     ),
#     u"SUNRISE": (u"\xF0\x9F\x8C\x85", ur"""\U0001F305"""
#     ),
#     u"CITYSCAPE AT DUSK": (u"\xF0\x9F\x8C\x86", ur"""\U0001F306"""
#     ),
#     u"SUNSET OVER BUILDINGS": (u"\xF0\x9F\x8C\x87", ur"""\U0001F307"""
#     ),
#     u"RAINBOW": (u"\xF0\x9F\x8C\x88", ur"""\U0001F308"""
#     ),
#     u"BRIDGE AT NIGHT": (u"\xF0\x9F\x8C\x89", ur"""\U0001F309"""
#     ),
#     u"WATER WAVE": (u"\xF0\x9F\x8C\x8A", ur"""\U0001F30A"""
#     ),
#     u"VOLCANO": (u"\xF0\x9F\x8C\x8B", ur"""\U0001F30B"""
#     ),
#     u"MILKY WAY": (u"\xF0\x9F\x8C\x8C", ur"""\U0001F30C"""
#     ),
#     u"EARTH GLOBE ASIA-AUSTRALIA": (u"\xF0\x9F\x8C\x8F", ur"""\U0001F30F"""
#     ),
#     u"NEW MOON SYMBOL": (u"\xF0\x9F\x8C\x91", ur"""\U0001F311"""
#     ),
#     u"FIRST QUARTER MOON SYMBOL": (u"\xF0\x9F\x8C\x93", ur"""\U0001F313"""
#     ),
#     u"WAXING GIBBOUS MOON SYMBOL": (u"\xF0\x9F\x8C\x94", ur"""\U0001F314"""
#     ),
#     u"FULL MOON SYMBOL": (u"\xF0\x9F\x8C\x95", ur"""\U0001F315"""
#     ),
#     u"CRESCENT MOON": (u"\xF0\x9F\x8C\x99", ur"""\U0001F319"""
#     ),
#     u"FIRST QUARTER MOON WITH FACE": (u"\xF0\x9F\x8C\x9B", ur"""\U0001F31B"""
#     ),
#     u"GLOWING STAR": (u"\xF0\x9F\x8C\x9F", ur"""\U0001F31F"""
#     ),
#     u"SHOOTING STAR": (u"\xF0\x9F\x8C\xA0", ur"""\U0001F320"""
#     ),
#     u"CHESTNUT": (u"\xF0\x9F\x8C\xB0", ur"""\U0001F330"""
#     ),
#     u"SEEDLING": (u"\xF0\x9F\x8C\xB1", ur"""\U0001F331"""
#     ),
#     u"PALM TREE": (u"\xF0\x9F\x8C\xB4", ur"""\U0001F334"""
#     ),
#     u"CACTUS": (u"\xF0\x9F\x8C\xB5", ur"""\U0001F335"""
#     ),
#     u"TULIP": (u"\xF0\x9F\x8C\xB7", ur"""\U0001F337"""
#     ),
#     u"CHERRY BLOSSOM": (u"\xF0\x9F\x8C\xB8", ur"""\U0001F338"""
#     ),
#     u"ROSE": (u"\xF0\x9F\x8C\xB9", ur"""\U0001F339"""
#     ),
#     u"HIBISCUS": (u"\xF0\x9F\x8C\xBA", ur"""\U0001F33A"""
#     ),
#     u"SUNFLOWER": (u"\xF0\x9F\x8C\xBB", ur"""\U0001F33B"""
#     ),
#     u"BLOSSOM": (u"\xF0\x9F\x8C\xBC", ur"""\U0001F33C"""
#     ),
#     u"EAR OF MAIZE": (u"\xF0\x9F\x8C\xBD", ur"""\U0001F33D"""
#     ),
#     u"EAR OF RICE": (u"\xF0\x9F\x8C\xBE", ur"""\U0001F33E"""
#     ),
#     u"HERB": (u"\xF0\x9F\x8C\xBF", ur"""\U0001F33F"""
#     ),
#     u"FOUR LEAF CLOVER": (u"\xF0\x9F\x8D\x80", ur"""\U0001F340"""
#     ),
#     u"MAPLE LEAF": (u"\xF0\x9F\x8D\x81", ur"""\U0001F341"""
#     ),
#     u"FALLEN LEAF": (u"\xF0\x9F\x8D\x82", ur"""\U0001F342"""
#     ),
#     u"LEAF FLUTTERING IN WIND": (u"\xF0\x9F\x8D\x83", ur"""\U0001F343"""
#     ),
#     u"MUSHROOM": (u"\xF0\x9F\x8D\x84", ur"""\U0001F344"""
#     ),
#     u"TOMATO": (u"\xF0\x9F\x8D\x85", ur"""\U0001F345"""
#     ),
#     u"AUBERGINE": (u"\xF0\x9F\x8D\x86", ur"""\U0001F346"""
#     ),
#     u"GRAPES": (u"\xF0\x9F\x8D\x87", ur"""\U0001F347"""
#     ),
#     u"MELON": (u"\xF0\x9F\x8D\x88", ur"""\U0001F348"""
#     ),
#     u"WATERMELON": (u"\xF0\x9F\x8D\x89", ur"""\U0001F349"""
#     ),
#     u"TANGERINE": (u"\xF0\x9F\x8D\x8A", ur"""\U0001F34A"""
#     ),
#     u"BANANA": (u"\xF0\x9F\x8D\x8C", ur"""\U0001F34C"""
#     ),
#     u"PINEAPPLE": (u"\xF0\x9F\x8D\x8D", ur"""\U0001F34D"""
#     ),
#     u"RED APPLE": (u"\xF0\x9F\x8D\x8E", ur"""\U0001F34E"""
#     ),
#     u"GREEN APPLE": (u"\xF0\x9F\x8D\x8F", ur"""\U0001F34F"""
#     ),
#     u"PEACH": (u"\xF0\x9F\x8D\x91", ur"""\U0001F351"""
#     ),
#     u"CHERRIES": (u"\xF0\x9F\x8D\x92", ur"""\U0001F352"""
#     ),
#     u"STRAWBERRY": (u"\xF0\x9F\x8D\x93", ur"""\U0001F353"""
#     ),
#     u"HAMBURGER": (u"\xF0\x9F\x8D\x94", ur"""\U0001F354"""
#     ),
#     u"SLICE OF PIZZA": (u"\xF0\x9F\x8D\x95", ur"""\U0001F355"""
#     ),
#     u"MEAT ON BONE": (u"\xF0\x9F\x8D\x96", ur"""\U0001F356"""
#     ),
#     u"POULTRY LEG": (u"\xF0\x9F\x8D\x97", ur"""\U0001F357"""
#     ),
#     u"RICE CRACKER": (u"\xF0\x9F\x8D\x98", ur"""\U0001F358"""
#     ),
#     u"RICE BALL": (u"\xF0\x9F\x8D\x99", ur"""\U0001F359"""
#     ),
#     u"COOKED RICE": (u"\xF0\x9F\x8D\x9A", ur"""\U0001F35A"""
#     ),
#     u"CURRY AND RICE": (u"\xF0\x9F\x8D\x9B", ur"""\U0001F35B"""
#     ),
#     u"STEAMING BOWL": (u"\xF0\x9F\x8D\x9C", ur"""\U0001F35C"""
#     ),
#     u"SPAGHETTI": (u"\xF0\x9F\x8D\x9D", ur"""\U0001F35D"""
#     ),
#     u"BREAD": (u"\xF0\x9F\x8D\x9E", ur"""\U0001F35E"""
#     ),
#     u"FRENCH FRIES": (u"\xF0\x9F\x8D\x9F", ur"""\U0001F35F"""
#     ),
#     u"ROASTED SWEET POTATO": (u"\xF0\x9F\x8D\xA0", ur"""\U0001F360"""
#     ),
#     u"DANGO": (u"\xF0\x9F\x8D\xA1", ur"""\U0001F361"""
#     ),
#     u"ODEN": (u"\xF0\x9F\x8D\xA2", ur"""\U0001F362"""
#     ),
#     u"SUSHI": (u"\xF0\x9F\x8D\xA3", ur"""\U0001F363"""
#     ),
#     u"FRIED SHRIMP": (u"\xF0\x9F\x8D\xA4", ur"""\U0001F364"""
#     ),
#     u"FISH CAKE WITH SWIRL DESIGN": (u"\xF0\x9F\x8D\xA5", ur"""\U0001F365"""
#     ),
#     u"SOFT ICE CREAM": (u"\xF0\x9F\x8D\xA6", ur"""\U0001F366"""
#     ),
#     u"SHAVED ICE": (u"\xF0\x9F\x8D\xA7", ur"""\U0001F367"""
#     ),
#     u"ICE CREAM": (u"\xF0\x9F\x8D\xA8", ur"""\U0001F368"""
#     ),
#     u"DOUGHNUT": (u"\xF0\x9F\x8D\xA9", ur"""\U0001F369"""
#     ),
#     u"COOKIE": (u"\xF0\x9F\x8D\xAA", ur"""\U0001F36A"""
#     ),
#     u"CHOCOLATE BAR": (u"\xF0\x9F\x8D\xAB", ur"""\U0001F36B"""
#     ),
#     u"CANDY": (u"\xF0\x9F\x8D\xAC", ur"""\U0001F36C"""
#     ),
#     u"LOLLIPOP": (u"\xF0\x9F\x8D\xAD", ur"""\U0001F36D"""
#     ),
#     u"CUSTARD": (u"\xF0\x9F\x8D\xAE", ur"""\U0001F36E"""
#     ),
#     u"HONEY POT": (u"\xF0\x9F\x8D\xAF", ur"""\U0001F36F"""
#     ),
#     u"SHORTCAKE": (u"\xF0\x9F\x8D\xB0", ur"""\U0001F370"""
#     ),
#     u"BENTO BOX": (u"\xF0\x9F\x8D\xB1", ur"""\U0001F371"""
#     ),
#     u"POT OF FOOD": (u"\xF0\x9F\x8D\xB2", ur"""\U0001F372"""
#     ),
#     u"COOKING": (u"\xF0\x9F\x8D\xB3", ur"""\U0001F373"""
#     ),
#     u"FORK AND KNIFE": (u"\xF0\x9F\x8D\xB4", ur"""\U0001F374"""
#     ),
#     u"TEACUP WITHOUT HANDLE": (u"\xF0\x9F\x8D\xB5", ur"""\U0001F375"""
#     ),
#     u"SAKE BOTTLE AND CUP": (u"\xF0\x9F\x8D\xB6", ur"""\U0001F376"""
#     ),
#     u"WINE GLASS": (u"\xF0\x9F\x8D\xB7", ur"""\U0001F377"""
#     ),
#     u"COCKTAIL GLASS": (u"\xF0\x9F\x8D\xB8", ur"""\U0001F378"""
#     ),
#     u"TROPICAL DRINK": (u"\xF0\x9F\x8D\xB9", ur"""\U0001F379"""
#     ),
#     u"BEER MUG": (u"\xF0\x9F\x8D\xBA", ur"""\U0001F37A"""
#     ),
#     u"CLINKING BEER MUGS": (u"\xF0\x9F\x8D\xBB", ur"""\U0001F37B"""
#     ),
#     u"RIBBON": (u"\xF0\x9F\x8E\x80", ur"""\U0001F380"""
#     ),
#     u"WRAPPED PRESENT": (u"\xF0\x9F\x8E\x81", ur"""\U0001F381"""
#     ),
#     u"BIRTHDAY CAKE": (u"\xF0\x9F\x8E\x82", ur"""\U0001F382"""
#     ),
#     u"JACK-O-LANTERN": (u"\xF0\x9F\x8E\x83", ur"""\U0001F383"""
#     ),
#     u"CHRISTMAS TREE": (u"\xF0\x9F\x8E\x84", ur"""\U0001F384"""
#     ),
#     u"FATHER CHRISTMAS": (u"\xF0\x9F\x8E\x85", ur"""\U0001F385"""
#     ),
#     u"FIREWORKS": (u"\xF0\x9F\x8E\x86", ur"""\U0001F386"""
#     ),
#     u"FIREWORK SPARKLER": (u"\xF0\x9F\x8E\x87", ur"""\U0001F387"""
#     ),
#     u"BALLOON": (u"\xF0\x9F\x8E\x88", ur"""\U0001F388"""
#     ),
#     u"PARTY POPPER": (u"\xF0\x9F\x8E\x89", ur"""\U0001F389"""
#     ),
#     u"CONFETTI BALL": (u"\xF0\x9F\x8E\x8A", ur"""\U0001F38A"""
#     ),
#     u"TANABATA TREE": (u"\xF0\x9F\x8E\x8B", ur"""\U0001F38B"""
#     ),
#     u"CROSSED FLAGS": (u"\xF0\x9F\x8E\x8C", ur"""\U0001F38C"""
#     ),
#     u"PINE DECORATION": (u"\xF0\x9F\x8E\x8D", ur"""\U0001F38D"""
#     ),
#     u"JAPANESE DOLLS": (u"\xF0\x9F\x8E\x8E", ur"""\U0001F38E"""
#     ),
#     u"CARP STREAMER": (u"\xF0\x9F\x8E\x8F", ur"""\U0001F38F"""
#     ),
#     u"WIND CHIME": (u"\xF0\x9F\x8E\x90", ur"""\U0001F390"""
#     ),
#     u"MOON VIEWING CEREMONY": (u"\xF0\x9F\x8E\x91", ur"""\U0001F391"""
#     ),
#     u"SCHOOL SATCHEL": (u"\xF0\x9F\x8E\x92", ur"""\U0001F392"""
#     ),
#     u"GRADUATION CAP": (u"\xF0\x9F\x8E\x93", ur"""\U0001F393"""
#     ),
#     u"CAROUSEL HORSE": (u"\xF0\x9F\x8E\xA0", ur"""\U0001F3A0"""
#     ),
#     u"FERRIS WHEEL": (u"\xF0\x9F\x8E\xA1", ur"""\U0001F3A1"""
#     ),
#     u"ROLLER COASTER": (u"\xF0\x9F\x8E\xA2", ur"""\U0001F3A2"""
#     ),
#     u"FISHING POLE AND FISH": (u"\xF0\x9F\x8E\xA3", ur"""\U0001F3A3"""
#     ),
#     u"MICROPHONE": (u"\xF0\x9F\x8E\xA4", ur"""\U0001F3A4"""
#     ),
#     u"MOVIE CAMERA": (u"\xF0\x9F\x8E\xA5", ur"""\U0001F3A5"""
#     ),
#     u"CINEMA": (u"\xF0\x9F\x8E\xA6", ur"""\U0001F3A6"""
#     ),
#     u"HEADPHONE": (u"\xF0\x9F\x8E\xA7", ur"""\U0001F3A7"""
#     ),
#     u"ARTIST PALETTE": (u"\xF0\x9F\x8E\xA8", ur"""\U0001F3A8"""
#     ),
#     u"TOP HAT": (u"\xF0\x9F\x8E\xA9", ur"""\U0001F3A9"""
#     ),
#     u"CIRCUS TENT": (u"\xF0\x9F\x8E\xAA", ur"""\U0001F3AA"""
#     ),
#     u"TICKET": (u"\xF0\x9F\x8E\xAB", ur"""\U0001F3AB"""
#     ),
#     u"CLAPPER BOARD": (u"\xF0\x9F\x8E\xAC", ur"""\U0001F3AC"""
#     ),
#     u"PERFORMING ARTS": (u"\xF0\x9F\x8E\xAD", ur"""\U0001F3AD"""
#     ),
#     u"VIDEO GAME": (u"\xF0\x9F\x8E\xAE", ur"""\U0001F3AE"""
#     ),
#     u"DIRECT HIT": (u"\xF0\x9F\x8E\xAF", ur"""\U0001F3AF"""
#     ),
#     u"SLOT MACHINE": (u"\xF0\x9F\x8E\xB0", ur"""\U0001F3B0"""
#     ),
#     u"BILLIARDS": (u"\xF0\x9F\x8E\xB1", ur"""\U0001F3B1"""
#     ),
#     u"GAME DIE": (u"\xF0\x9F\x8E\xB2", ur"""\U0001F3B2"""
#     ),
#     u"BOWLING": (u"\xF0\x9F\x8E\xB3", ur"""\U0001F3B3"""
#     ),
#     u"FLOWER PLAYING CARDS": (u"\xF0\x9F\x8E\xB4", ur"""\U0001F3B4"""
#     ),
#     u"MUSICAL NOTE": (u"\xF0\x9F\x8E\xB5", ur"""\U0001F3B5"""
#     ),
#     u"MULTIPLE MUSICAL NOTES": (u"\xF0\x9F\x8E\xB6", ur"""\U0001F3B6"""
#     ),
#     u"SAXOPHONE": (u"\xF0\x9F\x8E\xB7", ur"""\U0001F3B7"""
#     ),
#     u"GUITAR": (u"\xF0\x9F\x8E\xB8", ur"""\U0001F3B8"""
#     ),
#     u"MUSICAL KEYBOARD": (u"\xF0\x9F\x8E\xB9", ur"""\U0001F3B9"""
#     ),
#     u"TRUMPET": (u"\xF0\x9F\x8E\xBA", ur"""\U0001F3BA"""
#     ),
#     u"VIOLIN": (u"\xF0\x9F\x8E\xBB", ur"""\U0001F3BB"""
#     ),
#     u"MUSICAL SCORE": (u"\xF0\x9F\x8E\xBC", ur"""\U0001F3BC"""
#     ),
#     u"RUNNING SHIRT WITH SASH": (u"\xF0\x9F\x8E\xBD", ur"""\U0001F3BD"""
#     ),
#     u"TENNIS RACQUET AND BALL": (u"\xF0\x9F\x8E\xBE", ur"""\U0001F3BE"""
#     ),
#     u"SKI AND SKI BOOT": (u"\xF0\x9F\x8E\xBF", ur"""\U0001F3BF"""
#     ),
#     u"BASKETBALL AND HOOP": (u"\xF0\x9F\x8F\x80", ur"""\U0001F3C0"""
#     ),
#     u"CHEQUERED FLAG": (u"\xF0\x9F\x8F\x81", ur"""\U0001F3C1"""
#     ),
#     u"SNOWBOARDER": (u"\xF0\x9F\x8F\x82", ur"""\U0001F3C2"""
#     ),
#     u"RUNNER": (u"\xF0\x9F\x8F\x83", ur"""\U0001F3C3"""
#     ),
#     u"SURFER": (u"\xF0\x9F\x8F\x84", ur"""\U0001F3C4"""
#     ),
#     u"TROPHY": (u"\xF0\x9F\x8F\x86", ur"""\U0001F3C6"""
#     ),
#     u"AMERICAN FOOTBALL": (u"\xF0\x9F\x8F\x88", ur"""\U0001F3C8"""
#     ),
#     u"SWIMMER": (u"\xF0\x9F\x8F\x8A", ur"""\U0001F3CA"""
#     ),
#     u"HOUSE BUILDING": (u"\xF0\x9F\x8F\xA0", ur"""\U0001F3E0"""
#     ),
#     u"HOUSE WITH GARDEN": (u"\xF0\x9F\x8F\xA1", ur"""\U0001F3E1"""
#     ),
#     u"OFFICE BUILDING": (u"\xF0\x9F\x8F\xA2", ur"""\U0001F3E2"""
#     ),
#     u"JAPANESE POST OFFICE": (u"\xF0\x9F\x8F\xA3", ur"""\U0001F3E3"""
#     ),
#     u"HOSPITAL": (u"\xF0\x9F\x8F\xA5", ur"""\U0001F3E5"""
#     ),
#     u"BANK": (u"\xF0\x9F\x8F\xA6", ur"""\U0001F3E6"""
#     ),
#     u"AUTOMATED TELLER MACHINE": (u"\xF0\x9F\x8F\xA7", ur"""\U0001F3E7"""
#     ),
#     u"HOTEL": (u"\xF0\x9F\x8F\xA8", ur"""\U0001F3E8"""
#     ),
#     u"LOVE HOTEL": (u"\xF0\x9F\x8F\xA9", ur"""\U0001F3E9"""
#     ),
#     u"CONVENIENCE STORE": (u"\xF0\x9F\x8F\xAA", ur"""\U0001F3EA"""
#     ),
#     u"SCHOOL": (u"\xF0\x9F\x8F\xAB", ur"""\U0001F3EB"""
#     ),
#     u"DEPARTMENT STORE": (u"\xF0\x9F\x8F\xAC", ur"""\U0001F3EC"""
#     ),
#     u"FACTORY": (u"\xF0\x9F\x8F\xAD", ur"""\U0001F3ED"""
#     ),
#     u"IZAKAYA LANTERN": (u"\xF0\x9F\x8F\xAE", ur"""\U0001F3EE"""
#     ),
#     u"JAPANESE CASTLE": (u"\xF0\x9F\x8F\xAF", ur"""\U0001F3EF"""
#     ),
#     u"EUROPEAN CASTLE": (u"\xF0\x9F\x8F\xB0", ur"""\U0001F3F0"""
#     ),
#     u"SNAIL": (u"\xF0\x9F\x90\x8C", ur"""\U0001F40C"""
#     ),
#     u"SNAKE": (u"\xF0\x9F\x90\x8D", ur"""\U0001F40D"""
#     ),
#     u"HORSE": (u"\xF0\x9F\x90\x8E", ur"""\U0001F40E"""
#     ),
#     u"SHEEP": (u"\xF0\x9F\x90\x91", ur"""\U0001F411"""
#     ),
#     u"MONKEY": (u"\xF0\x9F\x90\x92", ur"""\U0001F412"""
#     ),
#     u"CHICKEN": (u"\xF0\x9F\x90\x94", ur"""\U0001F414"""
#     ),
#     u"BOAR": (u"\xF0\x9F\x90\x97", ur"""\U0001F417"""
#     ),
#     u"ELEPHANT": (u"\xF0\x9F\x90\x98", ur"""\U0001F418"""
#     ),
#     u"OCTOPUS": (u"\xF0\x9F\x90\x99", ur"""\U0001F419"""
#     ),
#     u"SPIRAL SHELL": (u"\xF0\x9F\x90\x9A", ur"""\U0001F41A"""
#     ),
#     u"BUG": (u"\xF0\x9F\x90\x9B", ur"""\U0001F41B"""
#     ),
#     u"ANT": (u"\xF0\x9F\x90\x9C", ur"""\U0001F41C"""
#     ),
#     u"HONEYBEE": (u"\xF0\x9F\x90\x9D", ur"""\U0001F41D"""
#     ),
#     u"LADY BEETLE": (u"\xF0\x9F\x90\x9E", ur"""\U0001F41E"""
#     ),
#     u"FISH": (u"\xF0\x9F\x90\x9F", ur"""\U0001F41F"""
#     ),
#     u"TROPICAL FISH": (u"\xF0\x9F\x90\xA0", ur"""\U0001F420"""
#     ),
#     u"BLOWFISH": (u"\xF0\x9F\x90\xA1", ur"""\U0001F421"""
#     ),
#     u"TURTLE": (u"\xF0\x9F\x90\xA2", ur"""\U0001F422"""
#     ),
#     u"HATCHING CHICK": (u"\xF0\x9F\x90\xA3", ur"""\U0001F423"""
#     ),
#     u"BABY CHICK": (u"\xF0\x9F\x90\xA4", ur"""\U0001F424"""
#     ),
#     u"FRONT-FACING BABY CHICK": (u"\xF0\x9F\x90\xA5", ur"""\U0001F425"""
#     ),
#     u"BIRD": (u"\xF0\x9F\x90\xA6", ur"""\U0001F426"""
#     ),
#     u"PENGUIN": (u"\xF0\x9F\x90\xA7", ur"""\U0001F427"""
#     ),
#     u"KOALA": (u"\xF0\x9F\x90\xA8", ur"""\U0001F428"""
#     ),
#     u"POODLE": (u"\xF0\x9F\x90\xA9", ur"""\U0001F429"""
#     ),
#     u"BACTRIAN CAMEL": (u"\xF0\x9F\x90\xAB", ur"""\U0001F42B"""
#     ),
#     u"DOLPHIN": (u"\xF0\x9F\x90\xAC", ur"""\U0001F42C"""
#     ),
#     u"MOUSE FACE": (u"\xF0\x9F\x90\xAD", ur"""\U0001F42D"""
#     ),
#     u"COW FACE": (u"\xF0\x9F\x90\xAE", ur"""\U0001F42E"""
#     ),
#     u"TIGER FACE": (u"\xF0\x9F\x90\xAF", ur"""\U0001F42F"""
#     ),
#     u"RABBIT FACE": (u"\xF0\x9F\x90\xB0", ur"""\U0001F430"""
#     ),
#     u"CAT FACE": (u"\xF0\x9F\x90\xB1", ur"""\U0001F431"""
#     ),
#     u"DRAGON FACE": (u"\xF0\x9F\x90\xB2", ur"""\U0001F432"""
#     ),
#     u"SPOUTING WHALE": (u"\xF0\x9F\x90\xB3", ur"""\U0001F433"""
#     ),
#     u"HORSE FACE": (u"\xF0\x9F\x90\xB4", ur"""\U0001F434"""
#     ),
#     u"MONKEY FACE": (u"\xF0\x9F\x90\xB5", ur"""\U0001F435"""
#     ),
#     u"DOG FACE": (u"\xF0\x9F\x90\xB6", ur"""\U0001F436"""
#     ),
#     u"PIG FACE": (u"\xF0\x9F\x90\xB7", ur"""\U0001F437"""
#     ),
#     u"FROG FACE": (u"\xF0\x9F\x90\xB8", ur"""\U0001F438"""
#     ),
#     u"HAMSTER FACE": (u"\xF0\x9F\x90\xB9", ur"""\U0001F439"""
#     ),
#     u"WOLF FACE": (u"\xF0\x9F\x90\xBA", ur"""\U0001F43A"""
#     ),
#     u"BEAR FACE": (u"\xF0\x9F\x90\xBB", ur"""\U0001F43B"""
#     ),
#     u"PANDA FACE": (u"\xF0\x9F\x90\xBC", ur"""\U0001F43C"""
#     ),
#     u"PIG NOSE": (u"\xF0\x9F\x90\xBD", ur"""\U0001F43D"""
#     ),
#     u"PAW PRINTS": (u"\xF0\x9F\x90\xBE", ur"""\U0001F43E"""
#     ),
#     u"EYES": (u"\xF0\x9F\x91\x80", ur"""\U0001F440"""
#     ),
#     u"EAR": (u"\xF0\x9F\x91\x82", ur"""\U0001F442"""
#     ),
#     u"NOSE": (u"\xF0\x9F\x91\x83", ur"""\U0001F443"""
#     ),
#     u"MOUTH": (u"\xF0\x9F\x91\x84", ur"""\U0001F444"""
#     ),
#     u"TONGUE": (u"\xF0\x9F\x91\x85", ur"""\U0001F445"""
#     ),
#     u"WHITE UP POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x86", ur"""\U0001F446"""
#     ),
#     u"WHITE DOWN POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x87", ur"""\U0001F447"""
#     ),
#     u"WHITE LEFT POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x88", ur"""\U0001F448"""
#     ),
#     u"WHITE RIGHT POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x89", ur"""\U0001F449"""
#     ),
#     u"FISTED HAND SIGN": (u"\xF0\x9F\x91\x8A", ur"""\U0001F44A"""
#     ),
#     u"WAVING HAND SIGN": (u"\xF0\x9F\x91\x8B", ur"""\U0001F44B"""
#     ),
#     u"OK HAND SIGN": (u"\xF0\x9F\x91\x8C", ur"""\U0001F44C"""
#     ),
#     u"THUMBS UP SIGN": (u"\xF0\x9F\x91\x8D", ur"""\U0001F44D"""
#     ),
#     u"THUMBS DOWN SIGN": (u"\xF0\x9F\x91\x8E", ur"""\U0001F44E"""
#     ),
#     u"CLAPPING HANDS SIGN": (u"\xF0\x9F\x91\x8F", ur"""\U0001F44F"""
#     ),
#     u"OPEN HANDS SIGN": (u"\xF0\x9F\x91\x90", ur"""\U0001F450"""
#     ),
    u"CROWN": (u"\xF0\x9F\x91\x91", ur"""\U0001F451"""
    ),
#     u"WOMANS HAT": (u"\xF0\x9F\x91\x92", ur"""\U0001F452"""
#     ),
#     u"EYEGLASSES": (u"\xF0\x9F\x91\x93", ur"""\U0001F453"""
#     ),
#     u"NECKTIE": (u"\xF0\x9F\x91\x94", ur"""\U0001F454"""
#     ),
#     u"T-SHIRT": (u"\xF0\x9F\x91\x95", ur"""\U0001F455"""
#     ),
#     u"JEANS": (u"\xF0\x9F\x91\x96", ur"""\U0001F456"""
#     ),
#     u"DRESS": (u"\xF0\x9F\x91\x97", ur"""\U0001F457"""
#     ),
#     u"KIMONO": (u"\xF0\x9F\x91\x98", ur"""\U0001F458"""
#     ),
#     u"BIKINI": (u"\xF0\x9F\x91\x99", ur"""\U0001F459"""
#     ),
#     u"WOMANS CLOTHES": (u"\xF0\x9F\x91\x9A", ur"""\U0001F45A"""
#     ),
#     u"PURSE": (u"\xF0\x9F\x91\x9B", ur"""\U0001F45B"""
#     ),
#     u"HANDBAG": (u"\xF0\x9F\x91\x9C", ur"""\U0001F45C"""
#     ),
#     u"POUCH": (u"\xF0\x9F\x91\x9D", ur"""\U0001F45D"""
#     ),
#     u"MANS SHOE": (u"\xF0\x9F\x91\x9E", ur"""\U0001F45E"""
#     ),
#     u"ATHLETIC SHOE": (u"\xF0\x9F\x91\x9F", ur"""\U0001F45F"""
#     ),
#     u"HIGH-HEELED SHOE": (u"\xF0\x9F\x91\xA0", ur"""\U0001F460"""
#     ),
#     u"WOMANS SANDAL": (u"\xF0\x9F\x91\xA1", ur"""\U0001F461"""
#     ),
#     u"WOMANS BOOTS": (u"\xF0\x9F\x91\xA2", ur"""\U0001F462"""
#     ),
#     u"FOOTPRINTS": (u"\xF0\x9F\x91\xA3", ur"""\U0001F463"""
#     ),
#     u"BUST IN SILHOUETTE": (u"\xF0\x9F\x91\xA4", ur"""\U0001F464"""
#     ),
#     u"BOY": (u"\xF0\x9F\x91\xA6", ur"""\U0001F466"""
#     ),
#     u"GIRL": (u"\xF0\x9F\x91\xA7", ur"""\U0001F467"""
#     ),
#     u"MAN": (u"\xF0\x9F\x91\xA8", ur"""\U0001F468"""
#     ),
#     u"WOMAN": (u"\xF0\x9F\x91\xA9", ur"""\U0001F469"""
#     ),
#     u"FAMILY": (u"\xF0\x9F\x91\xAA", ur"""\U0001F46A"""
#     ),
#     u"MAN AND WOMAN HOLDING HANDS": (u"\xF0\x9F\x91\xAB", ur"""\U0001F46B"""
#     ),
#     u"POLICE OFFICER": (u"\xF0\x9F\x91\xAE", ur"""\U0001F46E"""
#     ),
#     u"WOMAN WITH BUNNY EARS": (u"\xF0\x9F\x91\xAF", ur"""\U0001F46F"""
#     ),
#     u"BRIDE WITH VEIL": (u"\xF0\x9F\x91\xB0", ur"""\U0001F470"""
#     ),
#     u"PERSON WITH BLOND HAIR": (u"\xF0\x9F\x91\xB1", ur"""\U0001F471"""
#     ),
#     u"MAN WITH GUA PI MAO": (u"\xF0\x9F\x91\xB2", ur"""\U0001F472"""
#     ),
#     u"MAN WITH TURBAN": (u"\xF0\x9F\x91\xB3", ur"""\U0001F473"""
#     ),
#     u"OLDER MAN": (u"\xF0\x9F\x91\xB4", ur"""\U0001F474"""
#     ),
#     u"OLDER WOMAN": (u"\xF0\x9F\x91\xB5", ur"""\U0001F475"""
#     ),
#     u"BABY": (u"\xF0\x9F\x91\xB6", ur"""\U0001F476"""
#     ),
#     u"CONSTRUCTION WORKER": (u"\xF0\x9F\x91\xB7", ur"""\U0001F477"""
#     ),
#     u"PRINCESS": (u"\xF0\x9F\x91\xB8", ur"""\U0001F478"""
#     ),
#     u"JAPANESE OGRE": (u"\xF0\x9F\x91\xB9", ur"""\U0001F479"""
#     ),
#     u"JAPANESE GOBLIN": (u"\xF0\x9F\x91\xBA", ur"""\U0001F47A"""
#     ),
#     u"GHOST": (u"\xF0\x9F\x91\xBB", ur"""\U0001F47B"""
#     ),
#     u"BABY ANGEL": (u"\xF0\x9F\x91\xBC", ur"""\U0001F47C"""
#     ),
#     u"EXTRATERRESTRIAL ALIEN": (u"\xF0\x9F\x91\xBD", ur"""\U0001F47D"""
#     ),
#     u"ALIEN MONSTER": (u"\xF0\x9F\x91\xBE", ur"""\U0001F47E"""
#     ),
#     u"IMP": (u"\xF0\x9F\x91\xBF", ur"""\U0001F47F"""
#     ),
#     u"SKULL": (u"\xF0\x9F\x92\x80", ur"""\U0001F480"""
#     ),
#     u"INFORMATION DESK PERSON": (u"\xF0\x9F\x92\x81", ur"""\U0001F481"""
#     ),
#     u"GUARDSMAN": (u"\xF0\x9F\x92\x82", ur"""\U0001F482"""
#     ),
#     u"DANCER": (u"\xF0\x9F\x92\x83", ur"""\U0001F483"""
#     ),
#     u"LIPSTICK": (u"\xF0\x9F\x92\x84", ur"""\U0001F484"""
#     ),
#     u"NAIL POLISH": (u"\xF0\x9F\x92\x85", ur"""\U0001F485"""
#     ),
#     u"FACE MASSAGE": (u"\xF0\x9F\x92\x86", ur"""\U0001F486"""
#     ),
#     u"HAIRCUT": (u"\xF0\x9F\x92\x87", ur"""\U0001F487"""
#     ),
#     u"BARBER POLE": (u"\xF0\x9F\x92\x88", ur"""\U0001F488"""
#     ),
#     u"SYRINGE": (u"\xF0\x9F\x92\x89", ur"""\U0001F489"""
#     ),
#     u"PILL": (u"\xF0\x9F\x92\x8A", ur"""\U0001F48A"""
#     ),
#     u"KISS MARK": (u"\xF0\x9F\x92\x8B", ur"""\U0001F48B"""
#     ),
#     u"LOVE LETTER": (u"\xF0\x9F\x92\x8C", ur"""\U0001F48C"""
#     ),
#     u"RING": (u"\xF0\x9F\x92\x8D", ur"""\U0001F48D"""
#     ),
#     u"GEM STONE": (u"\xF0\x9F\x92\x8E", ur"""\U0001F48E"""
#     ),
#     u"KISS": (u"\xF0\x9F\x92\x8F", ur"""\U0001F48F"""
#     ),
#     u"BOUQUET": (u"\xF0\x9F\x92\x90", ur"""\U0001F490"""
#     ),
#     u"COUPLE WITH HEART": (u"\xF0\x9F\x92\x91", ur"""\U0001F491"""
#     ),
#     u"WEDDING": (u"\xF0\x9F\x92\x92", ur"""\U0001F492"""
#     ),
#     u"BEATING HEART": (u"\xF0\x9F\x92\x93", ur"""\U0001F493"""
#     ),
#     u"BROKEN HEART": (u"\xF0\x9F\x92\x94", ur"""\U0001F494"""
#     ),
#     u"TWO HEARTS": (u"\xF0\x9F\x92\x95", ur"""\U0001F495"""
#     ),
#     u"SPARKLING HEART": (u"\xF0\x9F\x92\x96", ur"""\U0001F496"""
#     ),
#     u"GROWING HEART": (u"\xF0\x9F\x92\x97", ur"""\U0001F497"""
#     ),
#     u"HEART WITH ARROW": (u"\xF0\x9F\x92\x98", ur"""\U0001F498"""
#     ),
#     u"BLUE HEART": (u"\xF0\x9F\x92\x99", ur"""\U0001F499"""
#     ),
#     u"GREEN HEART": (u"\xF0\x9F\x92\x9A", ur"""\U0001F49A"""
#     ),
#     u"YELLOW HEART": (u"\xF0\x9F\x92\x9B", ur"""\U0001F49B"""
#     ),
#     u"PURPLE HEART": (u"\xF0\x9F\x92\x9C", ur"""\U0001F49C"""
#     ),
#     u"HEART WITH RIBBON": (u"\xF0\x9F\x92\x9D", ur"""\U0001F49D"""
#     ),
#     u"REVOLVING HEARTS": (u"\xF0\x9F\x92\x9E", ur"""\U0001F49E"""
#     ),
#     u"HEART DECORATION": (u"\xF0\x9F\x92\x9F", ur"""\U0001F49F"""
#     ),
#     u"DIAMOND SHAPE WITH A DOT INSIDE": (u"\xF0\x9F\x92\xA0", ur"""\U0001F4A0"""
#     ),
#     u"ELECTRIC LIGHT BULB": (u"\xF0\x9F\x92\xA1", ur"""\U0001F4A1"""
#     ),
#     u"ANGER SYMBOL": (u"\xF0\x9F\x92\xA2", ur"""\U0001F4A2"""
#     ),
#     u"BOMB": (u"\xF0\x9F\x92\xA3", ur"""\U0001F4A3"""
#     ),
#     u"SLEEPING SYMBOL": (u"\xF0\x9F\x92\xA4", ur"""\U0001F4A4"""
#     ),
#     u"COLLISION SYMBOL": (u"\xF0\x9F\x92\xA5", ur"""\U0001F4A5"""
#     ),
#     u"SPLASHING SWEAT SYMBOL": (u"\xF0\x9F\x92\xA6", ur"""\U0001F4A6"""
#     ),
#     u"DROPLET": (u"\xF0\x9F\x92\xA7", ur"""\U0001F4A7"""
#     ),
#     u"DASH SYMBOL": (u"\xF0\x9F\x92\xA8", ur"""\U0001F4A8"""
#     ),
#     u"PILE OF POO": (u"\xF0\x9F\x92\xA9", ur"""\U0001F4A9"""
#     ),
#     u"FLEXED BICEPS": (u"\xF0\x9F\x92\xAA", ur"""\U0001F4AA"""
#     ),
#     u"DIZZY SYMBOL": (u"\xF0\x9F\x92\xAB", ur"""\U0001F4AB"""
#     ),
#     u"SPEECH BALLOON": (u"\xF0\x9F\x92\xAC", ur"""\U0001F4AC"""
#     ),
#     u"WHITE FLOWER": (u"\xF0\x9F\x92\xAE", ur"""\U0001F4AE"""
#     ),
#     u"HUNDRED POINTS SYMBOL": (u"\xF0\x9F\x92\xAF", ur"""\U0001F4AF"""
#     ),
    u"MONEY BAG": (u"\xF0\x9F\x92\xB0", ur"""\U0001F4B0"""
    ),
#     u"CURRENCY EXCHANGE": (u"\xF0\x9F\x92\xB1", ur"""\U0001F4B1"""
#     ),
#     u"HEAVY DOLLAR SIGN": (u"\xF0\x9F\x92\xB2", ur"""\U0001F4B2"""
#     ),
#     u"CREDIT CARD": (u"\xF0\x9F\x92\xB3", ur"""\U0001F4B3"""
#     ),
#     u"BANKNOTE WITH YEN SIGN": (u"\xF0\x9F\x92\xB4", ur"""\U0001F4B4"""
#     ),
#     u"BANKNOTE WITH DOLLAR SIGN": (u"\xF0\x9F\x92\xB5", ur"""\U0001F4B5"""
#     ),
#     u"MONEY WITH WINGS": (u"\xF0\x9F\x92\xB8", ur"""\U0001F4B8"""
#     ),
#     u"CHART WITH UPWARDS TREND AND YEN SIGN": (u"\xF0\x9F\x92\xB9", ur"""\U0001F4B9"""
#     ),
#     u"SEAT": (u"\xF0\x9F\x92\xBA", ur"""\U0001F4BA"""
#     ),
#     u"PERSONAL COMPUTER": (u"\xF0\x9F\x92\xBB", ur"""\U0001F4BB"""
#     ),
#     u"BRIEFCASE": (u"\xF0\x9F\x92\xBC", ur"""\U0001F4BC"""
#     ),
#     u"MINIDISC": (u"\xF0\x9F\x92\xBD", ur"""\U0001F4BD"""
#     ),
#     u"FLOPPY DISK": (u"\xF0\x9F\x92\xBE", ur"""\U0001F4BE"""
#     ),
#     u"OPTICAL DISC": (u"\xF0\x9F\x92\xBF", ur"""\U0001F4BF"""
#     ),
#     u"DVD": (u"\xF0\x9F\x93\x80", ur"""\U0001F4C0"""
#     ),
#     u"FILE FOLDER": (u"\xF0\x9F\x93\x81", ur"""\U0001F4C1"""
#     ),
#     u"OPEN FILE FOLDER": (u"\xF0\x9F\x93\x82", ur"""\U0001F4C2"""
#     ),
#     u"PAGE WITH CURL": (u"\xF0\x9F\x93\x83", ur"""\U0001F4C3"""
#     ),
#     u"PAGE FACING UP": (u"\xF0\x9F\x93\x84", ur"""\U0001F4C4"""
#     ),
#     u"CALENDAR": (u"\xF0\x9F\x93\x85", ur"""\U0001F4C5"""
#     ),
#     u"TEAR-OFF CALENDAR": (u"\xF0\x9F\x93\x86", ur"""\U0001F4C6"""
#     ),
#     u"CARD INDEX": (u"\xF0\x9F\x93\x87", ur"""\U0001F4C7"""
#     ),
#     u"CHART WITH UPWARDS TREND": (u"\xF0\x9F\x93\x88", ur"""\U0001F4C8"""
#     ),
#     u"CHART WITH DOWNWARDS TREND": (u"\xF0\x9F\x93\x89", ur"""\U0001F4C9"""
#     ),
#     u"BAR CHART": (u"\xF0\x9F\x93\x8A", ur"""\U0001F4CA"""
#     ),
#     u"CLIPBOARD": (u"\xF0\x9F\x93\x8B", ur"""\U0001F4CB"""
#     ),
#     u"PUSHPIN": (u"\xF0\x9F\x93\x8C", ur"""\U0001F4CC"""
#     ),
#     u"ROUND PUSHPIN": (u"\xF0\x9F\x93\x8D", ur"""\U0001F4CD"""
#     ),
#     u"PAPERCLIP": (u"\xF0\x9F\x93\x8E", ur"""\U0001F4CE"""
#     ),
#     u"STRAIGHT RULER": (u"\xF0\x9F\x93\x8F", ur"""\U0001F4CF"""
#     ),
#     u"TRIANGULAR RULER": (u"\xF0\x9F\x93\x90", ur"""\U0001F4D0"""
#     ),
#     u"BOOKMARK TABS": (u"\xF0\x9F\x93\x91", ur"""\U0001F4D1"""
#     ),
#     u"LEDGER": (u"\xF0\x9F\x93\x92", ur"""\U0001F4D2"""
#     ),
#     u"NOTEBOOK": (u"\xF0\x9F\x93\x93", ur"""\U0001F4D3"""
#     ),
#     u"NOTEBOOK WITH DECORATIVE COVER": (u"\xF0\x9F\x93\x94", ur"""\U0001F4D4"""
#     ),
#     u"CLOSED BOOK": (u"\xF0\x9F\x93\x95", ur"""\U0001F4D5"""
#     ),
#     u"OPEN BOOK": (u"\xF0\x9F\x93\x96", ur"""\U0001F4D6"""
#     ),
#     u"GREEN BOOK": (u"\xF0\x9F\x93\x97", ur"""\U0001F4D7"""
#     ),
#     u"BLUE BOOK": (u"\xF0\x9F\x93\x98", ur"""\U0001F4D8"""
#     ),
#     u"ORANGE BOOK": (u"\xF0\x9F\x93\x99", ur"""\U0001F4D9"""
#     ),
#     u"BOOKS": (u"\xF0\x9F\x93\x9A", ur"""\U0001F4DA"""
#     ),
#     u"NAME BADGE": (u"\xF0\x9F\x93\x9B", ur"""\U0001F4DB"""
#     ),
#     u"SCROLL": (u"\xF0\x9F\x93\x9C", ur"""\U0001F4DC"""
#     ),
#     u"MEMO": (u"\xF0\x9F\x93\x9D", ur"""\U0001F4DD"""
#     ),
#     u"TELEPHONE RECEIVER": (u"\xF0\x9F\x93\x9E", ur"""\U0001F4DE"""
#     ),
#     u"PAGER": (u"\xF0\x9F\x93\x9F", ur"""\U0001F4DF"""
#     ),
#     u"FAX MACHINE": (u"\xF0\x9F\x93\xA0", ur"""\U0001F4E0"""
#     ),
#     u"SATELLITE ANTENNA": (u"\xF0\x9F\x93\xA1", ur"""\U0001F4E1"""
#     ),
#     u"PUBLIC ADDRESS LOUDSPEAKER": (u"\xF0\x9F\x93\xA2", ur"""\U0001F4E2"""
#     ),
#     u"CHEERING MEGAPHONE": (u"\xF0\x9F\x93\xA3", ur"""\U0001F4E3"""
#     ),
#     u"OUTBOX TRAY": (u"\xF0\x9F\x93\xA4", ur"""\U0001F4E4"""
#     ),
#     u"INBOX TRAY": (u"\xF0\x9F\x93\xA5", ur"""\U0001F4E5"""
#     ),
#     u"PACKAGE": (u"\xF0\x9F\x93\xA6", ur"""\U0001F4E6"""
#     ),
#     u"E-MAIL SYMBOL": (u"\xF0\x9F\x93\xA7", ur"""\U0001F4E7"""
#     ),
#     u"INCOMING ENVELOPE": (u"\xF0\x9F\x93\xA8", ur"""\U0001F4E8"""
#     ),
#     u"ENVELOPE WITH DOWNWARDS ARROW ABOVE": (u"\xF0\x9F\x93\xA9", ur"""\U0001F4E9"""
#     ),
#     u"CLOSED MAILBOX WITH LOWERED FLAG": (u"\xF0\x9F\x93\xAA", ur"""\U0001F4EA"""
#     ),
#     u"CLOSED MAILBOX WITH RAISED FLAG": (u"\xF0\x9F\x93\xAB", ur"""\U0001F4EB"""
#     ),
#     u"POSTBOX": (u"\xF0\x9F\x93\xAE", ur"""\U0001F4EE"""
#     ),
#     u"NEWSPAPER": (u"\xF0\x9F\x93\xB0", ur"""\U0001F4F0"""
#     ),
#     u"MOBILE PHONE": (u"\xF0\x9F\x93\xB1", ur"""\U0001F4F1"""
#     ),
#     u"MOBILE PHONE WITH RIGHTWARDS ARROW AT LEFT": (u"\xF0\x9F\x93\xB2", ur"""\U0001F4F2"""
#     ),
#     u"VIBRATION MODE": (u"\xF0\x9F\x93\xB3", ur"""\U0001F4F3"""
#     ),
#     u"MOBILE PHONE OFF": (u"\xF0\x9F\x93\xB4", ur"""\U0001F4F4"""
#     ),
#     u"ANTENNA WITH BARS": (u"\xF0\x9F\x93\xB6", ur"""\U0001F4F6"""
#     ),
#     u"CAMERA": (u"\xF0\x9F\x93\xB7", ur"""\U0001F4F7"""
#     ),
#     u"VIDEO CAMERA": (u"\xF0\x9F\x93\xB9", ur"""\U0001F4F9"""
#     ),
#     u"TELEVISION": (u"\xF0\x9F\x93\xBA", ur"""\U0001F4FA"""
#     ),
#     u"RADIO": (u"\xF0\x9F\x93\xBB", ur"""\U0001F4FB"""
#     ),
#     u"VIDEOCASSETTE": (u"\xF0\x9F\x93\xBC", ur"""\U0001F4FC"""
#     ),
#     u"CLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x83", ur"""\U0001F503"""
#     ),
#     u"SPEAKER WITH THREE SOUND WAVES": (u"\xF0\x9F\x94\x8A", ur"""\U0001F50A"""
#     ),
#     u"BATTERY": (u"\xF0\x9F\x94\x8B", ur"""\U0001F50B"""
#     ),
#     u"ELECTRIC PLUG": (u"\xF0\x9F\x94\x8C", ur"""\U0001F50C"""
#     ),
#     u"LEFT-POINTING MAGNIFYING GLASS": (u"\xF0\x9F\x94\x8D", ur"""\U0001F50D"""
#     ),
#     u"RIGHT-POINTING MAGNIFYING GLASS": (u"\xF0\x9F\x94\x8E", ur"""\U0001F50E"""
#     ),
#     u"LOCK WITH INK PEN": (u"\xF0\x9F\x94\x8F", ur"""\U0001F50F"""
#     ),
#     u"CLOSED LOCK WITH KEY": (u"\xF0\x9F\x94\x90", ur"""\U0001F510"""
#     ),
#     u"KEY": (u"\xF0\x9F\x94\x91", ur"""\U0001F511"""
#     ),
#     u"LOCK": (u"\xF0\x9F\x94\x92", ur"""\U0001F512"""
#     ),
#     u"OPEN LOCK": (u"\xF0\x9F\x94\x93", ur"""\U0001F513"""
#     ),
#     u"BELL": (u"\xF0\x9F\x94\x94", ur"""\U0001F514"""
#     ),
#     u"BOOKMARK": (u"\xF0\x9F\x94\x96", ur"""\U0001F516"""
#     ),
#     u"LINK SYMBOL": (u"\xF0\x9F\x94\x97", ur"""\U0001F517"""
#     ),
#     u"RADIO BUTTON": (u"\xF0\x9F\x94\x98", ur"""\U0001F518"""
#     ),
#     u"BACK WITH LEFTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x99", ur"""\U0001F519"""
#     ),
#     u"END WITH LEFTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9A", ur"""\U0001F51A"""
#     ),
#     u"ON WITH EXCLAMATION MARK WITH LEFT RIGHT ARROW ABOVE": (u"\xF0\x9F\x94\x9B", ur"""\U0001F51B"""
#     ),
#     u"SOON WITH RIGHTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9C", ur"""\U0001F51C"""
#     ),
#     u"TOP WITH UPWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9D", ur"""\U0001F51D"""
#     ),
#     u"NO ONE UNDER EIGHTEEN SYMBOL": (u"\xF0\x9F\x94\x9E", ur"""\U0001F51E"""
#     ),
#     u"KEYCAP TEN": (u"\xF0\x9F\x94\x9F", ur"""\U0001F51F"""
#     ),
#     u"INPUT SYMBOL FOR LATIN CAPITAL LETTERS": (u"\xF0\x9F\x94\xA0", ur"""\U0001F520"""
#     ),
#     u"INPUT SYMBOL FOR LATIN SMALL LETTERS": (u"\xF0\x9F\x94\xA1", ur"""\U0001F521"""
#     ),
#     u"INPUT SYMBOL FOR NUMBERS": (u"\xF0\x9F\x94\xA2", ur"""\U0001F522"""
#     ),
#     u"INPUT SYMBOL FOR SYMBOLS": (u"\xF0\x9F\x94\xA3", ur"""\U0001F523"""
#     ),
#     u"INPUT SYMBOL FOR LATIN LETTERS": (u"\xF0\x9F\x94\xA4", ur"""\U0001F524"""
#     ),
#     u"FIRE": (u"\xF0\x9F\x94\xA5", ur"""\U0001F525"""
#     ),
#     u"ELECTRIC TORCH": (u"\xF0\x9F\x94\xA6", ur"""\U0001F526"""
#     ),
#     u"WRENCH": (u"\xF0\x9F\x94\xA7", ur"""\U0001F527"""
#     ),
#     u"HAMMER": (u"\xF0\x9F\x94\xA8", ur"""\U0001F528"""
#     ),
#     u"NUT AND BOLT": (u"\xF0\x9F\x94\xA9", ur"""\U0001F529"""
#     ),
#     u"HOCHO": (u"\xF0\x9F\x94\xAA", ur"""\U0001F52A"""
#     ),
#     u"PISTOL": (u"\xF0\x9F\x94\xAB", ur"""\U0001F52B"""
#     ),
#     u"CRYSTAL BALL": (u"\xF0\x9F\x94\xAE", ur"""\U0001F52E"""
#     ),
#     u"SIX POINTED STAR WITH MIDDLE DOT": (u"\xF0\x9F\x94\xAF", ur"""\U0001F52F"""
#     ),
#     u"JAPANESE SYMBOL FOR BEGINNER": (u"\xF0\x9F\x94\xB0", ur"""\U0001F530"""
#     ),
#     u"TRIDENT EMBLEM": (u"\xF0\x9F\x94\xB1", ur"""\U0001F531"""
#     ),
#     u"BLACK SQUARE BUTTON": (u"\xF0\x9F\x94\xB2", ur"""\U0001F532"""
#     ),
#     u"WHITE SQUARE BUTTON": (u"\xF0\x9F\x94\xB3", ur"""\U0001F533"""
#     ),
#     u"LARGE RED CIRCLE": (u"\xF0\x9F\x94\xB4", ur"""\U0001F534"""
#     ),
#     u"LARGE BLUE CIRCLE": (u"\xF0\x9F\x94\xB5", ur"""\U0001F535"""
#     ),
#     u"LARGE ORANGE DIAMOND": (u"\xF0\x9F\x94\xB6", ur"""\U0001F536"""
#     ),
#     u"LARGE BLUE DIAMOND": (u"\xF0\x9F\x94\xB7", ur"""\U0001F537"""
#     ),
#     u"SMALL ORANGE DIAMOND": (u"\xF0\x9F\x94\xB8", ur"""\U0001F538"""
#     ),
#     u"SMALL BLUE DIAMOND": (u"\xF0\x9F\x94\xB9", ur"""\U0001F539"""
#     ),
#     u"UP-POINTING RED TRIANGLE": (u"\xF0\x9F\x94\xBA", ur"""\U0001F53A"""
#     ),
#     u"DOWN-POINTING RED TRIANGLE": (u"\xF0\x9F\x94\xBB", ur"""\U0001F53B"""
#     ),
#     u"UP-POINTING SMALL RED TRIANGLE": (u"\xF0\x9F\x94\xBC", ur"""\U0001F53C"""
#     ),
#     u"DOWN-POINTING SMALL RED TRIANGLE": (u"\xF0\x9F\x94\xBD", ur"""\U0001F53D"""
#     ),
#     u"CLOCK FACE ONE OCLOCK": (u"\xF0\x9F\x95\x90", ur"""\U0001F550"""
#     ),
#     u"CLOCK FACE TWO OCLOCK": (u"\xF0\x9F\x95\x91", ur"""\U0001F551"""
#     ),
#     u"CLOCK FACE THREE OCLOCK": (u"\xF0\x9F\x95\x92", ur"""\U0001F552"""
#     ),
#     u"CLOCK FACE FOUR OCLOCK": (u"\xF0\x9F\x95\x93", ur"""\U0001F553"""
#     ),
#     u"CLOCK FACE FIVE OCLOCK": (u"\xF0\x9F\x95\x94", ur"""\U0001F554"""
#     ),
#     u"CLOCK FACE SIX OCLOCK": (u"\xF0\x9F\x95\x95", ur"""\U0001F555"""
#     ),
#     u"CLOCK FACE SEVEN OCLOCK": (u"\xF0\x9F\x95\x96", ur"""\U0001F556"""
#     ),
#     u"CLOCK FACE EIGHT OCLOCK": (u"\xF0\x9F\x95\x97", ur"""\U0001F557"""
#     ),
#     u"CLOCK FACE NINE OCLOCK": (u"\xF0\x9F\x95\x98", ur"""\U0001F558"""
#     ),
#     u"CLOCK FACE TEN OCLOCK": (u"\xF0\x9F\x95\x99", ur"""\U0001F559"""
#     ),
#     u"CLOCK FACE ELEVEN OCLOCK": (u"\xF0\x9F\x95\x9A", ur"""\U0001F55A"""
#     ),
#     u"CLOCK FACE TWELVE OCLOCK": (u"\xF0\x9F\x95\x9B", ur"""\U0001F55B"""
#     ),
#     u"MOUNT FUJI": (u"\xF0\x9F\x97\xBB", ur"""\U0001F5FB"""
#     ),
#     u"TOKYO TOWER": (u"\xF0\x9F\x97\xBC", ur"""\U0001F5FC"""
#     ),
#     u"STATUE OF LIBERTY": (u"\xF0\x9F\x97\xBD", ur"""\U0001F5FD"""
#     ),
#     u"SILHOUETTE OF JAPAN": (u"\xF0\x9F\x97\xBE", ur"""\U0001F5FE"""
#     ),
#     u"MOYAI": (u"\xF0\x9F\x97\xBF", ur"""\U0001F5FF"""
#     ),
#     u"GRINNING FACE": (u"\xF0\x9F\x98\x80", ur"""\U0001F600"""
#     ),
#     u"SMILING FACE WITH HALO": (u"\xF0\x9F\x98\x87", ur"""\U0001F607"""
#     ),
#     u"SMILING FACE WITH HORNS": (u"\xF0\x9F\x98\x88", ur"""\U0001F608"""
#     ),
#     u"SMILING FACE WITH SUNGLASSES": (u"\xF0\x9F\x98\x8E", ur"""\U0001F60E"""
#     ),
#     u"NEUTRAL FACE": (u"\xF0\x9F\x98\x90", ur"""\U0001F610"""
#     ),
#     u"EXPRESSIONLESS FACE": (u"\xF0\x9F\x98\x91", ur"""\U0001F611"""
#     ),
#     u"CONFUSED FACE": (u"\xF0\x9F\x98\x95", ur"""\U0001F615"""
#     ),
#     u"KISSING FACE": (u"\xF0\x9F\x98\x97", ur"""\U0001F617"""
#     ),
#     u"KISSING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x99", ur"""\U0001F619"""
#     ),
#     u"FACE WITH STUCK-OUT TONGUE": (u"\xF0\x9F\x98\x9B", ur"""\U0001F61B"""
#     ),
#     u"WORRIED FACE": (u"\xF0\x9F\x98\x9F", ur"""\U0001F61F"""
#     ),
#     u"FROWNING FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xA6", ur"""\U0001F626"""
#     ),
#     u"ANGUISHED FACE": (u"\xF0\x9F\x98\xA7", ur"""\U0001F627"""
#     ),
#     u"GRIMACING FACE": (u"\xF0\x9F\x98\xAC", ur"""\U0001F62C"""
#     ),
#     u"FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xAE", ur"""\U0001F62E"""
#     ),
#     u"HUSHED FACE": (u"\xF0\x9F\x98\xAF", ur"""\U0001F62F"""
#     ),
#     u"SLEEPING FACE": (u"\xF0\x9F\x98\xB4", ur"""\U0001F634"""
#     ),
#     u"FACE WITHOUT MOUTH": (u"\xF0\x9F\x98\xB6", ur"""\U0001F636"""
#     ),
#     u"HELICOPTER": (u"\xF0\x9F\x9A\x81", ur"""\U0001F681"""
#     ),
#     u"STEAM LOCOMOTIVE": (u"\xF0\x9F\x9A\x82", ur"""\U0001F682"""
#     ),
#     u"TRAIN": (u"\xF0\x9F\x9A\x86", ur"""\U0001F686"""
#     ),
#     u"LIGHT RAIL": (u"\xF0\x9F\x9A\x88", ur"""\U0001F688"""
#     ),
#     u"TRAM": (u"\xF0\x9F\x9A\x8A", ur"""\U0001F68A"""
#     ),
#     u"ONCOMING BUS": (u"\xF0\x9F\x9A\x8D", ur"""\U0001F68D"""
#     ),
#     u"TROLLEYBUS": (u"\xF0\x9F\x9A\x8E", ur"""\U0001F68E"""
#     ),
#     u"MINIBUS": (u"\xF0\x9F\x9A\x90", ur"""\U0001F690"""
#     ),
#     u"ONCOMING POLICE CAR": (u"\xF0\x9F\x9A\x94", ur"""\U0001F694"""
#     ),
#     u"ONCOMING TAXI": (u"\xF0\x9F\x9A\x96", ur"""\U0001F696"""
#     ),
#     u"ONCOMING AUTOMOBILE": (u"\xF0\x9F\x9A\x98", ur"""\U0001F698"""
#     ),
#     u"ARTICULATED LORRY": (u"\xF0\x9F\x9A\x9B", ur"""\U0001F69B"""
#     ),
#     u"TRACTOR": (u"\xF0\x9F\x9A\x9C", ur"""\U0001F69C"""
#     ),
#     u"MONORAIL": (u"\xF0\x9F\x9A\x9D", ur"""\U0001F69D"""
#     ),
#     u"MOUNTAIN RAILWAY": (u"\xF0\x9F\x9A\x9E", ur"""\U0001F69E"""
#     ),
#     u"SUSPENSION RAILWAY": (u"\xF0\x9F\x9A\x9F", ur"""\U0001F69F"""
#     ),
#     u"MOUNTAIN CABLEWAY": (u"\xF0\x9F\x9A\xA0", ur"""\U0001F6A0"""
#     ),
#     u"AERIAL TRAMWAY": (u"\xF0\x9F\x9A\xA1", ur"""\U0001F6A1"""
#     ),
#     u"ROWBOAT": (u"\xF0\x9F\x9A\xA3", ur"""\U0001F6A3"""
#     ),
#     u"VERTICAL TRAFFIC LIGHT": (u"\xF0\x9F\x9A\xA6", ur"""\U0001F6A6"""
#     ),
#     u"PUT LITTER IN ITS PLACE SYMBOL": (u"\xF0\x9F\x9A\xAE", ur"""\U0001F6AE"""
#     ),
#     u"DO NOT LITTER SYMBOL": (u"\xF0\x9F\x9A\xAF", ur"""\U0001F6AF"""
#     ),
#     u"POTABLE WATER SYMBOL": (u"\xF0\x9F\x9A\xB0", ur"""\U0001F6B0"""
#     ),
#     u"NON-POTABLE WATER SYMBOL": (u"\xF0\x9F\x9A\xB1", ur"""\U0001F6B1"""
#     ),
#     u"NO BICYCLES": (u"\xF0\x9F\x9A\xB3", ur"""\U0001F6B3"""
#     ),
#     u"BICYCLIST": (u"\xF0\x9F\x9A\xB4", ur"""\U0001F6B4"""
#     ),
#     u"MOUNTAIN BICYCLIST": (u"\xF0\x9F\x9A\xB5", ur"""\U0001F6B5"""
#     ),
#     u"NO PEDESTRIANS": (u"\xF0\x9F\x9A\xB7", ur"""\U0001F6B7"""
#     ),
#     u"CHILDREN CROSSING": (u"\xF0\x9F\x9A\xB8", ur"""\U0001F6B8"""
#     ),
#     u"SHOWER": (u"\xF0\x9F\x9A\xBF", ur"""\U0001F6BF"""
#     ),
#     u"BATHTUB": (u"\xF0\x9F\x9B\x81", ur"""\U0001F6C1"""
#     ),
#     u"PASSPORT CONTROL": (u"\xF0\x9F\x9B\x82", ur"""\U0001F6C2"""
#     ),
#     u"CUSTOMS": (u"\xF0\x9F\x9B\x83", ur"""\U0001F6C3"""
#     ),
#     u"BAGGAGE CLAIM": (u"\xF0\x9F\x9B\x84", ur"""\U0001F6C4"""
#     ),
#     u"LEFT LUGGAGE": (u"\xF0\x9F\x9B\x85", ur"""\U0001F6C5"""
#     ),
#     u"EARTH GLOBE EUROPE-AFRICA": (u"\xF0\x9F\x8C\x8D", ur"""\U0001F30D"""
#     ),
#     u"EARTH GLOBE AMERICAS": (u"\xF0\x9F\x8C\x8E", ur"""\U0001F30E"""
#     ),
#     u"GLOBE WITH MERIDIANS": (u"\xF0\x9F\x8C\x90", ur"""\U0001F310"""
#     ),
#     u"WAXING CRESCENT MOON SYMBOL": (u"\xF0\x9F\x8C\x92", ur"""\U0001F312"""
#     ),
#     u"WANING GIBBOUS MOON SYMBOL": (u"\xF0\x9F\x8C\x96", ur"""\U0001F316"""
#     ),
#     u"LAST QUARTER MOON SYMBOL": (u"\xF0\x9F\x8C\x97", ur"""\U0001F317"""
#     ),
#     u"WANING CRESCENT MOON SYMBOL": (u"\xF0\x9F\x8C\x98", ur"""\U0001F318"""
#     ),
#     u"NEW MOON WITH FACE": (u"\xF0\x9F\x8C\x9A", ur"""\U0001F31A"""
#     ),
#     u"LAST QUARTER MOON WITH FACE": (u"\xF0\x9F\x8C\x9C", ur"""\U0001F31C"""
#     ),
#     u"FULL MOON WITH FACE": (u"\xF0\x9F\x8C\x9D", ur"""\U0001F31D"""
#     ),
#     u"SUN WITH FACE": (u"\xF0\x9F\x8C\x9E", ur"""\U0001F31E"""
#     ),
#     u"EVERGREEN TREE": (u"\xF0\x9F\x8C\xB2", ur"""\U0001F332"""
#     ),
#     u"DECIDUOUS TREE": (u"\xF0\x9F\x8C\xB3", ur"""\U0001F333"""
#     ),
#     u"LEMON": (u"\xF0\x9F\x8D\x8B", ur"""\U0001F34B"""
#     ),
#     u"PEAR": (u"\xF0\x9F\x8D\x90", ur"""\U0001F350"""
#     ),
#     u"BABY BOTTLE": (u"\xF0\x9F\x8D\xBC", ur"""\U0001F37C"""
#     ),
#     u"HORSE RACING": (u"\xF0\x9F\x8F\x87", ur"""\U0001F3C7"""
#     ),
#     u"RUGBY FOOTBALL": (u"\xF0\x9F\x8F\x89", ur"""\U0001F3C9"""
#     ),
#     u"EUROPEAN POST OFFICE": (u"\xF0\x9F\x8F\xA4", ur"""\U0001F3E4"""
#     ),
#     u"RAT": (u"\xF0\x9F\x90\x80", ur"""\U0001F400"""
#     ),
#     u"MOUSE": (u"\xF0\x9F\x90\x81", ur"""\U0001F401"""
#     ),
#     u"OX": (u"\xF0\x9F\x90\x82", ur"""\U0001F402"""
#     ),
#     u"WATER BUFFALO": (u"\xF0\x9F\x90\x83", ur"""\U0001F403"""
#     ),
#     u"COW": (u"\xF0\x9F\x90\x84", ur"""\U0001F404"""
#     ),
#     u"TIGER": (u"\xF0\x9F\x90\x85", ur"""\U0001F405"""
#     ),
#     u"LEOPARD": (u"\xF0\x9F\x90\x86", ur"""\U0001F406"""
#     ),
#     u"RABBIT": (u"\xF0\x9F\x90\x87", ur"""\U0001F407"""
#     ),
#     u"CAT": (u"\xF0\x9F\x90\x88", ur"""\U0001F408"""
#     ),
#     u"DRAGON": (u"\xF0\x9F\x90\x89", ur"""\U0001F409"""
#     ),
#     u"CROCODILE": (u"\xF0\x9F\x90\x8A", ur"""\U0001F40A"""
#     ),
#     u"WHALE": (u"\xF0\x9F\x90\x8B", ur"""\U0001F40B"""
#     ),
#     u"RAM": (u"\xF0\x9F\x90\x8F", ur"""\U0001F40F"""
#     ),
#     u"GOAT": (u"\xF0\x9F\x90\x90", ur"""\U0001F410"""
#     ),
#     u"ROOSTER": (u"\xF0\x9F\x90\x93", ur"""\U0001F413"""
#     ),
#     u"DOG": (u"\xF0\x9F\x90\x95", ur"""\U0001F415"""
#     ),
#     u"PIG": (u"\xF0\x9F\x90\x96", ur"""\U0001F416"""
#     ),
#     u"DROMEDARY CAMEL": (u"\xF0\x9F\x90\xAA", ur"""\U0001F42A"""
#     ),
#     u"BUSTS IN SILHOUETTE": (u"\xF0\x9F\x91\xA5", ur"""\U0001F465"""
#     ),
#     u"TWO MEN HOLDING HANDS": (u"\xF0\x9F\x91\xAC", ur"""\U0001F46C"""
#     ),
#     u"TWO WOMEN HOLDING HANDS": (u"\xF0\x9F\x91\xAD", ur"""\U0001F46D"""
#     ),
#     u"THOUGHT BALLOON": (u"\xF0\x9F\x92\xAD", ur"""\U0001F4AD"""
#     ),
#     u"BANKNOTE WITH EURO SIGN": (u"\xF0\x9F\x92\xB6", ur"""\U0001F4B6"""
#     ),
#     u"BANKNOTE WITH POUND SIGN": (u"\xF0\x9F\x92\xB7", ur"""\U0001F4B7"""
#     ),
#     u"OPEN MAILBOX WITH RAISED FLAG": (u"\xF0\x9F\x93\xAC", ur"""\U0001F4EC"""
#     ),
#     u"OPEN MAILBOX WITH LOWERED FLAG": (u"\xF0\x9F\x93\xAD", ur"""\U0001F4ED"""
#     ),
#     u"POSTAL HORN": (u"\xF0\x9F\x93\xAF", ur"""\U0001F4EF"""
#     ),
#     u"NO MOBILE PHONES": (u"\xF0\x9F\x93\xB5", ur"""\U0001F4F5"""
#     ),
#     u"TWISTED RIGHTWARDS ARROWS": (u"\xF0\x9F\x94\x80", ur"""\U0001F500"""
#     ),
#     u"CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x81", ur"""\U0001F501"""
#     ),
#     u"CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS WITH CIRCLED ONE OVERLAY": (u"\xF0\x9F\x94\x82", ur"""\U0001F502"""
#     ),
#     u"ANTICLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x84", ur"""\U0001F504"""
#     ),
#     u"LOW BRIGHTNESS SYMBOL": (u"\xF0\x9F\x94\x85", ur"""\U0001F505"""
#     ),
#     u"HIGH BRIGHTNESS SYMBOL": (u"\xF0\x9F\x94\x86", ur"""\U0001F506"""
#     ),
#     u"SPEAKER WITH CANCELLATION STROKE": (u"\xF0\x9F\x94\x87", ur"""\U0001F507"""
#     ),
#     u"SPEAKER WITH ONE SOUND WAVE": (u"\xF0\x9F\x94\x89", ur"""\U0001F509"""
#     ),
#     u"BELL WITH CANCELLATION STROKE": (u"\xF0\x9F\x94\x95", ur"""\U0001F515"""
#     ),
#     u"MICROSCOPE": (u"\xF0\x9F\x94\xAC", ur"""\U0001F52C"""
#     ),
#     u"TELESCOPE": (u"\xF0\x9F\x94\xAD", ur"""\U0001F52D"""
#     ),
#     u"CLOCK FACE ONE-THIRTY": (u"\xF0\x9F\x95\x9C", ur"""\U0001F55C"""
#     ),
#     u"CLOCK FACE TWO-THIRTY": (u"\xF0\x9F\x95\x9D", ur"""\U0001F55D"""
#     ),
#     u"CLOCK FACE THREE-THIRTY": (u"\xF0\x9F\x95\x9E", ur"""\U0001F55E"""
#     ),
#     u"CLOCK FACE FOUR-THIRTY": (u"\xF0\x9F\x95\x9F", ur"""\U0001F55F"""
#     ),
#     u"CLOCK FACE FIVE-THIRTY": (u"\xF0\x9F\x95\xA0", ur"""\U0001F560"""
#     ),
#     u"CLOCK FACE SIX-THIRTY": (u"\xF0\x9F\x95\xA1", ur"""\U0001F561"""
#     ),
#     u"CLOCK FACE SEVEN-THIRTY": (u"\xF0\x9F\x95\xA2", ur"""\U0001F562"""
#     ),
#     u"CLOCK FACE EIGHT-THIRTY": (u"\xF0\x9F\x95\xA3", ur"""\U0001F563"""
#     ),
#     u"CLOCK FACE NINE-THIRTY": (u"\xF0\x9F\x95\xA4", ur"""\U0001F564"""
#     ),
#     u"CLOCK FACE TEN-THIRTY": (u"\xF0\x9F\x95\xA5", ur"""\U0001F565"""
#     ),
#     u"CLOCK FACE ELEVEN-THIRTY": (u"\xF0\x9F\x95\xA6", ur"""\U0001F566"""
#     ),
#     u"CLOCK FACE TWELVE-THIRTY": (u"\xF0\x9F\x95\xA7", ur"""\U0001F567"""
#     ),

}



REGEX_FEATURE_CONFIG_ALL = {
    # ---- Emoticons ----
    u"Emoticon Happy": (u":-)", 
                        r"""[:=][o-]?[)}>\]]|               # :-) :o) :)
                        [({<\[][o-]?[:=]|                   # (-: (o: (:
                        \^(_*|[-oO]?)\^                     # ^^ ^-^
                        """
    ), 
    u"Emoticon Laughing": (u":-D", r"""([:=][-]?|x)[D]"""),   # :-D xD
    u"Emoticon Winking": (u";-)", 
                        r"""[;\*][-o]?[)}>\]]|              # ;-) ;o) ;)
                        [({<\[][-o]?[;\*]                   # (-; (
                        """
    ), 
    u"Emotion Tongue": (u":-P", 
                        r"""[:=][-]?[pqP](?!\w)|            # :-P :P
                        (?<!\w)[pqP][-]?[:=]                # q-: P-:
                        """
    ),  
    u"Emoticon Surprise": (u":-O", 
                            r"""(?<!\w|\.)                  # Boundary
                            ([:=]-?[oO0]|                   # :-O
                            [oO0]-?[:=]|                    # O-:
                            [oO](_*|\.)[oO])                # Oo O____o O.o
                            (?!\w)
                            """
    ), 
    u"Emoticon Dissatisfied": (u":-/", 
                                r"""(?<!\w)                 # Boundary
                                [:=][-o]?[\/\\|]|           # :-/ :-\ :-| :/
                                [\/\\|][-o]?[:=]|           # \-: \:
                                -_+-                        # -_- -___-
                                """
    ), 
    u"Emoticon Sad": (u":-(", 
                        r"""[:=][o-]?[({<\[]|               # :-( :(
                        (?<!(\w|%))                         # Boundary
                        [)}>\[][o-]?[:=]                    # )-: ): )o: 
                        """
    ), 
    u"Emoticon Crying": (u";-(", 
                        r"""(([:=]')|(;'?))[o-]?[({<\[]|    # ;-( :'(
                        (?<!(\w|%))                         # Boundary
                        [)}>\[][o-]?(('[:=])|('?;))         # )-; )-';
                        """
    ), 
      
    # ---- Punctuation----
    # u"AllPunctuation": (u"", r"""((\.{2,}|[?!]{2,})1*)"""),
    u"Question Mark": (u"??", r"""\?{2,}"""),                 # ??
    u"Exclamation Mark": (u"!!", r"""\!{2,}"""),              # !!
    u"Question and Exclamation Mark": (u"?!", r"""[\!\?]*((\?\!)+|              # ?!
                    (\!\?)+)[\!\?]*                         # !?
                    """
    ),                                          # Unicode interrobang: U+203D
    u"Ellipsis": (u"...", r"""\.{2,}|                         # .. ...
                \.(\ \.){2,}                                # . . .
                """
    ),                                          # Unicode Ellipsis: U+2026
    # ---- Markup----
#     u"Hashtag": (u"#", r"""\#(irony|ironic|sarcasm|sarcastic)"""),  # #irony
#     u"Pseudo-Tag": (u"Tag", 
#                     r"""([<\[][\/\\]
#                     (irony|ironic|sarcasm|sarcastic)        # </irony>
#                     [>\]])|                                 #
#                     ((?<!(\w|[<\[]))[\/\\]                  #
#                     (irony|ironic|sarcasm|sarcastic)        # /irony
#                     (?![>\]]))
#                     """
#     ),
  
    # ---- Acronyms, onomatopoeia ----
    u"Acroym for Laughter": (u"lol", 
                    r"""(?<!\w)                             # Boundary
                    (l(([oua]|aw)l)+([sz]?|wut)|            # lol, lawl, luls
                    rot?fl(mf?ao)?)|                        # rofl, roflmao
                    lmf?ao                                  # lmao, lmfao
                    (?!\w)                                  # Boundary
                    """
    ),                                    
    u"Acronym for Grin": (u"*g*", 
                        r"""\*([Gg]{1,2}|                   # *g* *gg*
                        grin)\*                             # *grin*
                        """
    ),
    u"Onomatopoeia for Laughter": (u"haha", 
                        r"""(?<!\w)                         # Boundary
                        (mu|ba)?                            # mu- ba-
                        (ha|h(e|3)|hi){2,}                  # haha, hehe, hihi
                        (?!\w)                              # Boundary
                        """
    ),
    u"Interjection": (u"ITJ", 
                        r"""(?<!\w)((a+h+a?)|               # ah, aha
                        (e+h+)|                             # eh
                        (u+g?h+)|                           # ugh
                        (huh)|                              # huh
                        ([uo]h( |-)h?[uo]h)|                # uh huh, 
                        (m*hm+)                             # hmm, mhm
                        |(h(u|r)?mp(h|f))|                  # hmpf
                        (ar+gh+)|                           # argh
                        (wow+))(?!\w)                       # wow
                        """
    ),
    u"GRINNING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x81", ur"""\U0001F601"""
    ),
    u"FACE WITH TEARS OF JOY": (u"\xF0\x9F\x98\x82", ur"""\U0001F602"""
    ),
    u"SMILING FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\x83", ur"""\U0001F603"""
    ),
    u"SMILING FACE WITH OPEN MOUTH AND SMILING EYES": (u"\xF0\x9F\x98\x84", ur"""\U0001F604"""
    ),
    u"SMILING FACE WITH OPEN MOUTH AND COLD SWEAT": (u"\xF0\x9F\x98\x85", ur"""\U0001F605"""
    ),
    u"SMILING FACE WITH OPEN MOUTH AND TIGHTLY-CLOSED EYES": (u"\xF0\x9F\x98\x86", ur"""\U0001F606"""
    ),
    u"WINKING FACE": (u"\xF0\x9F\x98\x89", ur"""\U0001F609"""
    ),
    u"SMILING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x8A", ur"""\U0001F60A"""
    ),
    u"FACE SAVOURING DELICIOUS FOOD": (u"\xF0\x9F\x98\x8B", ur"""\U0001F60B"""
    ),
    u"RELIEVED FACE": (u"\xF0\x9F\x98\x8C", ur"""\U0001F60C"""
    ),
    u"SMILING FACE WITH HEART-SHAPED EYES": (u"\xF0\x9F\x98\x8D", ur"""\U0001F60D"""
    ),
    u"SMIRKING FACE": (u"\xF0\x9F\x98\x8F", ur"""\U0001F60F"""
    ),
    u"UNAMUSED FACE": (u"\xF0\x9F\x98\x92", ur"""\U0001F612"""
    ),
    u"FACE WITH COLD SWEAT": (u"\xF0\x9F\x98\x93", ur"""\U0001F613"""
    ),
    u"PENSIVE FACE": (u"\xF0\x9F\x98\x94", ur"""\U0001F614"""
    ),
    u"CONFOUNDED FACE": (u"\xF0\x9F\x98\x96", ur"""\U0001F616"""
    ),
    u"FACE THROWING A KISS": (u"\xF0\x9F\x98\x98", ur"""\U0001F618"""
    ),
    u"KISSING FACE WITH CLOSED EYES": (u"\xF0\x9F\x98\x9A", ur"""\U0001F61A"""
    ),
    u"FACE WITH STUCK-OUT TONGUE AND WINKING EYE": (u"\xF0\x9F\x98\x9C", ur"""\U0001F61C"""
    ),
    u"FACE WITH STUCK-OUT TONGUE AND TIGHTLY-CLOSED EYES": (u"\xF0\x9F\x98\x9D", ur"""\U0001F61D"""
    ),
    u"DISAPPOINTED FACE": (u"\xF0\x9F\x98\x9E", ur"""\U0001F61E"""
    ),
    u"ANGRY FACE": (u"\xF0\x9F\x98\xA0", ur"""\U0001F620"""
    ),
    u"POUTING FACE": (u"\xF0\x9F\x98\xA1", ur"""\U0001F621"""
    ),
    u"CRYING FACE": (u"\xF0\x9F\x98\xA2", ur"""\U0001F622"""
    ),
    u"PERSEVERING FACE": (u"\xF0\x9F\x98\xA3", ur"""\U0001F623"""
    ),
    u"FACE WITH LOOK OF TRIUMPH": (u"\xF0\x9F\x98\xA4", ur"""\U0001F624"""
    ),
    u"DISAPPOINTED BUT RELIEVED FACE": (u"\xF0\x9F\x98\xA5", ur"""\U0001F625"""
    ),
    u"FEARFUL FACE": (u"\xF0\x9F\x98\xA8", ur"""\U0001F628"""
    ),
    u"WEARY FACE": (u"\xF0\x9F\x98\xA9", ur"""\U0001F629"""
    ),
    u"SLEEPY FACE": (u"\xF0\x9F\x98\xAA", ur"""\U0001F62A"""
    ),
    u"TIRED FACE": (u"\xF0\x9F\x98\xAB", ur"""\U0001F62B"""
    ),
    u"LOUDLY CRYING FACE": (u"\xF0\x9F\x98\xAD", ur"""\U0001F62D"""
    ),
    u"FACE WITH OPEN MOUTH AND COLD SWEAT": (u"\xF0\x9F\x98\xB0", ur"""\U0001F630"""
    ),
    u"FACE SCREAMING IN FEAR": (u"\xF0\x9F\x98\xB1", ur"""\U0001F631"""
    ),
    u"ASTONISHED FACE": (u"\xF0\x9F\x98\xB2", ur"""\U0001F632"""
    ),
    u"FLUSHED FACE": (u"\xF0\x9F\x98\xB3", ur"""\U0001F633"""
    ),
    u"DIZZY FACE": (u"\xF0\x9F\x98\xB5", ur"""\U0001F635"""
    ),
    u"FACE WITH MEDICAL MASK": (u"\xF0\x9F\x98\xB7", ur"""\U0001F637"""
    ),
    u"GRINNING CAT FACE WITH SMILING EYES": (u"\xF0\x9F\x98\xB8", ur"""\U0001F638"""
    ),
    u"CAT FACE WITH TEARS OF JOY": (u"\xF0\x9F\x98\xB9", ur"""\U0001F639"""
    ),
    u"SMILING CAT FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xBA", ur"""\U0001F63A"""
    ),
    u"SMILING CAT FACE WITH HEART-SHAPED EYES": (u"\xF0\x9F\x98\xBB", ur"""\U0001F63B"""
    ),
    u"CAT FACE WITH WRY SMILE": (u"\xF0\x9F\x98\xBC", ur"""\U0001F63C"""
    ),
    u"KISSING CAT FACE WITH CLOSED EYES": (u"\xF0\x9F\x98\xBD", ur"""\U0001F63D"""
    ),
    u"POUTING CAT FACE": (u"\xF0\x9F\x98\xBE", ur"""\U0001F63E"""
    ),
    u"CRYING CAT FACE": (u"\xF0\x9F\x98\xBF", ur"""\U0001F63F"""
    ),
    u"WEARY CAT FACE": (u"\xF0\x9F\x99\x80", ur"""\U0001F640"""
    ),
    u"FACE WITH NO GOOD GESTURE": (u"\xF0\x9F\x99\x85", ur"""\U0001F645"""
    ),
    u"FACE WITH OK GESTURE": (u"\xF0\x9F\x99\x86", ur"""\U0001F646"""
    ),
    u"PERSON BOWING DEEPLY": (u"\xF0\x9F\x99\x87", ur"""\U0001F647"""
    ),
    u"SEE-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x88", ur"""\U0001F648"""
    ),
    u"HEAR-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x89", ur"""\U0001F649"""
    ),
    u"SPEAK-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x8A", ur"""\U0001F64A"""
    ),
    u"HAPPY PERSON RAISING ONE HAND": (u"\xF0\x9F\x99\x8B", ur"""\U0001F64B"""
    ),
    u"PERSON RAISING BOTH HANDS IN CELEBRATION": (u"\xF0\x9F\x99\x8C", ur"""\U0001F64C"""
    ),
    u"PERSON FROWNING": (u"\xF0\x9F\x99\x8D", ur"""\U0001F64D"""
    ),
    u"PERSON WITH POUTING FACE": (u"\xF0\x9F\x99\x8E", ur"""\U0001F64E"""
    ),
    u"PERSON WITH FOLDED HANDS": (u"\xF0\x9F\x99\x8F", ur"""\U0001F64F"""
    ),
    u"BLACK SCISSORS": (u"\xE2\x9C\x82", ur"""\U00002702"""
    ),
    u"WHITE HEAVY CHECK MARK": (u"\xE2\x9C\x85", ur"""\U00002705"""
    ),
    u"AIRPLANE": (u"\xE2\x9C\x88", ur"""\U00002708"""
    ),
    u"ENVELOPE": (u"\xE2\x9C\x89", ur"""\U00002709"""
    ),
    u"RAISED FIST": (u"\xE2\x9C\x8A", ur"""\U0000270A"""
    ),
    u"RAISED HAND": (u"\xE2\x9C\x8B", ur"""\U0000270B"""
    ),
    u"VICTORY HAND": (u"\xE2\x9C\x8C", ur"""\U0000270C"""
    ),
    u"PENCIL": (u"\xE2\x9C\x8F", ur"""\U0000270F"""
    ),
    u"BLACK NIB": (u"\xE2\x9C\x92", ur"""\U00002712"""
    ),
    u"HEAVY CHECK MARK": (u"\xE2\x9C\x94", ur"""\U00002714"""
    ),
    u"HEAVY MULTIPLICATION X": (u"\xE2\x9C\x96", ur"""\U00002716"""
    ),
    u"SPARKLES": (u"\xE2\x9C\xA8", ur"""\U00002728"""
    ),
    u"EIGHT SPOKED ASTERISK": (u"\xE2\x9C\xB3", ur"""\U00002733"""
    ),
    u"EIGHT POINTED BLACK STAR": (u"\xE2\x9C\xB4", ur"""\U00002734"""
    ),
    u"SNOWFLAKE": (u"\xE2\x9D\x84", ur"""\U00002744"""
    ),
    u"SPARKLE": (u"\xE2\x9D\x87", ur"""\U00002747"""
    ),
    u"CROSS MARK": (u"\xE2\x9D\x8C", ur"""\U0000274C"""
    ),
    u"NEGATIVE SQUARED CROSS MARK": (u"\xE2\x9D\x8E", ur"""\U0000274E"""
    ),
    u"BLACK QUESTION MARK ORNAMENT": (u"\xE2\x9D\x93", ur"""\U00002753"""
    ),
    u"WHITE QUESTION MARK ORNAMENT": (u"\xE2\x9D\x94", ur"""\U00002754"""
    ),
    u"WHITE EXCLAMATION MARK ORNAMENT": (u"\xE2\x9D\x95", ur"""\U00002755"""
    ),
    u"HEAVY EXCLAMATION MARK SYMBOL": (u"\xE2\x9D\x97", ur"""\U00002757"""
    ),
    u"HEAVY BLACK HEART": (u"\xE2\x9D\xA4", ur"""\U00002764"""
    ),
    u"HEAVY PLUS SIGN": (u"\xE2\x9E\x95", ur"""\U00002795"""
    ),
    u"HEAVY MINUS SIGN": (u"\xE2\x9E\x96", ur"""\U00002796"""
    ),
    u"HEAVY DIVISION SIGN": (u"\xE2\x9E\x97", ur"""\U00002797"""
    ),
    u"BLACK RIGHTWARDS ARROW": (u"\xE2\x9E\xA1", ur"""\U000027A1"""
    ),
    u"CURLY LOOP": (u"\xE2\x9E\xB0", ur"""\U000027B0"""
    ),
    u"ROCKET": (u"\xF0\x9F\x9A\x80", ur"""\U0001F680"""
    ),
    u"RAILWAY CAR": (u"\xF0\x9F\x9A\x83", ur"""\U0001F683"""
    ),
    u"HIGH-SPEED TRAIN": (u"\xF0\x9F\x9A\x84", ur"""\U0001F684"""
    ),
    u"HIGH-SPEED TRAIN WITH BULLET NOSE": (u"\xF0\x9F\x9A\x85", ur"""\U0001F685"""
    ),
    u"METRO": (u"\xF0\x9F\x9A\x87", ur"""\U0001F687"""
    ),
    u"STATION": (u"\xF0\x9F\x9A\x89", ur"""\U0001F689"""
    ),
    u"BUS": (u"\xF0\x9F\x9A\x8C", ur"""\U0001F68C"""
    ),
    u"BUS STOP": (u"\xF0\x9F\x9A\x8F", ur"""\U0001F68F"""
    ),
    u"AMBULANCE": (u"\xF0\x9F\x9A\x91", ur"""\U0001F691"""
    ),
    u"FIRE ENGINE": (u"\xF0\x9F\x9A\x92", ur"""\U0001F692"""
    ),
    u"POLICE CAR": (u"\xF0\x9F\x9A\x93", ur"""\U0001F693"""
    ),
    u"TAXI": (u"\xF0\x9F\x9A\x95", ur"""\U0001F695"""
    ),
    u"AUTOMOBILE": (u"\xF0\x9F\x9A\x97", ur"""\U0001F697"""
    ),
    u"RECREATIONAL VEHICLE": (u"\xF0\x9F\x9A\x99", ur"""\U0001F699"""
    ),
    u"DELIVERY TRUCK": (u"\xF0\x9F\x9A\x9A", ur"""\U0001F69A"""
    ),
    u"SHIP": (u"\xF0\x9F\x9A\xA2", ur"""\U0001F6A2"""
    ),
    u"SPEEDBOAT": (u"\xF0\x9F\x9A\xA4", ur"""\U0001F6A4"""
    ),
    u"HORIZONTAL TRAFFIC LIGHT": (u"\xF0\x9F\x9A\xA5", ur"""\U0001F6A5"""
    ),
    u"CONSTRUCTION SIGN": (u"\xF0\x9F\x9A\xA7", ur"""\U0001F6A7"""
    ),
    u"POLICE CARS REVOLVING LIGHT": (u"\xF0\x9F\x9A\xA8", ur"""\U0001F6A8"""
    ),
    u"TRIANGULAR FLAG ON POST": (u"\xF0\x9F\x9A\xA9", ur"""\U0001F6A9"""
    ),
    u"DOOR": (u"\xF0\x9F\x9A\xAA", ur"""\U0001F6AA"""
    ),
    u"NO ENTRY SIGN": (u"\xF0\x9F\x9A\xAB", ur"""\U0001F6AB"""
    ),
    u"SMOKING SYMBOL": (u"\xF0\x9F\x9A\xAC", ur"""\U0001F6AC"""
    ),
    u"NO SMOKING SYMBOL": (u"\xF0\x9F\x9A\xAD", ur"""\U0001F6AD"""
    ),
    u"BICYCLE": (u"\xF0\x9F\x9A\xB2", ur"""\U0001F6B2"""
    ),
    u"PEDESTRIAN": (u"\xF0\x9F\x9A\xB6", ur"""\U0001F6B6"""
    ),
    u"MENS SYMBOL": (u"\xF0\x9F\x9A\xB9", ur"""\U0001F6B9"""
    ),
    u"WOMENS SYMBOL": (u"\xF0\x9F\x9A\xBA", ur"""\U0001F6BA"""
    ),
    u"RESTROOM": (u"\xF0\x9F\x9A\xBB", ur"""\U0001F6BB"""
    ),
    u"BABY SYMBOL": (u"\xF0\x9F\x9A\xBC", ur"""\U0001F6BC"""
    ),
    u"TOILET": (u"\xF0\x9F\x9A\xBD", ur"""\U0001F6BD"""
    ),
    u"WATER CLOSET": (u"\xF0\x9F\x9A\xBE", ur"""\U0001F6BE"""
    ),
    u"BATH": (u"\xF0\x9F\x9B\x80", ur"""\U0001F6C0"""
    ),
    u"CIRCLED LATIN CAPITAL LETTER M": (u"\xE2\x93\x82", ur"""\U000024C2"""
    ),
    u"NEGATIVE SQUARED LATIN CAPITAL LETTER A": (u"\xF0\x9F\x85\xB0", ur"""\U0001F170"""
    ),
    u"NEGATIVE SQUARED LATIN CAPITAL LETTER B": (u"\xF0\x9F\x85\xB1", ur"""\U0001F171"""
    ),
    u"NEGATIVE SQUARED LATIN CAPITAL LETTER O": (u"\xF0\x9F\x85\xBE", ur"""\U0001F17E"""
    ),
    u"NEGATIVE SQUARED LATIN CAPITAL LETTER P": (u"\xF0\x9F\x85\xBF", ur"""\U0001F17F"""
    ),
    u"NEGATIVE SQUARED AB": (u"\xF0\x9F\x86\x8E", ur"""\U0001F18E"""
    ),
    u"SQUARED CL": (u"\xF0\x9F\x86\x91", ur"""\U0001F191"""
    ),
    u"SQUARED COOL": (u"\xF0\x9F\x86\x92", ur"""\U0001F192"""
    ),
    u"SQUARED FREE": (u"\xF0\x9F\x86\x93", ur"""\U0001F193"""
    ),
    u"SQUARED ID": (u"\xF0\x9F\x86\x94", ur"""\U0001F194"""
    ),
    u"SQUARED NEW": (u"\xF0\x9F\x86\x95", ur"""\U0001F195"""
    ),
    u"SQUARED NG": (u"\xF0\x9F\x86\x96", ur"""\U0001F196"""
    ),
    u"SQUARED OK": (u"\xF0\x9F\x86\x97", ur"""\U0001F197"""
    ),
    u"SQUARED SOS": (u"\xF0\x9F\x86\x98", ur"""\U0001F198"""
    ),
    u"SQUARED UP WITH EXCLAMATION MARK": (u"\xF0\x9F\x86\x99", ur"""\U0001F199"""
    ),
    u"SQUARED VS": (u"\xF0\x9F\x86\x9A", ur"""\U0001F19A"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER D + REGIONAL INDICATOR SYMBOL LETTER E": (u"\xF0\x9F\x87\xA9\xF0\x9F\x87\xAA", ur"""\U0001F1E9 \U0001F1EA"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER G + REGIONAL INDICATOR SYMBOL LETTER B": (u"\xF0\x9F\x87\xAC\xF0\x9F\x87\xA7", ur"""\U0001F1EC \U0001F1E7"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER C + REGIONAL INDICATOR SYMBOL LETTER N": (u"\xF0\x9F\x87\xA8\xF0\x9F\x87\xB3", ur"""\U0001F1E8 \U0001F1F3"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER J + REGIONAL INDICATOR SYMBOL LETTER P": (u"\xF0\x9F\x87\xAF\xF0\x9F\x87\xB5", ur"""\U0001F1EF \U0001F1F5"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER K + REGIONAL INDICATOR SYMBOL LETTER R": (u"\xF0\x9F\x87\xB0\xF0\x9F\x87\xB7", ur"""\U0001F1F0 \U0001F1F7"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER F + REGIONAL INDICATOR SYMBOL LETTER R": (u"\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7", ur"""\U0001F1EB \U0001F1F7"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER E + REGIONAL INDICATOR SYMBOL LETTER S": (u"\xF0\x9F\x87\xAA\xF0\x9F\x87\xB8", ur"""\U0001F1EA \U0001F1F8"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER I + REGIONAL INDICATOR SYMBOL LETTER T": (u"\xF0\x9F\x87\xAE\xF0\x9F\x87\xB9", ur"""\U0001F1EE \U0001F1F9"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER U + REGIONAL INDICATOR SYMBOL LETTER S": (u"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8", ur"""\U0001F1FA \U0001F1F8"""
    ),
    u"REGIONAL INDICATOR SYMBOL LETTER R + REGIONAL INDICATOR SYMBOL LETTER U": (u"\xF0\x9F\x87\xB7\xF0\x9F\x87\xBA", ur"""\U0001F1F7 \U0001F1FA"""
    ),
    u"SQUARED KATAKANA KOKO": (u"\xF0\x9F\x88\x81", ur"""\U0001F201"""
    ),
    u"SQUARED KATAKANA SA": (u"\xF0\x9F\x88\x82", ur"""\U0001F202"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-7121": (u"\xF0\x9F\x88\x9A", ur"""\U0001F21A"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-6307": (u"\xF0\x9F\x88\xAF", ur"""\U0001F22F"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-7981": (u"\xF0\x9F\x88\xB2", ur"""\U0001F232"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-7A7A": (u"\xF0\x9F\x88\xB3", ur"""\U0001F233"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-5408": (u"\xF0\x9F\x88\xB4", ur"""\U0001F234"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-6E80": (u"\xF0\x9F\x88\xB5", ur"""\U0001F235"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-6709": (u"\xF0\x9F\x88\xB6", ur"""\U0001F236"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-6708": (u"\xF0\x9F\x88\xB7", ur"""\U0001F237"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-7533": (u"\xF0\x9F\x88\xB8", ur"""\U0001F238"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-5272": (u"\xF0\x9F\x88\xB9", ur"""\U0001F239"""
    ),
    u"SQUARED CJK UNIFIED IDEOGRAPH-55B6": (u"\xF0\x9F\x88\xBA", ur"""\U0001F23A"""
    ),
    u"CIRCLED IDEOGRAPH ADVANTAGE": (u"\xF0\x9F\x89\x90", ur"""\U0001F250"""
    ),
    u"CIRCLED IDEOGRAPH ACCEPT": (u"\xF0\x9F\x89\x91", ur"""\U0001F251"""
    ),
    u"COPYRIGHT SIGN": (u"\xC2\xA9", ur"""\U000000A9"""
    ),
    u"REGISTERED SIGN": (u"\xC2\xAE", ur"""\U000000AE"""
    ),
    u"DOUBLE EXCLAMATION MARK": (u"\xE2\x80\xBC", ur"""\U0000203C"""
    ),
    u"EXCLAMATION QUESTION MARK": (u"\xE2\x81\x89", ur"""\U00002049"""
    ),
    u"DIGIT EIGHT + COMBINING ENCLOSING KEYCAP": (u"\x38\xE2\x83\xA3", ur"""\U00000038 \U000020E3"""
    ),
    u"DIGIT NINE + COMBINING ENCLOSING KEYCAP": (u"\x39\xE2\x83\xA3", ur"""\U00000039 \U000020E3"""
    ),
    u"DIGIT SEVEN + COMBINING ENCLOSING KEYCAP": (u"\x37\xE2\x83\xA3", ur"""\U00000037 \U000020E3"""
    ),
    u"DIGIT SIX + COMBINING ENCLOSING KEYCAP": (u"\x36\xE2\x83\xA3", ur"""\U00000036 \U000020E3"""
    ),
    u"DIGIT ONE + COMBINING ENCLOSING KEYCAP": (u"\x31\xE2\x83\xA3", ur"""\U00000031 \U000020E3"""
    ),
    u"DIGIT ZERO + COMBINING ENCLOSING KEYCAP": (u"\x30\xE2\x83\xA3", ur"""\U00000030 \U000020E3"""
    ),
    u"DIGIT TWO + COMBINING ENCLOSING KEYCAP": (u"\x32\xE2\x83\xA3", ur"""\U00000032 \U000020E3"""
    ),
    u"DIGIT THREE + COMBINING ENCLOSING KEYCAP": (u"\x33\xE2\x83\xA3", ur"""\U00000033 \U000020E3"""
    ),
    u"DIGIT FIVE + COMBINING ENCLOSING KEYCAP": (u"\x35\xE2\x83\xA3", ur"""\U00000035 \U000020E3"""
    ),
    u"DIGIT FOUR + COMBINING ENCLOSING KEYCAP": (u"\x34\xE2\x83\xA3", ur"""\U00000034 \U000020E3"""
    ),
    u"NUMBER SIGN + COMBINING ENCLOSING KEYCAP": (u"\x23\xE2\x83\xA3", ur"""\U00000023 \U000020E3"""
    ),
    u"TRADE MARK SIGN": (u"\xE2\x84\xA2", ur"""\U00002122"""
    ),
    u"INFORMATION SOURCE": (u"\xE2\x84\xB9", ur"""\U00002139"""
    ),
    u"LEFT RIGHT ARROW": (u"\xE2\x86\x94", ur"""\U00002194"""
    ),
    u"UP DOWN ARROW": (u"\xE2\x86\x95", ur"""\U00002195"""
    ),
    u"NORTH WEST ARROW": (u"\xE2\x86\x96", ur"""\U00002196"""
    ),
    u"NORTH EAST ARROW": (u"\xE2\x86\x97", ur"""\U00002197"""
    ),
    u"SOUTH EAST ARROW": (u"\xE2\x86\x98", ur"""\U00002198"""
    ),
    u"SOUTH WEST ARROW": (u"\xE2\x86\x99", ur"""\U00002199"""
    ),
    u"LEFTWARDS ARROW WITH HOOK": (u"\xE2\x86\xA9", ur"""\U000021A9"""
    ),
    u"RIGHTWARDS ARROW WITH HOOK": (u"\xE2\x86\xAA", ur"""\U000021AA"""
    ),
    u"WATCH": (u"\xE2\x8C\x9A", ur"""\U0000231A"""
    ),
    u"HOURGLASS": (u"\xE2\x8C\x9B", ur"""\U0000231B"""
    ),
    u"BLACK RIGHT-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xA9", ur"""\U000023E9"""
    ),
    u"BLACK LEFT-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAA", ur"""\U000023EA"""
    ),
    u"BLACK UP-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAB", ur"""\U000023EB"""
    ),
    u"BLACK DOWN-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAC", ur"""\U000023EC"""
    ),
    u"ALARM CLOCK": (u"\xE2\x8F\xB0", ur"""\U000023F0"""
    ),
    u"HOURGLASS WITH FLOWING SAND": (u"\xE2\x8F\xB3", ur"""\U000023F3"""
    ),
    u"BLACK SMALL SQUARE": (u"\xE2\x96\xAA", ur"""\U000025AA"""
    ),
    u"WHITE SMALL SQUARE": (u"\xE2\x96\xAB", ur"""\U000025AB"""
    ),
    u"BLACK RIGHT-POINTING TRIANGLE": (u"\xE2\x96\xB6", ur"""\U000025B6"""
    ),
    u"BLACK LEFT-POINTING TRIANGLE": (u"\xE2\x97\x80", ur"""\U000025C0"""
    ),
    u"WHITE MEDIUM SQUARE": (u"\xE2\x97\xBB", ur"""\U000025FB"""
    ),
    u"BLACK MEDIUM SQUARE": (u"\xE2\x97\xBC", ur"""\U000025FC"""
    ),
    u"WHITE MEDIUM SMALL SQUARE": (u"\xE2\x97\xBD", ur"""\U000025FD"""
    ),
    u"BLACK MEDIUM SMALL SQUARE": (u"\xE2\x97\xBE", ur"""\U000025FE"""
    ),
    u"BLACK SUN WITH RAYS": (u"\xE2\x98\x80", ur"""\U00002600"""
    ),
    u"CLOUD": (u"\xE2\x98\x81", ur"""\U00002601"""
    ),
    u"BLACK TELEPHONE": (u"\xE2\x98\x8E", ur"""\U0000260E"""
    ),
    u"BALLOT BOX WITH CHECK": (u"\xE2\x98\x91", ur"""\U00002611"""
    ),
    u"UMBRELLA WITH RAIN DROPS": (u"\xE2\x98\x94", ur"""\U00002614"""
    ),
    u"HOT BEVERAGE": (u"\xE2\x98\x95", ur"""\U00002615"""
    ),
    u"WHITE UP POINTING INDEX": (u"\xE2\x98\x9D", ur"""\U0000261D"""
    ),
    u"WHITE SMILING FACE": (u"\xE2\x98\xBA", ur"""\U0000263A"""
    ),
    u"ARIES": (u"\xE2\x99\x88", ur"""\U00002648"""
    ),
    u"TAURUS": (u"\xE2\x99\x89", ur"""\U00002649"""
    ),
    u"GEMINI": (u"\xE2\x99\x8A", ur"""\U0000264A"""
    ),
    u"CANCER": (u"\xE2\x99\x8B", ur"""\U0000264B"""
    ),
    u"LEO": (u"\xE2\x99\x8C", ur"""\U0000264C"""
    ),
    u"VIRGO": (u"\xE2\x99\x8D", ur"""\U0000264D"""
    ),
    u"LIBRA": (u"\xE2\x99\x8E", ur"""\U0000264E"""
    ),
    u"SCORPIUS": (u"\xE2\x99\x8F", ur"""\U0000264F"""
    ),
    u"SAGITTARIUS": (u"\xE2\x99\x90", ur"""\U00002650"""
    ),
    u"CAPRICORN": (u"\xE2\x99\x91", ur"""\U00002651"""
    ),
    u"AQUARIUS": (u"\xE2\x99\x92", ur"""\U00002652"""
    ),
    u"PISCES": (u"\xE2\x99\x93", ur"""\U00002653"""
    ),
    u"BLACK SPADE SUIT": (u"\xE2\x99\xA0", ur"""\U00002660"""
    ),
    u"BLACK CLUB SUIT": (u"\xE2\x99\xA3", ur"""\U00002663"""
    ),
    u"BLACK HEART SUIT": (u"\xE2\x99\xA5", ur"""\U00002665"""
    ),
    u"BLACK DIAMOND SUIT": (u"\xE2\x99\xA6", ur"""\U00002666"""
    ),
    u"HOT SPRINGS": (u"\xE2\x99\xA8", ur"""\U00002668"""
    ),
    u"BLACK UNIVERSAL RECYCLING SYMBOL": (u"\xE2\x99\xBB", ur"""\U0000267B"""
    ),
    u"WHEELCHAIR SYMBOL": (u"\xE2\x99\xBF", ur"""\U0000267F"""
    ),
    u"ANCHOR": (u"\xE2\x9A\x93", ur"""\U00002693"""
    ),
    u"WARNING SIGN": (u"\xE2\x9A\xA0", ur"""\U000026A0"""
    ),
    u"HIGH VOLTAGE SIGN": (u"\xE2\x9A\xA1", ur"""\U000026A1"""
    ),
    u"MEDIUM WHITE CIRCLE": (u"\xE2\x9A\xAA", ur"""\U000026AA"""
    ),
    u"MEDIUM BLACK CIRCLE": (u"\xE2\x9A\xAB", ur"""\U000026AB"""
    ),
    u"SOCCER BALL": (u"\xE2\x9A\xBD", ur"""\U000026BD"""
    ),
    u"BASEBALL": (u"\xE2\x9A\xBE", ur"""\U000026BE"""
    ),
    u"SNOWMAN WITHOUT SNOW": (u"\xE2\x9B\x84", ur"""\U000026C4"""
    ),
    u"SUN BEHIND CLOUD": (u"\xE2\x9B\x85", ur"""\U000026C5"""
    ),
    u"OPHIUCHUS": (u"\xE2\x9B\x8E", ur"""\U000026CE"""
    ),
    u"NO ENTRY": (u"\xE2\x9B\x94", ur"""\U000026D4"""
    ),
    u"CHURCH": (u"\xE2\x9B\xAA", ur"""\U000026EA"""
    ),
    u"FOUNTAIN": (u"\xE2\x9B\xB2", ur"""\U000026F2"""
    ),
    u"FLAG IN HOLE": (u"\xE2\x9B\xB3", ur"""\U000026F3"""
    ),
    u"SAILBOAT": (u"\xE2\x9B\xB5", ur"""\U000026F5"""
    ),
    u"TENT": (u"\xE2\x9B\xBA", ur"""\U000026FA"""
    ),
    u"FUEL PUMP": (u"\xE2\x9B\xBD", ur"""\U000026FD"""
    ),
    u"ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS": (u"\xE2\xA4\xB4", ur"""\U00002934"""
    ),
    u"ARROW POINTING RIGHTWARDS THEN CURVING DOWNWARDS": (u"\xE2\xA4\xB5", ur"""\U00002935"""
    ),
    u"LEFTWARDS BLACK ARROW": (u"\xE2\xAC\x85", ur"""\U00002B05"""
    ),
    u"UPWARDS BLACK ARROW": (u"\xE2\xAC\x86", ur"""\U00002B06"""
    ),
    u"DOWNWARDS BLACK ARROW": (u"\xE2\xAC\x87", ur"""\U00002B07"""
    ),
    u"BLACK LARGE SQUARE": (u"\xE2\xAC\x9B", ur"""\U00002B1B"""
    ),
    u"WHITE LARGE SQUARE": (u"\xE2\xAC\x9C", ur"""\U00002B1C"""
    ),
    u"WHITE MEDIUM STAR": (u"\xE2\xAD\x90", ur"""\U00002B50"""
    ),
    u"HEAVY LARGE CIRCLE": (u"\xE2\xAD\x95", ur"""\U00002B55"""
    ),
    u"WAVY DASH": (u"\xE3\x80\xB0", ur"""\U00003030"""
    ),
    u"PART ALTERNATION MARK": (u"\xE3\x80\xBD", ur"""\U0000303D"""
    ),
    u"CIRCLED IDEOGRAPH CONGRATULATION": (u"\xE3\x8A\x97", ur"""\U00003297"""
    ),
    u"CIRCLED IDEOGRAPH SECRET": (u"\xE3\x8A\x99", ur"""\U00003299"""
    ),
    u"MAHJONG TILE RED DRAGON": (u"\xF0\x9F\x80\x84", ur"""\U0001F004"""
    ),
    u"PLAYING CARD BLACK JOKER": (u"\xF0\x9F\x83\x8F", ur"""\U0001F0CF"""
    ),
    u"CYCLONE": (u"\xF0\x9F\x8C\x80", ur"""\U0001F300"""
    ),
    u"FOGGY": (u"\xF0\x9F\x8C\x81", ur"""\U0001F301"""
    ),
    u"CLOSED UMBRELLA": (u"\xF0\x9F\x8C\x82", ur"""\U0001F302"""
    ),
    u"NIGHT WITH STARS": (u"\xF0\x9F\x8C\x83", ur"""\U0001F303"""
    ),
    u"SUNRISE OVER MOUNTAINS": (u"\xF0\x9F\x8C\x84", ur"""\U0001F304"""
    ),
    u"SUNRISE": (u"\xF0\x9F\x8C\x85", ur"""\U0001F305"""
    ),
    u"CITYSCAPE AT DUSK": (u"\xF0\x9F\x8C\x86", ur"""\U0001F306"""
    ),
    u"SUNSET OVER BUILDINGS": (u"\xF0\x9F\x8C\x87", ur"""\U0001F307"""
    ),
    u"RAINBOW": (u"\xF0\x9F\x8C\x88", ur"""\U0001F308"""
    ),
    u"BRIDGE AT NIGHT": (u"\xF0\x9F\x8C\x89", ur"""\U0001F309"""
    ),
    u"WATER WAVE": (u"\xF0\x9F\x8C\x8A", ur"""\U0001F30A"""
    ),
    u"VOLCANO": (u"\xF0\x9F\x8C\x8B", ur"""\U0001F30B"""
    ),
    u"MILKY WAY": (u"\xF0\x9F\x8C\x8C", ur"""\U0001F30C"""
    ),
    u"EARTH GLOBE ASIA-AUSTRALIA": (u"\xF0\x9F\x8C\x8F", ur"""\U0001F30F"""
    ),
    u"NEW MOON SYMBOL": (u"\xF0\x9F\x8C\x91", ur"""\U0001F311"""
    ),
    u"FIRST QUARTER MOON SYMBOL": (u"\xF0\x9F\x8C\x93", ur"""\U0001F313"""
    ),
    u"WAXING GIBBOUS MOON SYMBOL": (u"\xF0\x9F\x8C\x94", ur"""\U0001F314"""
    ),
    u"FULL MOON SYMBOL": (u"\xF0\x9F\x8C\x95", ur"""\U0001F315"""
    ),
    u"CRESCENT MOON": (u"\xF0\x9F\x8C\x99", ur"""\U0001F319"""
    ),
    u"FIRST QUARTER MOON WITH FACE": (u"\xF0\x9F\x8C\x9B", ur"""\U0001F31B"""
    ),
    u"GLOWING STAR": (u"\xF0\x9F\x8C\x9F", ur"""\U0001F31F"""
    ),
    u"SHOOTING STAR": (u"\xF0\x9F\x8C\xA0", ur"""\U0001F320"""
    ),
    u"CHESTNUT": (u"\xF0\x9F\x8C\xB0", ur"""\U0001F330"""
    ),
    u"SEEDLING": (u"\xF0\x9F\x8C\xB1", ur"""\U0001F331"""
    ),
    u"PALM TREE": (u"\xF0\x9F\x8C\xB4", ur"""\U0001F334"""
    ),
    u"CACTUS": (u"\xF0\x9F\x8C\xB5", ur"""\U0001F335"""
    ),
    u"TULIP": (u"\xF0\x9F\x8C\xB7", ur"""\U0001F337"""
    ),
    u"CHERRY BLOSSOM": (u"\xF0\x9F\x8C\xB8", ur"""\U0001F338"""
    ),
    u"ROSE": (u"\xF0\x9F\x8C\xB9", ur"""\U0001F339"""
    ),
    u"HIBISCUS": (u"\xF0\x9F\x8C\xBA", ur"""\U0001F33A"""
    ),
    u"SUNFLOWER": (u"\xF0\x9F\x8C\xBB", ur"""\U0001F33B"""
    ),
    u"BLOSSOM": (u"\xF0\x9F\x8C\xBC", ur"""\U0001F33C"""
    ),
    u"EAR OF MAIZE": (u"\xF0\x9F\x8C\xBD", ur"""\U0001F33D"""
    ),
    u"EAR OF RICE": (u"\xF0\x9F\x8C\xBE", ur"""\U0001F33E"""
    ),
    u"HERB": (u"\xF0\x9F\x8C\xBF", ur"""\U0001F33F"""
    ),
    u"FOUR LEAF CLOVER": (u"\xF0\x9F\x8D\x80", ur"""\U0001F340"""
    ),
    u"MAPLE LEAF": (u"\xF0\x9F\x8D\x81", ur"""\U0001F341"""
    ),
    u"FALLEN LEAF": (u"\xF0\x9F\x8D\x82", ur"""\U0001F342"""
    ),
    u"LEAF FLUTTERING IN WIND": (u"\xF0\x9F\x8D\x83", ur"""\U0001F343"""
    ),
    u"MUSHROOM": (u"\xF0\x9F\x8D\x84", ur"""\U0001F344"""
    ),
    u"TOMATO": (u"\xF0\x9F\x8D\x85", ur"""\U0001F345"""
    ),
    u"AUBERGINE": (u"\xF0\x9F\x8D\x86", ur"""\U0001F346"""
    ),
    u"GRAPES": (u"\xF0\x9F\x8D\x87", ur"""\U0001F347"""
    ),
    u"MELON": (u"\xF0\x9F\x8D\x88", ur"""\U0001F348"""
    ),
    u"WATERMELON": (u"\xF0\x9F\x8D\x89", ur"""\U0001F349"""
    ),
    u"TANGERINE": (u"\xF0\x9F\x8D\x8A", ur"""\U0001F34A"""
    ),
    u"BANANA": (u"\xF0\x9F\x8D\x8C", ur"""\U0001F34C"""
    ),
    u"PINEAPPLE": (u"\xF0\x9F\x8D\x8D", ur"""\U0001F34D"""
    ),
    u"RED APPLE": (u"\xF0\x9F\x8D\x8E", ur"""\U0001F34E"""
    ),
    u"GREEN APPLE": (u"\xF0\x9F\x8D\x8F", ur"""\U0001F34F"""
    ),
    u"PEACH": (u"\xF0\x9F\x8D\x91", ur"""\U0001F351"""
    ),
    u"CHERRIES": (u"\xF0\x9F\x8D\x92", ur"""\U0001F352"""
    ),
    u"STRAWBERRY": (u"\xF0\x9F\x8D\x93", ur"""\U0001F353"""
    ),
    u"HAMBURGER": (u"\xF0\x9F\x8D\x94", ur"""\U0001F354"""
    ),
    u"SLICE OF PIZZA": (u"\xF0\x9F\x8D\x95", ur"""\U0001F355"""
    ),
    u"MEAT ON BONE": (u"\xF0\x9F\x8D\x96", ur"""\U0001F356"""
    ),
    u"POULTRY LEG": (u"\xF0\x9F\x8D\x97", ur"""\U0001F357"""
    ),
    u"RICE CRACKER": (u"\xF0\x9F\x8D\x98", ur"""\U0001F358"""
    ),
    u"RICE BALL": (u"\xF0\x9F\x8D\x99", ur"""\U0001F359"""
    ),
    u"COOKED RICE": (u"\xF0\x9F\x8D\x9A", ur"""\U0001F35A"""
    ),
    u"CURRY AND RICE": (u"\xF0\x9F\x8D\x9B", ur"""\U0001F35B"""
    ),
    u"STEAMING BOWL": (u"\xF0\x9F\x8D\x9C", ur"""\U0001F35C"""
    ),
    u"SPAGHETTI": (u"\xF0\x9F\x8D\x9D", ur"""\U0001F35D"""
    ),
    u"BREAD": (u"\xF0\x9F\x8D\x9E", ur"""\U0001F35E"""
    ),
    u"FRENCH FRIES": (u"\xF0\x9F\x8D\x9F", ur"""\U0001F35F"""
    ),
    u"ROASTED SWEET POTATO": (u"\xF0\x9F\x8D\xA0", ur"""\U0001F360"""
    ),
    u"DANGO": (u"\xF0\x9F\x8D\xA1", ur"""\U0001F361"""
    ),
    u"ODEN": (u"\xF0\x9F\x8D\xA2", ur"""\U0001F362"""
    ),
    u"SUSHI": (u"\xF0\x9F\x8D\xA3", ur"""\U0001F363"""
    ),
    u"FRIED SHRIMP": (u"\xF0\x9F\x8D\xA4", ur"""\U0001F364"""
    ),
    u"FISH CAKE WITH SWIRL DESIGN": (u"\xF0\x9F\x8D\xA5", ur"""\U0001F365"""
    ),
    u"SOFT ICE CREAM": (u"\xF0\x9F\x8D\xA6", ur"""\U0001F366"""
    ),
    u"SHAVED ICE": (u"\xF0\x9F\x8D\xA7", ur"""\U0001F367"""
    ),
    u"ICE CREAM": (u"\xF0\x9F\x8D\xA8", ur"""\U0001F368"""
    ),
    u"DOUGHNUT": (u"\xF0\x9F\x8D\xA9", ur"""\U0001F369"""
    ),
    u"COOKIE": (u"\xF0\x9F\x8D\xAA", ur"""\U0001F36A"""
    ),
    u"CHOCOLATE BAR": (u"\xF0\x9F\x8D\xAB", ur"""\U0001F36B"""
    ),
    u"CANDY": (u"\xF0\x9F\x8D\xAC", ur"""\U0001F36C"""
    ),
    u"LOLLIPOP": (u"\xF0\x9F\x8D\xAD", ur"""\U0001F36D"""
    ),
    u"CUSTARD": (u"\xF0\x9F\x8D\xAE", ur"""\U0001F36E"""
    ),
    u"HONEY POT": (u"\xF0\x9F\x8D\xAF", ur"""\U0001F36F"""
    ),
    u"SHORTCAKE": (u"\xF0\x9F\x8D\xB0", ur"""\U0001F370"""
    ),
    u"BENTO BOX": (u"\xF0\x9F\x8D\xB1", ur"""\U0001F371"""
    ),
    u"POT OF FOOD": (u"\xF0\x9F\x8D\xB2", ur"""\U0001F372"""
    ),
    u"COOKING": (u"\xF0\x9F\x8D\xB3", ur"""\U0001F373"""
    ),
    u"FORK AND KNIFE": (u"\xF0\x9F\x8D\xB4", ur"""\U0001F374"""
    ),
    u"TEACUP WITHOUT HANDLE": (u"\xF0\x9F\x8D\xB5", ur"""\U0001F375"""
    ),
    u"SAKE BOTTLE AND CUP": (u"\xF0\x9F\x8D\xB6", ur"""\U0001F376"""
    ),
    u"WINE GLASS": (u"\xF0\x9F\x8D\xB7", ur"""\U0001F377"""
    ),
    u"COCKTAIL GLASS": (u"\xF0\x9F\x8D\xB8", ur"""\U0001F378"""
    ),
    u"TROPICAL DRINK": (u"\xF0\x9F\x8D\xB9", ur"""\U0001F379"""
    ),
    u"BEER MUG": (u"\xF0\x9F\x8D\xBA", ur"""\U0001F37A"""
    ),
    u"CLINKING BEER MUGS": (u"\xF0\x9F\x8D\xBB", ur"""\U0001F37B"""
    ),
    u"RIBBON": (u"\xF0\x9F\x8E\x80", ur"""\U0001F380"""
    ),
    u"WRAPPED PRESENT": (u"\xF0\x9F\x8E\x81", ur"""\U0001F381"""
    ),
    u"BIRTHDAY CAKE": (u"\xF0\x9F\x8E\x82", ur"""\U0001F382"""
    ),
    u"JACK-O-LANTERN": (u"\xF0\x9F\x8E\x83", ur"""\U0001F383"""
    ),
    u"CHRISTMAS TREE": (u"\xF0\x9F\x8E\x84", ur"""\U0001F384"""
    ),
    u"FATHER CHRISTMAS": (u"\xF0\x9F\x8E\x85", ur"""\U0001F385"""
    ),
    u"FIREWORKS": (u"\xF0\x9F\x8E\x86", ur"""\U0001F386"""
    ),
    u"FIREWORK SPARKLER": (u"\xF0\x9F\x8E\x87", ur"""\U0001F387"""
    ),
    u"BALLOON": (u"\xF0\x9F\x8E\x88", ur"""\U0001F388"""
    ),
    u"PARTY POPPER": (u"\xF0\x9F\x8E\x89", ur"""\U0001F389"""
    ),
    u"CONFETTI BALL": (u"\xF0\x9F\x8E\x8A", ur"""\U0001F38A"""
    ),
    u"TANABATA TREE": (u"\xF0\x9F\x8E\x8B", ur"""\U0001F38B"""
    ),
    u"CROSSED FLAGS": (u"\xF0\x9F\x8E\x8C", ur"""\U0001F38C"""
    ),
    u"PINE DECORATION": (u"\xF0\x9F\x8E\x8D", ur"""\U0001F38D"""
    ),
    u"JAPANESE DOLLS": (u"\xF0\x9F\x8E\x8E", ur"""\U0001F38E"""
    ),
    u"CARP STREAMER": (u"\xF0\x9F\x8E\x8F", ur"""\U0001F38F"""
    ),
    u"WIND CHIME": (u"\xF0\x9F\x8E\x90", ur"""\U0001F390"""
    ),
    u"MOON VIEWING CEREMONY": (u"\xF0\x9F\x8E\x91", ur"""\U0001F391"""
    ),
    u"SCHOOL SATCHEL": (u"\xF0\x9F\x8E\x92", ur"""\U0001F392"""
    ),
    u"GRADUATION CAP": (u"\xF0\x9F\x8E\x93", ur"""\U0001F393"""
    ),
    u"CAROUSEL HORSE": (u"\xF0\x9F\x8E\xA0", ur"""\U0001F3A0"""
    ),
    u"FERRIS WHEEL": (u"\xF0\x9F\x8E\xA1", ur"""\U0001F3A1"""
    ),
    u"ROLLER COASTER": (u"\xF0\x9F\x8E\xA2", ur"""\U0001F3A2"""
    ),
    u"FISHING POLE AND FISH": (u"\xF0\x9F\x8E\xA3", ur"""\U0001F3A3"""
    ),
    u"MICROPHONE": (u"\xF0\x9F\x8E\xA4", ur"""\U0001F3A4"""
    ),
    u"MOVIE CAMERA": (u"\xF0\x9F\x8E\xA5", ur"""\U0001F3A5"""
    ),
    u"CINEMA": (u"\xF0\x9F\x8E\xA6", ur"""\U0001F3A6"""
    ),
    u"HEADPHONE": (u"\xF0\x9F\x8E\xA7", ur"""\U0001F3A7"""
    ),
    u"ARTIST PALETTE": (u"\xF0\x9F\x8E\xA8", ur"""\U0001F3A8"""
    ),
    u"TOP HAT": (u"\xF0\x9F\x8E\xA9", ur"""\U0001F3A9"""
    ),
    u"CIRCUS TENT": (u"\xF0\x9F\x8E\xAA", ur"""\U0001F3AA"""
    ),
    u"TICKET": (u"\xF0\x9F\x8E\xAB", ur"""\U0001F3AB"""
    ),
    u"CLAPPER BOARD": (u"\xF0\x9F\x8E\xAC", ur"""\U0001F3AC"""
    ),
    u"PERFORMING ARTS": (u"\xF0\x9F\x8E\xAD", ur"""\U0001F3AD"""
    ),
    u"VIDEO GAME": (u"\xF0\x9F\x8E\xAE", ur"""\U0001F3AE"""
    ),
    u"DIRECT HIT": (u"\xF0\x9F\x8E\xAF", ur"""\U0001F3AF"""
    ),
    u"SLOT MACHINE": (u"\xF0\x9F\x8E\xB0", ur"""\U0001F3B0"""
    ),
    u"BILLIARDS": (u"\xF0\x9F\x8E\xB1", ur"""\U0001F3B1"""
    ),
    u"GAME DIE": (u"\xF0\x9F\x8E\xB2", ur"""\U0001F3B2"""
    ),
    u"BOWLING": (u"\xF0\x9F\x8E\xB3", ur"""\U0001F3B3"""
    ),
    u"FLOWER PLAYING CARDS": (u"\xF0\x9F\x8E\xB4", ur"""\U0001F3B4"""
    ),
    u"MUSICAL NOTE": (u"\xF0\x9F\x8E\xB5", ur"""\U0001F3B5"""
    ),
    u"MULTIPLE MUSICAL NOTES": (u"\xF0\x9F\x8E\xB6", ur"""\U0001F3B6"""
    ),
    u"SAXOPHONE": (u"\xF0\x9F\x8E\xB7", ur"""\U0001F3B7"""
    ),
    u"GUITAR": (u"\xF0\x9F\x8E\xB8", ur"""\U0001F3B8"""
    ),
    u"MUSICAL KEYBOARD": (u"\xF0\x9F\x8E\xB9", ur"""\U0001F3B9"""
    ),
    u"TRUMPET": (u"\xF0\x9F\x8E\xBA", ur"""\U0001F3BA"""
    ),
    u"VIOLIN": (u"\xF0\x9F\x8E\xBB", ur"""\U0001F3BB"""
    ),
    u"MUSICAL SCORE": (u"\xF0\x9F\x8E\xBC", ur"""\U0001F3BC"""
    ),
    u"RUNNING SHIRT WITH SASH": (u"\xF0\x9F\x8E\xBD", ur"""\U0001F3BD"""
    ),
    u"TENNIS RACQUET AND BALL": (u"\xF0\x9F\x8E\xBE", ur"""\U0001F3BE"""
    ),
    u"SKI AND SKI BOOT": (u"\xF0\x9F\x8E\xBF", ur"""\U0001F3BF"""
    ),
    u"BASKETBALL AND HOOP": (u"\xF0\x9F\x8F\x80", ur"""\U0001F3C0"""
    ),
    u"CHEQUERED FLAG": (u"\xF0\x9F\x8F\x81", ur"""\U0001F3C1"""
    ),
    u"SNOWBOARDER": (u"\xF0\x9F\x8F\x82", ur"""\U0001F3C2"""
    ),
    u"RUNNER": (u"\xF0\x9F\x8F\x83", ur"""\U0001F3C3"""
    ),
    u"SURFER": (u"\xF0\x9F\x8F\x84", ur"""\U0001F3C4"""
    ),
    u"TROPHY": (u"\xF0\x9F\x8F\x86", ur"""\U0001F3C6"""
    ),
    u"AMERICAN FOOTBALL": (u"\xF0\x9F\x8F\x88", ur"""\U0001F3C8"""
    ),
    u"SWIMMER": (u"\xF0\x9F\x8F\x8A", ur"""\U0001F3CA"""
    ),
    u"HOUSE BUILDING": (u"\xF0\x9F\x8F\xA0", ur"""\U0001F3E0"""
    ),
    u"HOUSE WITH GARDEN": (u"\xF0\x9F\x8F\xA1", ur"""\U0001F3E1"""
    ),
    u"OFFICE BUILDING": (u"\xF0\x9F\x8F\xA2", ur"""\U0001F3E2"""
    ),
    u"JAPANESE POST OFFICE": (u"\xF0\x9F\x8F\xA3", ur"""\U0001F3E3"""
    ),
    u"HOSPITAL": (u"\xF0\x9F\x8F\xA5", ur"""\U0001F3E5"""
    ),
    u"BANK": (u"\xF0\x9F\x8F\xA6", ur"""\U0001F3E6"""
    ),
    u"AUTOMATED TELLER MACHINE": (u"\xF0\x9F\x8F\xA7", ur"""\U0001F3E7"""
    ),
    u"HOTEL": (u"\xF0\x9F\x8F\xA8", ur"""\U0001F3E8"""
    ),
    u"LOVE HOTEL": (u"\xF0\x9F\x8F\xA9", ur"""\U0001F3E9"""
    ),
    u"CONVENIENCE STORE": (u"\xF0\x9F\x8F\xAA", ur"""\U0001F3EA"""
    ),
    u"SCHOOL": (u"\xF0\x9F\x8F\xAB", ur"""\U0001F3EB"""
    ),
    u"DEPARTMENT STORE": (u"\xF0\x9F\x8F\xAC", ur"""\U0001F3EC"""
    ),
    u"FACTORY": (u"\xF0\x9F\x8F\xAD", ur"""\U0001F3ED"""
    ),
    u"IZAKAYA LANTERN": (u"\xF0\x9F\x8F\xAE", ur"""\U0001F3EE"""
    ),
    u"JAPANESE CASTLE": (u"\xF0\x9F\x8F\xAF", ur"""\U0001F3EF"""
    ),
    u"EUROPEAN CASTLE": (u"\xF0\x9F\x8F\xB0", ur"""\U0001F3F0"""
    ),
    u"SNAIL": (u"\xF0\x9F\x90\x8C", ur"""\U0001F40C"""
    ),
    u"SNAKE": (u"\xF0\x9F\x90\x8D", ur"""\U0001F40D"""
    ),
    u"HORSE": (u"\xF0\x9F\x90\x8E", ur"""\U0001F40E"""
    ),
    u"SHEEP": (u"\xF0\x9F\x90\x91", ur"""\U0001F411"""
    ),
    u"MONKEY": (u"\xF0\x9F\x90\x92", ur"""\U0001F412"""
    ),
    u"CHICKEN": (u"\xF0\x9F\x90\x94", ur"""\U0001F414"""
    ),
    u"BOAR": (u"\xF0\x9F\x90\x97", ur"""\U0001F417"""
    ),
    u"ELEPHANT": (u"\xF0\x9F\x90\x98", ur"""\U0001F418"""
    ),
    u"OCTOPUS": (u"\xF0\x9F\x90\x99", ur"""\U0001F419"""
    ),
    u"SPIRAL SHELL": (u"\xF0\x9F\x90\x9A", ur"""\U0001F41A"""
    ),
    u"BUG": (u"\xF0\x9F\x90\x9B", ur"""\U0001F41B"""
    ),
    u"ANT": (u"\xF0\x9F\x90\x9C", ur"""\U0001F41C"""
    ),
    u"HONEYBEE": (u"\xF0\x9F\x90\x9D", ur"""\U0001F41D"""
    ),
    u"LADY BEETLE": (u"\xF0\x9F\x90\x9E", ur"""\U0001F41E"""
    ),
    u"FISH": (u"\xF0\x9F\x90\x9F", ur"""\U0001F41F"""
    ),
    u"TROPICAL FISH": (u"\xF0\x9F\x90\xA0", ur"""\U0001F420"""
    ),
    u"BLOWFISH": (u"\xF0\x9F\x90\xA1", ur"""\U0001F421"""
    ),
    u"TURTLE": (u"\xF0\x9F\x90\xA2", ur"""\U0001F422"""
    ),
    u"HATCHING CHICK": (u"\xF0\x9F\x90\xA3", ur"""\U0001F423"""
    ),
    u"BABY CHICK": (u"\xF0\x9F\x90\xA4", ur"""\U0001F424"""
    ),
    u"FRONT-FACING BABY CHICK": (u"\xF0\x9F\x90\xA5", ur"""\U0001F425"""
    ),
    u"BIRD": (u"\xF0\x9F\x90\xA6", ur"""\U0001F426"""
    ),
    u"PENGUIN": (u"\xF0\x9F\x90\xA7", ur"""\U0001F427"""
    ),
    u"KOALA": (u"\xF0\x9F\x90\xA8", ur"""\U0001F428"""
    ),
    u"POODLE": (u"\xF0\x9F\x90\xA9", ur"""\U0001F429"""
    ),
    u"BACTRIAN CAMEL": (u"\xF0\x9F\x90\xAB", ur"""\U0001F42B"""
    ),
    u"DOLPHIN": (u"\xF0\x9F\x90\xAC", ur"""\U0001F42C"""
    ),
    u"MOUSE FACE": (u"\xF0\x9F\x90\xAD", ur"""\U0001F42D"""
    ),
    u"COW FACE": (u"\xF0\x9F\x90\xAE", ur"""\U0001F42E"""
    ),
    u"TIGER FACE": (u"\xF0\x9F\x90\xAF", ur"""\U0001F42F"""
    ),
    u"RABBIT FACE": (u"\xF0\x9F\x90\xB0", ur"""\U0001F430"""
    ),
    u"CAT FACE": (u"\xF0\x9F\x90\xB1", ur"""\U0001F431"""
    ),
    u"DRAGON FACE": (u"\xF0\x9F\x90\xB2", ur"""\U0001F432"""
    ),
    u"SPOUTING WHALE": (u"\xF0\x9F\x90\xB3", ur"""\U0001F433"""
    ),
    u"HORSE FACE": (u"\xF0\x9F\x90\xB4", ur"""\U0001F434"""
    ),
    u"MONKEY FACE": (u"\xF0\x9F\x90\xB5", ur"""\U0001F435"""
    ),
    u"DOG FACE": (u"\xF0\x9F\x90\xB6", ur"""\U0001F436"""
    ),
    u"PIG FACE": (u"\xF0\x9F\x90\xB7", ur"""\U0001F437"""
    ),
    u"FROG FACE": (u"\xF0\x9F\x90\xB8", ur"""\U0001F438"""
    ),
    u"HAMSTER FACE": (u"\xF0\x9F\x90\xB9", ur"""\U0001F439"""
    ),
    u"WOLF FACE": (u"\xF0\x9F\x90\xBA", ur"""\U0001F43A"""
    ),
    u"BEAR FACE": (u"\xF0\x9F\x90\xBB", ur"""\U0001F43B"""
    ),
    u"PANDA FACE": (u"\xF0\x9F\x90\xBC", ur"""\U0001F43C"""
    ),
    u"PIG NOSE": (u"\xF0\x9F\x90\xBD", ur"""\U0001F43D"""
    ),
    u"PAW PRINTS": (u"\xF0\x9F\x90\xBE", ur"""\U0001F43E"""
    ),
    u"EYES": (u"\xF0\x9F\x91\x80", ur"""\U0001F440"""
    ),
    u"EAR": (u"\xF0\x9F\x91\x82", ur"""\U0001F442"""
    ),
    u"NOSE": (u"\xF0\x9F\x91\x83", ur"""\U0001F443"""
    ),
    u"MOUTH": (u"\xF0\x9F\x91\x84", ur"""\U0001F444"""
    ),
    u"TONGUE": (u"\xF0\x9F\x91\x85", ur"""\U0001F445"""
    ),
    u"WHITE UP POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x86", ur"""\U0001F446"""
    ),
    u"WHITE DOWN POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x87", ur"""\U0001F447"""
    ),
    u"WHITE LEFT POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x88", ur"""\U0001F448"""
    ),
    u"WHITE RIGHT POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x89", ur"""\U0001F449"""
    ),
    u"FISTED HAND SIGN": (u"\xF0\x9F\x91\x8A", ur"""\U0001F44A"""
    ),
    u"WAVING HAND SIGN": (u"\xF0\x9F\x91\x8B", ur"""\U0001F44B"""
    ),
    u"OK HAND SIGN": (u"\xF0\x9F\x91\x8C", ur"""\U0001F44C"""
    ),
    u"THUMBS UP SIGN": (u"\xF0\x9F\x91\x8D", ur"""\U0001F44D"""
    ),
    u"THUMBS DOWN SIGN": (u"\xF0\x9F\x91\x8E", ur"""\U0001F44E"""
    ),
    u"CLAPPING HANDS SIGN": (u"\xF0\x9F\x91\x8F", ur"""\U0001F44F"""
    ),
    u"OPEN HANDS SIGN": (u"\xF0\x9F\x91\x90", ur"""\U0001F450"""
    ),
    u"CROWN": (u"\xF0\x9F\x91\x91", ur"""\U0001F451"""
    ),
    u"WOMANS HAT": (u"\xF0\x9F\x91\x92", ur"""\U0001F452"""
    ),
    u"EYEGLASSES": (u"\xF0\x9F\x91\x93", ur"""\U0001F453"""
    ),
    u"NECKTIE": (u"\xF0\x9F\x91\x94", ur"""\U0001F454"""
    ),
    u"T-SHIRT": (u"\xF0\x9F\x91\x95", ur"""\U0001F455"""
    ),
    u"JEANS": (u"\xF0\x9F\x91\x96", ur"""\U0001F456"""
    ),
    u"DRESS": (u"\xF0\x9F\x91\x97", ur"""\U0001F457"""
    ),
    u"KIMONO": (u"\xF0\x9F\x91\x98", ur"""\U0001F458"""
    ),
    u"BIKINI": (u"\xF0\x9F\x91\x99", ur"""\U0001F459"""
    ),
    u"WOMANS CLOTHES": (u"\xF0\x9F\x91\x9A", ur"""\U0001F45A"""
    ),
    u"PURSE": (u"\xF0\x9F\x91\x9B", ur"""\U0001F45B"""
    ),
    u"HANDBAG": (u"\xF0\x9F\x91\x9C", ur"""\U0001F45C"""
    ),
    u"POUCH": (u"\xF0\x9F\x91\x9D", ur"""\U0001F45D"""
    ),
    u"MANS SHOE": (u"\xF0\x9F\x91\x9E", ur"""\U0001F45E"""
    ),
    u"ATHLETIC SHOE": (u"\xF0\x9F\x91\x9F", ur"""\U0001F45F"""
    ),
    u"HIGH-HEELED SHOE": (u"\xF0\x9F\x91\xA0", ur"""\U0001F460"""
    ),
    u"WOMANS SANDAL": (u"\xF0\x9F\x91\xA1", ur"""\U0001F461"""
    ),
    u"WOMANS BOOTS": (u"\xF0\x9F\x91\xA2", ur"""\U0001F462"""
    ),
    u"FOOTPRINTS": (u"\xF0\x9F\x91\xA3", ur"""\U0001F463"""
    ),
    u"BUST IN SILHOUETTE": (u"\xF0\x9F\x91\xA4", ur"""\U0001F464"""
    ),
    u"BOY": (u"\xF0\x9F\x91\xA6", ur"""\U0001F466"""
    ),
    u"GIRL": (u"\xF0\x9F\x91\xA7", ur"""\U0001F467"""
    ),
    u"MAN": (u"\xF0\x9F\x91\xA8", ur"""\U0001F468"""
    ),
    u"WOMAN": (u"\xF0\x9F\x91\xA9", ur"""\U0001F469"""
    ),
    u"FAMILY": (u"\xF0\x9F\x91\xAA", ur"""\U0001F46A"""
    ),
    u"MAN AND WOMAN HOLDING HANDS": (u"\xF0\x9F\x91\xAB", ur"""\U0001F46B"""
    ),
    u"POLICE OFFICER": (u"\xF0\x9F\x91\xAE", ur"""\U0001F46E"""
    ),
    u"WOMAN WITH BUNNY EARS": (u"\xF0\x9F\x91\xAF", ur"""\U0001F46F"""
    ),
    u"BRIDE WITH VEIL": (u"\xF0\x9F\x91\xB0", ur"""\U0001F470"""
    ),
    u"PERSON WITH BLOND HAIR": (u"\xF0\x9F\x91\xB1", ur"""\U0001F471"""
    ),
    u"MAN WITH GUA PI MAO": (u"\xF0\x9F\x91\xB2", ur"""\U0001F472"""
    ),
    u"MAN WITH TURBAN": (u"\xF0\x9F\x91\xB3", ur"""\U0001F473"""
    ),
    u"OLDER MAN": (u"\xF0\x9F\x91\xB4", ur"""\U0001F474"""
    ),
    u"OLDER WOMAN": (u"\xF0\x9F\x91\xB5", ur"""\U0001F475"""
    ),
    u"BABY": (u"\xF0\x9F\x91\xB6", ur"""\U0001F476"""
    ),
    u"CONSTRUCTION WORKER": (u"\xF0\x9F\x91\xB7", ur"""\U0001F477"""
    ),
    u"PRINCESS": (u"\xF0\x9F\x91\xB8", ur"""\U0001F478"""
    ),
    u"JAPANESE OGRE": (u"\xF0\x9F\x91\xB9", ur"""\U0001F479"""
    ),
    u"JAPANESE GOBLIN": (u"\xF0\x9F\x91\xBA", ur"""\U0001F47A"""
    ),
    u"GHOST": (u"\xF0\x9F\x91\xBB", ur"""\U0001F47B"""
    ),
    u"BABY ANGEL": (u"\xF0\x9F\x91\xBC", ur"""\U0001F47C"""
    ),
    u"EXTRATERRESTRIAL ALIEN": (u"\xF0\x9F\x91\xBD", ur"""\U0001F47D"""
    ),
    u"ALIEN MONSTER": (u"\xF0\x9F\x91\xBE", ur"""\U0001F47E"""
    ),
    u"IMP": (u"\xF0\x9F\x91\xBF", ur"""\U0001F47F"""
    ),
    u"SKULL": (u"\xF0\x9F\x92\x80", ur"""\U0001F480"""
    ),
    u"INFORMATION DESK PERSON": (u"\xF0\x9F\x92\x81", ur"""\U0001F481"""
    ),
    u"GUARDSMAN": (u"\xF0\x9F\x92\x82", ur"""\U0001F482"""
    ),
    u"DANCER": (u"\xF0\x9F\x92\x83", ur"""\U0001F483"""
    ),
    u"LIPSTICK": (u"\xF0\x9F\x92\x84", ur"""\U0001F484"""
    ),
    u"NAIL POLISH": (u"\xF0\x9F\x92\x85", ur"""\U0001F485"""
    ),
    u"FACE MASSAGE": (u"\xF0\x9F\x92\x86", ur"""\U0001F486"""
    ),
    u"HAIRCUT": (u"\xF0\x9F\x92\x87", ur"""\U0001F487"""
    ),
    u"BARBER POLE": (u"\xF0\x9F\x92\x88", ur"""\U0001F488"""
    ),
    u"SYRINGE": (u"\xF0\x9F\x92\x89", ur"""\U0001F489"""
    ),
    u"PILL": (u"\xF0\x9F\x92\x8A", ur"""\U0001F48A"""
    ),
    u"KISS MARK": (u"\xF0\x9F\x92\x8B", ur"""\U0001F48B"""
    ),
    u"LOVE LETTER": (u"\xF0\x9F\x92\x8C", ur"""\U0001F48C"""
    ),
    u"RING": (u"\xF0\x9F\x92\x8D", ur"""\U0001F48D"""
    ),
    u"GEM STONE": (u"\xF0\x9F\x92\x8E", ur"""\U0001F48E"""
    ),
    u"KISS": (u"\xF0\x9F\x92\x8F", ur"""\U0001F48F"""
    ),
    u"BOUQUET": (u"\xF0\x9F\x92\x90", ur"""\U0001F490"""
    ),
    u"COUPLE WITH HEART": (u"\xF0\x9F\x92\x91", ur"""\U0001F491"""
    ),
    u"WEDDING": (u"\xF0\x9F\x92\x92", ur"""\U0001F492"""
    ),
    u"BEATING HEART": (u"\xF0\x9F\x92\x93", ur"""\U0001F493"""
    ),
    u"BROKEN HEART": (u"\xF0\x9F\x92\x94", ur"""\U0001F494"""
    ),
    u"TWO HEARTS": (u"\xF0\x9F\x92\x95", ur"""\U0001F495"""
    ),
    u"SPARKLING HEART": (u"\xF0\x9F\x92\x96", ur"""\U0001F496"""
    ),
    u"GROWING HEART": (u"\xF0\x9F\x92\x97", ur"""\U0001F497"""
    ),
    u"HEART WITH ARROW": (u"\xF0\x9F\x92\x98", ur"""\U0001F498"""
    ),
    u"BLUE HEART": (u"\xF0\x9F\x92\x99", ur"""\U0001F499"""
    ),
    u"GREEN HEART": (u"\xF0\x9F\x92\x9A", ur"""\U0001F49A"""
    ),
    u"YELLOW HEART": (u"\xF0\x9F\x92\x9B", ur"""\U0001F49B"""
    ),
    u"PURPLE HEART": (u"\xF0\x9F\x92\x9C", ur"""\U0001F49C"""
    ),
    u"HEART WITH RIBBON": (u"\xF0\x9F\x92\x9D", ur"""\U0001F49D"""
    ),
    u"REVOLVING HEARTS": (u"\xF0\x9F\x92\x9E", ur"""\U0001F49E"""
    ),
    u"HEART DECORATION": (u"\xF0\x9F\x92\x9F", ur"""\U0001F49F"""
    ),
    u"DIAMOND SHAPE WITH A DOT INSIDE": (u"\xF0\x9F\x92\xA0", ur"""\U0001F4A0"""
    ),
    u"ELECTRIC LIGHT BULB": (u"\xF0\x9F\x92\xA1", ur"""\U0001F4A1"""
    ),
    u"ANGER SYMBOL": (u"\xF0\x9F\x92\xA2", ur"""\U0001F4A2"""
    ),
    u"BOMB": (u"\xF0\x9F\x92\xA3", ur"""\U0001F4A3"""
    ),
    u"SLEEPING SYMBOL": (u"\xF0\x9F\x92\xA4", ur"""\U0001F4A4"""
    ),
    u"COLLISION SYMBOL": (u"\xF0\x9F\x92\xA5", ur"""\U0001F4A5"""
    ),
    u"SPLASHING SWEAT SYMBOL": (u"\xF0\x9F\x92\xA6", ur"""\U0001F4A6"""
    ),
    u"DROPLET": (u"\xF0\x9F\x92\xA7", ur"""\U0001F4A7"""
    ),
    u"DASH SYMBOL": (u"\xF0\x9F\x92\xA8", ur"""\U0001F4A8"""
    ),
    u"PILE OF POO": (u"\xF0\x9F\x92\xA9", ur"""\U0001F4A9"""
    ),
    u"FLEXED BICEPS": (u"\xF0\x9F\x92\xAA", ur"""\U0001F4AA"""
    ),
    u"DIZZY SYMBOL": (u"\xF0\x9F\x92\xAB", ur"""\U0001F4AB"""
    ),
    u"SPEECH BALLOON": (u"\xF0\x9F\x92\xAC", ur"""\U0001F4AC"""
    ),
    u"WHITE FLOWER": (u"\xF0\x9F\x92\xAE", ur"""\U0001F4AE"""
    ),
    u"HUNDRED POINTS SYMBOL": (u"\xF0\x9F\x92\xAF", ur"""\U0001F4AF"""
    ),
    u"MONEY BAG": (u"\xF0\x9F\x92\xB0", ur"""\U0001F4B0"""
    ),
    u"CURRENCY EXCHANGE": (u"\xF0\x9F\x92\xB1", ur"""\U0001F4B1"""
    ),
    u"HEAVY DOLLAR SIGN": (u"\xF0\x9F\x92\xB2", ur"""\U0001F4B2"""
    ),
    u"CREDIT CARD": (u"\xF0\x9F\x92\xB3", ur"""\U0001F4B3"""
    ),
    u"BANKNOTE WITH YEN SIGN": (u"\xF0\x9F\x92\xB4", ur"""\U0001F4B4"""
    ),
    u"BANKNOTE WITH DOLLAR SIGN": (u"\xF0\x9F\x92\xB5", ur"""\U0001F4B5"""
    ),
    u"MONEY WITH WINGS": (u"\xF0\x9F\x92\xB8", ur"""\U0001F4B8"""
    ),
    u"CHART WITH UPWARDS TREND AND YEN SIGN": (u"\xF0\x9F\x92\xB9", ur"""\U0001F4B9"""
    ),
    u"SEAT": (u"\xF0\x9F\x92\xBA", ur"""\U0001F4BA"""
    ),
    u"PERSONAL COMPUTER": (u"\xF0\x9F\x92\xBB", ur"""\U0001F4BB"""
    ),
    u"BRIEFCASE": (u"\xF0\x9F\x92\xBC", ur"""\U0001F4BC"""
    ),
    u"MINIDISC": (u"\xF0\x9F\x92\xBD", ur"""\U0001F4BD"""
    ),
    u"FLOPPY DISK": (u"\xF0\x9F\x92\xBE", ur"""\U0001F4BE"""
    ),
    u"OPTICAL DISC": (u"\xF0\x9F\x92\xBF", ur"""\U0001F4BF"""
    ),
    u"DVD": (u"\xF0\x9F\x93\x80", ur"""\U0001F4C0"""
    ),
    u"FILE FOLDER": (u"\xF0\x9F\x93\x81", ur"""\U0001F4C1"""
    ),
    u"OPEN FILE FOLDER": (u"\xF0\x9F\x93\x82", ur"""\U0001F4C2"""
    ),
    u"PAGE WITH CURL": (u"\xF0\x9F\x93\x83", ur"""\U0001F4C3"""
    ),
    u"PAGE FACING UP": (u"\xF0\x9F\x93\x84", ur"""\U0001F4C4"""
    ),
    u"CALENDAR": (u"\xF0\x9F\x93\x85", ur"""\U0001F4C5"""
    ),
    u"TEAR-OFF CALENDAR": (u"\xF0\x9F\x93\x86", ur"""\U0001F4C6"""
    ),
    u"CARD INDEX": (u"\xF0\x9F\x93\x87", ur"""\U0001F4C7"""
    ),
    u"CHART WITH UPWARDS TREND": (u"\xF0\x9F\x93\x88", ur"""\U0001F4C8"""
    ),
    u"CHART WITH DOWNWARDS TREND": (u"\xF0\x9F\x93\x89", ur"""\U0001F4C9"""
    ),
    u"BAR CHART": (u"\xF0\x9F\x93\x8A", ur"""\U0001F4CA"""
    ),
    u"CLIPBOARD": (u"\xF0\x9F\x93\x8B", ur"""\U0001F4CB"""
    ),
    u"PUSHPIN": (u"\xF0\x9F\x93\x8C", ur"""\U0001F4CC"""
    ),
    u"ROUND PUSHPIN": (u"\xF0\x9F\x93\x8D", ur"""\U0001F4CD"""
    ),
    u"PAPERCLIP": (u"\xF0\x9F\x93\x8E", ur"""\U0001F4CE"""
    ),
    u"STRAIGHT RULER": (u"\xF0\x9F\x93\x8F", ur"""\U0001F4CF"""
    ),
    u"TRIANGULAR RULER": (u"\xF0\x9F\x93\x90", ur"""\U0001F4D0"""
    ),
    u"BOOKMARK TABS": (u"\xF0\x9F\x93\x91", ur"""\U0001F4D1"""
    ),
    u"LEDGER": (u"\xF0\x9F\x93\x92", ur"""\U0001F4D2"""
    ),
    u"NOTEBOOK": (u"\xF0\x9F\x93\x93", ur"""\U0001F4D3"""
    ),
    u"NOTEBOOK WITH DECORATIVE COVER": (u"\xF0\x9F\x93\x94", ur"""\U0001F4D4"""
    ),
    u"CLOSED BOOK": (u"\xF0\x9F\x93\x95", ur"""\U0001F4D5"""
    ),
    u"OPEN BOOK": (u"\xF0\x9F\x93\x96", ur"""\U0001F4D6"""
    ),
    u"GREEN BOOK": (u"\xF0\x9F\x93\x97", ur"""\U0001F4D7"""
    ),
    u"BLUE BOOK": (u"\xF0\x9F\x93\x98", ur"""\U0001F4D8"""
    ),
    u"ORANGE BOOK": (u"\xF0\x9F\x93\x99", ur"""\U0001F4D9"""
    ),
    u"BOOKS": (u"\xF0\x9F\x93\x9A", ur"""\U0001F4DA"""
    ),
    u"NAME BADGE": (u"\xF0\x9F\x93\x9B", ur"""\U0001F4DB"""
    ),
    u"SCROLL": (u"\xF0\x9F\x93\x9C", ur"""\U0001F4DC"""
    ),
    u"MEMO": (u"\xF0\x9F\x93\x9D", ur"""\U0001F4DD"""
    ),
    u"TELEPHONE RECEIVER": (u"\xF0\x9F\x93\x9E", ur"""\U0001F4DE"""
    ),
    u"PAGER": (u"\xF0\x9F\x93\x9F", ur"""\U0001F4DF"""
    ),
    u"FAX MACHINE": (u"\xF0\x9F\x93\xA0", ur"""\U0001F4E0"""
    ),
    u"SATELLITE ANTENNA": (u"\xF0\x9F\x93\xA1", ur"""\U0001F4E1"""
    ),
    u"PUBLIC ADDRESS LOUDSPEAKER": (u"\xF0\x9F\x93\xA2", ur"""\U0001F4E2"""
    ),
    u"CHEERING MEGAPHONE": (u"\xF0\x9F\x93\xA3", ur"""\U0001F4E3"""
    ),
    u"OUTBOX TRAY": (u"\xF0\x9F\x93\xA4", ur"""\U0001F4E4"""
    ),
    u"INBOX TRAY": (u"\xF0\x9F\x93\xA5", ur"""\U0001F4E5"""
    ),
    u"PACKAGE": (u"\xF0\x9F\x93\xA6", ur"""\U0001F4E6"""
    ),
    u"E-MAIL SYMBOL": (u"\xF0\x9F\x93\xA7", ur"""\U0001F4E7"""
    ),
    u"INCOMING ENVELOPE": (u"\xF0\x9F\x93\xA8", ur"""\U0001F4E8"""
    ),
    u"ENVELOPE WITH DOWNWARDS ARROW ABOVE": (u"\xF0\x9F\x93\xA9", ur"""\U0001F4E9"""
    ),
    u"CLOSED MAILBOX WITH LOWERED FLAG": (u"\xF0\x9F\x93\xAA", ur"""\U0001F4EA"""
    ),
    u"CLOSED MAILBOX WITH RAISED FLAG": (u"\xF0\x9F\x93\xAB", ur"""\U0001F4EB"""
    ),
    u"POSTBOX": (u"\xF0\x9F\x93\xAE", ur"""\U0001F4EE"""
    ),
    u"NEWSPAPER": (u"\xF0\x9F\x93\xB0", ur"""\U0001F4F0"""
    ),
    u"MOBILE PHONE": (u"\xF0\x9F\x93\xB1", ur"""\U0001F4F1"""
    ),
    u"MOBILE PHONE WITH RIGHTWARDS ARROW AT LEFT": (u"\xF0\x9F\x93\xB2", ur"""\U0001F4F2"""
    ),
    u"VIBRATION MODE": (u"\xF0\x9F\x93\xB3", ur"""\U0001F4F3"""
    ),
    u"MOBILE PHONE OFF": (u"\xF0\x9F\x93\xB4", ur"""\U0001F4F4"""
    ),
    u"ANTENNA WITH BARS": (u"\xF0\x9F\x93\xB6", ur"""\U0001F4F6"""
    ),
    u"CAMERA": (u"\xF0\x9F\x93\xB7", ur"""\U0001F4F7"""
    ),
    u"VIDEO CAMERA": (u"\xF0\x9F\x93\xB9", ur"""\U0001F4F9"""
    ),
    u"TELEVISION": (u"\xF0\x9F\x93\xBA", ur"""\U0001F4FA"""
    ),
    u"RADIO": (u"\xF0\x9F\x93\xBB", ur"""\U0001F4FB"""
    ),
    u"VIDEOCASSETTE": (u"\xF0\x9F\x93\xBC", ur"""\U0001F4FC"""
    ),
    u"CLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x83", ur"""\U0001F503"""
    ),
    u"SPEAKER WITH THREE SOUND WAVES": (u"\xF0\x9F\x94\x8A", ur"""\U0001F50A"""
    ),
    u"BATTERY": (u"\xF0\x9F\x94\x8B", ur"""\U0001F50B"""
    ),
    u"ELECTRIC PLUG": (u"\xF0\x9F\x94\x8C", ur"""\U0001F50C"""
    ),
    u"LEFT-POINTING MAGNIFYING GLASS": (u"\xF0\x9F\x94\x8D", ur"""\U0001F50D"""
    ),
    u"RIGHT-POINTING MAGNIFYING GLASS": (u"\xF0\x9F\x94\x8E", ur"""\U0001F50E"""
    ),
    u"LOCK WITH INK PEN": (u"\xF0\x9F\x94\x8F", ur"""\U0001F50F"""
    ),
    u"CLOSED LOCK WITH KEY": (u"\xF0\x9F\x94\x90", ur"""\U0001F510"""
    ),
    u"KEY": (u"\xF0\x9F\x94\x91", ur"""\U0001F511"""
    ),
    u"LOCK": (u"\xF0\x9F\x94\x92", ur"""\U0001F512"""
    ),
    u"OPEN LOCK": (u"\xF0\x9F\x94\x93", ur"""\U0001F513"""
    ),
    u"BELL": (u"\xF0\x9F\x94\x94", ur"""\U0001F514"""
    ),
    u"BOOKMARK": (u"\xF0\x9F\x94\x96", ur"""\U0001F516"""
    ),
    u"LINK SYMBOL": (u"\xF0\x9F\x94\x97", ur"""\U0001F517"""
    ),
    u"RADIO BUTTON": (u"\xF0\x9F\x94\x98", ur"""\U0001F518"""
    ),
    u"BACK WITH LEFTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x99", ur"""\U0001F519"""
    ),
    u"END WITH LEFTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9A", ur"""\U0001F51A"""
    ),
    u"ON WITH EXCLAMATION MARK WITH LEFT RIGHT ARROW ABOVE": (u"\xF0\x9F\x94\x9B", ur"""\U0001F51B"""
    ),
    u"SOON WITH RIGHTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9C", ur"""\U0001F51C"""
    ),
    u"TOP WITH UPWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9D", ur"""\U0001F51D"""
    ),
    u"NO ONE UNDER EIGHTEEN SYMBOL": (u"\xF0\x9F\x94\x9E", ur"""\U0001F51E"""
    ),
    u"KEYCAP TEN": (u"\xF0\x9F\x94\x9F", ur"""\U0001F51F"""
    ),
    u"INPUT SYMBOL FOR LATIN CAPITAL LETTERS": (u"\xF0\x9F\x94\xA0", ur"""\U0001F520"""
    ),
    u"INPUT SYMBOL FOR LATIN SMALL LETTERS": (u"\xF0\x9F\x94\xA1", ur"""\U0001F521"""
    ),
    u"INPUT SYMBOL FOR NUMBERS": (u"\xF0\x9F\x94\xA2", ur"""\U0001F522"""
    ),
    u"INPUT SYMBOL FOR SYMBOLS": (u"\xF0\x9F\x94\xA3", ur"""\U0001F523"""
    ),
    u"INPUT SYMBOL FOR LATIN LETTERS": (u"\xF0\x9F\x94\xA4", ur"""\U0001F524"""
    ),
    u"FIRE": (u"\xF0\x9F\x94\xA5", ur"""\U0001F525"""
    ),
    u"ELECTRIC TORCH": (u"\xF0\x9F\x94\xA6", ur"""\U0001F526"""
    ),
    u"WRENCH": (u"\xF0\x9F\x94\xA7", ur"""\U0001F527"""
    ),
    u"HAMMER": (u"\xF0\x9F\x94\xA8", ur"""\U0001F528"""
    ),
    u"NUT AND BOLT": (u"\xF0\x9F\x94\xA9", ur"""\U0001F529"""
    ),
    u"HOCHO": (u"\xF0\x9F\x94\xAA", ur"""\U0001F52A"""
    ),
    u"PISTOL": (u"\xF0\x9F\x94\xAB", ur"""\U0001F52B"""
    ),
    u"CRYSTAL BALL": (u"\xF0\x9F\x94\xAE", ur"""\U0001F52E"""
    ),
    u"SIX POINTED STAR WITH MIDDLE DOT": (u"\xF0\x9F\x94\xAF", ur"""\U0001F52F"""
    ),
    u"JAPANESE SYMBOL FOR BEGINNER": (u"\xF0\x9F\x94\xB0", ur"""\U0001F530"""
    ),
    u"TRIDENT EMBLEM": (u"\xF0\x9F\x94\xB1", ur"""\U0001F531"""
    ),
    u"BLACK SQUARE BUTTON": (u"\xF0\x9F\x94\xB2", ur"""\U0001F532"""
    ),
    u"WHITE SQUARE BUTTON": (u"\xF0\x9F\x94\xB3", ur"""\U0001F533"""
    ),
    u"LARGE RED CIRCLE": (u"\xF0\x9F\x94\xB4", ur"""\U0001F534"""
    ),
    u"LARGE BLUE CIRCLE": (u"\xF0\x9F\x94\xB5", ur"""\U0001F535"""
    ),
    u"LARGE ORANGE DIAMOND": (u"\xF0\x9F\x94\xB6", ur"""\U0001F536"""
    ),
    u"LARGE BLUE DIAMOND": (u"\xF0\x9F\x94\xB7", ur"""\U0001F537"""
    ),
    u"SMALL ORANGE DIAMOND": (u"\xF0\x9F\x94\xB8", ur"""\U0001F538"""
    ),
    u"SMALL BLUE DIAMOND": (u"\xF0\x9F\x94\xB9", ur"""\U0001F539"""
    ),
    u"UP-POINTING RED TRIANGLE": (u"\xF0\x9F\x94\xBA", ur"""\U0001F53A"""
    ),
    u"DOWN-POINTING RED TRIANGLE": (u"\xF0\x9F\x94\xBB", ur"""\U0001F53B"""
    ),
    u"UP-POINTING SMALL RED TRIANGLE": (u"\xF0\x9F\x94\xBC", ur"""\U0001F53C"""
    ),
    u"DOWN-POINTING SMALL RED TRIANGLE": (u"\xF0\x9F\x94\xBD", ur"""\U0001F53D"""
    ),
    u"CLOCK FACE ONE OCLOCK": (u"\xF0\x9F\x95\x90", ur"""\U0001F550"""
    ),
    u"CLOCK FACE TWO OCLOCK": (u"\xF0\x9F\x95\x91", ur"""\U0001F551"""
    ),
    u"CLOCK FACE THREE OCLOCK": (u"\xF0\x9F\x95\x92", ur"""\U0001F552"""
    ),
    u"CLOCK FACE FOUR OCLOCK": (u"\xF0\x9F\x95\x93", ur"""\U0001F553"""
    ),
    u"CLOCK FACE FIVE OCLOCK": (u"\xF0\x9F\x95\x94", ur"""\U0001F554"""
    ),
    u"CLOCK FACE SIX OCLOCK": (u"\xF0\x9F\x95\x95", ur"""\U0001F555"""
    ),
    u"CLOCK FACE SEVEN OCLOCK": (u"\xF0\x9F\x95\x96", ur"""\U0001F556"""
    ),
    u"CLOCK FACE EIGHT OCLOCK": (u"\xF0\x9F\x95\x97", ur"""\U0001F557"""
    ),
    u"CLOCK FACE NINE OCLOCK": (u"\xF0\x9F\x95\x98", ur"""\U0001F558"""
    ),
    u"CLOCK FACE TEN OCLOCK": (u"\xF0\x9F\x95\x99", ur"""\U0001F559"""
    ),
    u"CLOCK FACE ELEVEN OCLOCK": (u"\xF0\x9F\x95\x9A", ur"""\U0001F55A"""
    ),
    u"CLOCK FACE TWELVE OCLOCK": (u"\xF0\x9F\x95\x9B", ur"""\U0001F55B"""
    ),
    u"MOUNT FUJI": (u"\xF0\x9F\x97\xBB", ur"""\U0001F5FB"""
    ),
    u"TOKYO TOWER": (u"\xF0\x9F\x97\xBC", ur"""\U0001F5FC"""
    ),
    u"STATUE OF LIBERTY": (u"\xF0\x9F\x97\xBD", ur"""\U0001F5FD"""
    ),
    u"SILHOUETTE OF JAPAN": (u"\xF0\x9F\x97\xBE", ur"""\U0001F5FE"""
    ),
    u"MOYAI": (u"\xF0\x9F\x97\xBF", ur"""\U0001F5FF"""
    ),
    u"GRINNING FACE": (u"\xF0\x9F\x98\x80", ur"""\U0001F600"""
    ),
    u"SMILING FACE WITH HALO": (u"\xF0\x9F\x98\x87", ur"""\U0001F607"""
    ),
    u"SMILING FACE WITH HORNS": (u"\xF0\x9F\x98\x88", ur"""\U0001F608"""
    ),
    u"SMILING FACE WITH SUNGLASSES": (u"\xF0\x9F\x98\x8E", ur"""\U0001F60E"""
    ),
    u"NEUTRAL FACE": (u"\xF0\x9F\x98\x90", ur"""\U0001F610"""
    ),
    u"EXPRESSIONLESS FACE": (u"\xF0\x9F\x98\x91", ur"""\U0001F611"""
    ),
    u"CONFUSED FACE": (u"\xF0\x9F\x98\x95", ur"""\U0001F615"""
    ),
    u"KISSING FACE": (u"\xF0\x9F\x98\x97", ur"""\U0001F617"""
    ),
    u"KISSING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x99", ur"""\U0001F619"""
    ),
    u"FACE WITH STUCK-OUT TONGUE": (u"\xF0\x9F\x98\x9B", ur"""\U0001F61B"""
    ),
    u"WORRIED FACE": (u"\xF0\x9F\x98\x9F", ur"""\U0001F61F"""
    ),
    u"FROWNING FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xA6", ur"""\U0001F626"""
    ),
    u"ANGUISHED FACE": (u"\xF0\x9F\x98\xA7", ur"""\U0001F627"""
    ),
    u"GRIMACING FACE": (u"\xF0\x9F\x98\xAC", ur"""\U0001F62C"""
    ),
    u"FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xAE", ur"""\U0001F62E"""
    ),
    u"HUSHED FACE": (u"\xF0\x9F\x98\xAF", ur"""\U0001F62F"""
    ),
    u"SLEEPING FACE": (u"\xF0\x9F\x98\xB4", ur"""\U0001F634"""
    ),
    u"FACE WITHOUT MOUTH": (u"\xF0\x9F\x98\xB6", ur"""\U0001F636"""
    ),
    u"HELICOPTER": (u"\xF0\x9F\x9A\x81", ur"""\U0001F681"""
    ),
    u"STEAM LOCOMOTIVE": (u"\xF0\x9F\x9A\x82", ur"""\U0001F682"""
    ),
    u"TRAIN": (u"\xF0\x9F\x9A\x86", ur"""\U0001F686"""
    ),
    u"LIGHT RAIL": (u"\xF0\x9F\x9A\x88", ur"""\U0001F688"""
    ),
    u"TRAM": (u"\xF0\x9F\x9A\x8A", ur"""\U0001F68A"""
    ),
    u"ONCOMING BUS": (u"\xF0\x9F\x9A\x8D", ur"""\U0001F68D"""
    ),
    u"TROLLEYBUS": (u"\xF0\x9F\x9A\x8E", ur"""\U0001F68E"""
    ),
    u"MINIBUS": (u"\xF0\x9F\x9A\x90", ur"""\U0001F690"""
    ),
    u"ONCOMING POLICE CAR": (u"\xF0\x9F\x9A\x94", ur"""\U0001F694"""
    ),
    u"ONCOMING TAXI": (u"\xF0\x9F\x9A\x96", ur"""\U0001F696"""
    ),
    u"ONCOMING AUTOMOBILE": (u"\xF0\x9F\x9A\x98", ur"""\U0001F698"""
    ),
    u"ARTICULATED LORRY": (u"\xF0\x9F\x9A\x9B", ur"""\U0001F69B"""
    ),
    u"TRACTOR": (u"\xF0\x9F\x9A\x9C", ur"""\U0001F69C"""
    ),
    u"MONORAIL": (u"\xF0\x9F\x9A\x9D", ur"""\U0001F69D"""
    ),
    u"MOUNTAIN RAILWAY": (u"\xF0\x9F\x9A\x9E", ur"""\U0001F69E"""
    ),
    u"SUSPENSION RAILWAY": (u"\xF0\x9F\x9A\x9F", ur"""\U0001F69F"""
    ),
    u"MOUNTAIN CABLEWAY": (u"\xF0\x9F\x9A\xA0", ur"""\U0001F6A0"""
    ),
    u"AERIAL TRAMWAY": (u"\xF0\x9F\x9A\xA1", ur"""\U0001F6A1"""
    ),
    u"ROWBOAT": (u"\xF0\x9F\x9A\xA3", ur"""\U0001F6A3"""
    ),
    u"VERTICAL TRAFFIC LIGHT": (u"\xF0\x9F\x9A\xA6", ur"""\U0001F6A6"""
    ),
    u"PUT LITTER IN ITS PLACE SYMBOL": (u"\xF0\x9F\x9A\xAE", ur"""\U0001F6AE"""
    ),
    u"DO NOT LITTER SYMBOL": (u"\xF0\x9F\x9A\xAF", ur"""\U0001F6AF"""
    ),
    u"POTABLE WATER SYMBOL": (u"\xF0\x9F\x9A\xB0", ur"""\U0001F6B0"""
    ),
    u"NON-POTABLE WATER SYMBOL": (u"\xF0\x9F\x9A\xB1", ur"""\U0001F6B1"""
    ),
    u"NO BICYCLES": (u"\xF0\x9F\x9A\xB3", ur"""\U0001F6B3"""
    ),
    u"BICYCLIST": (u"\xF0\x9F\x9A\xB4", ur"""\U0001F6B4"""
    ),
    u"MOUNTAIN BICYCLIST": (u"\xF0\x9F\x9A\xB5", ur"""\U0001F6B5"""
    ),
    u"NO PEDESTRIANS": (u"\xF0\x9F\x9A\xB7", ur"""\U0001F6B7"""
    ),
    u"CHILDREN CROSSING": (u"\xF0\x9F\x9A\xB8", ur"""\U0001F6B8"""
    ),
    u"SHOWER": (u"\xF0\x9F\x9A\xBF", ur"""\U0001F6BF"""
    ),
    u"BATHTUB": (u"\xF0\x9F\x9B\x81", ur"""\U0001F6C1"""
    ),
    u"PASSPORT CONTROL": (u"\xF0\x9F\x9B\x82", ur"""\U0001F6C2"""
    ),
    u"CUSTOMS": (u"\xF0\x9F\x9B\x83", ur"""\U0001F6C3"""
    ),
    u"BAGGAGE CLAIM": (u"\xF0\x9F\x9B\x84", ur"""\U0001F6C4"""
    ),
    u"LEFT LUGGAGE": (u"\xF0\x9F\x9B\x85", ur"""\U0001F6C5"""
    ),
    u"EARTH GLOBE EUROPE-AFRICA": (u"\xF0\x9F\x8C\x8D", ur"""\U0001F30D"""
    ),
    u"EARTH GLOBE AMERICAS": (u"\xF0\x9F\x8C\x8E", ur"""\U0001F30E"""
    ),
    u"GLOBE WITH MERIDIANS": (u"\xF0\x9F\x8C\x90", ur"""\U0001F310"""
    ),
    u"WAXING CRESCENT MOON SYMBOL": (u"\xF0\x9F\x8C\x92", ur"""\U0001F312"""
    ),
    u"WANING GIBBOUS MOON SYMBOL": (u"\xF0\x9F\x8C\x96", ur"""\U0001F316"""
    ),
    u"LAST QUARTER MOON SYMBOL": (u"\xF0\x9F\x8C\x97", ur"""\U0001F317"""
    ),
    u"WANING CRESCENT MOON SYMBOL": (u"\xF0\x9F\x8C\x98", ur"""\U0001F318"""
    ),
    u"NEW MOON WITH FACE": (u"\xF0\x9F\x8C\x9A", ur"""\U0001F31A"""
    ),
    u"LAST QUARTER MOON WITH FACE": (u"\xF0\x9F\x8C\x9C", ur"""\U0001F31C"""
    ),
    u"FULL MOON WITH FACE": (u"\xF0\x9F\x8C\x9D", ur"""\U0001F31D"""
    ),
    u"SUN WITH FACE": (u"\xF0\x9F\x8C\x9E", ur"""\U0001F31E"""
    ),
    u"EVERGREEN TREE": (u"\xF0\x9F\x8C\xB2", ur"""\U0001F332"""
    ),
    u"DECIDUOUS TREE": (u"\xF0\x9F\x8C\xB3", ur"""\U0001F333"""
    ),
    u"LEMON": (u"\xF0\x9F\x8D\x8B", ur"""\U0001F34B"""
    ),
    u"PEAR": (u"\xF0\x9F\x8D\x90", ur"""\U0001F350"""
    ),
    u"BABY BOTTLE": (u"\xF0\x9F\x8D\xBC", ur"""\U0001F37C"""
    ),
    u"HORSE RACING": (u"\xF0\x9F\x8F\x87", ur"""\U0001F3C7"""
    ),
    u"RUGBY FOOTBALL": (u"\xF0\x9F\x8F\x89", ur"""\U0001F3C9"""
    ),
    u"EUROPEAN POST OFFICE": (u"\xF0\x9F\x8F\xA4", ur"""\U0001F3E4"""
    ),
    u"RAT": (u"\xF0\x9F\x90\x80", ur"""\U0001F400"""
    ),
    u"MOUSE": (u"\xF0\x9F\x90\x81", ur"""\U0001F401"""
    ),
    u"OX": (u"\xF0\x9F\x90\x82", ur"""\U0001F402"""
    ),
    u"WATER BUFFALO": (u"\xF0\x9F\x90\x83", ur"""\U0001F403"""
    ),
    u"COW": (u"\xF0\x9F\x90\x84", ur"""\U0001F404"""
    ),
    u"TIGER": (u"\xF0\x9F\x90\x85", ur"""\U0001F405"""
    ),
    u"LEOPARD": (u"\xF0\x9F\x90\x86", ur"""\U0001F406"""
    ),
    u"RABBIT": (u"\xF0\x9F\x90\x87", ur"""\U0001F407"""
    ),
    u"CAT": (u"\xF0\x9F\x90\x88", ur"""\U0001F408"""
    ),
    u"DRAGON": (u"\xF0\x9F\x90\x89", ur"""\U0001F409"""
    ),
    u"CROCODILE": (u"\xF0\x9F\x90\x8A", ur"""\U0001F40A"""
    ),
    u"WHALE": (u"\xF0\x9F\x90\x8B", ur"""\U0001F40B"""
    ),
    u"RAM": (u"\xF0\x9F\x90\x8F", ur"""\U0001F40F"""
    ),
    u"GOAT": (u"\xF0\x9F\x90\x90", ur"""\U0001F410"""
    ),
    u"ROOSTER": (u"\xF0\x9F\x90\x93", ur"""\U0001F413"""
    ),
    u"DOG": (u"\xF0\x9F\x90\x95", ur"""\U0001F415"""
    ),
    u"PIG": (u"\xF0\x9F\x90\x96", ur"""\U0001F416"""
    ),
    u"DROMEDARY CAMEL": (u"\xF0\x9F\x90\xAA", ur"""\U0001F42A"""
    ),
    u"BUSTS IN SILHOUETTE": (u"\xF0\x9F\x91\xA5", ur"""\U0001F465"""
    ),
    u"TWO MEN HOLDING HANDS": (u"\xF0\x9F\x91\xAC", ur"""\U0001F46C"""
    ),
    u"TWO WOMEN HOLDING HANDS": (u"\xF0\x9F\x91\xAD", ur"""\U0001F46D"""
    ),
    u"THOUGHT BALLOON": (u"\xF0\x9F\x92\xAD", ur"""\U0001F4AD"""
    ),
    u"BANKNOTE WITH EURO SIGN": (u"\xF0\x9F\x92\xB6", ur"""\U0001F4B6"""
    ),
    u"BANKNOTE WITH POUND SIGN": (u"\xF0\x9F\x92\xB7", ur"""\U0001F4B7"""
    ),
    u"OPEN MAILBOX WITH RAISED FLAG": (u"\xF0\x9F\x93\xAC", ur"""\U0001F4EC"""
    ),
    u"OPEN MAILBOX WITH LOWERED FLAG": (u"\xF0\x9F\x93\xAD", ur"""\U0001F4ED"""
    ),
    u"POSTAL HORN": (u"\xF0\x9F\x93\xAF", ur"""\U0001F4EF"""
    ),
    u"NO MOBILE PHONES": (u"\xF0\x9F\x93\xB5", ur"""\U0001F4F5"""
    ),
    u"TWISTED RIGHTWARDS ARROWS": (u"\xF0\x9F\x94\x80", ur"""\U0001F500"""
    ),
    u"CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x81", ur"""\U0001F501"""
    ),
    u"CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS WITH CIRCLED ONE OVERLAY": (u"\xF0\x9F\x94\x82", ur"""\U0001F502"""
    ),
    u"ANTICLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x84", ur"""\U0001F504"""
    ),
    u"LOW BRIGHTNESS SYMBOL": (u"\xF0\x9F\x94\x85", ur"""\U0001F505"""
    ),
    u"HIGH BRIGHTNESS SYMBOL": (u"\xF0\x9F\x94\x86", ur"""\U0001F506"""
    ),
    u"SPEAKER WITH CANCELLATION STROKE": (u"\xF0\x9F\x94\x87", ur"""\U0001F507"""
    ),
    u"SPEAKER WITH ONE SOUND WAVE": (u"\xF0\x9F\x94\x89", ur"""\U0001F509"""
    ),
    u"BELL WITH CANCELLATION STROKE": (u"\xF0\x9F\x94\x95", ur"""\U0001F515"""
    ),
    u"MICROSCOPE": (u"\xF0\x9F\x94\xAC", ur"""\U0001F52C"""
    ),
    u"TELESCOPE": (u"\xF0\x9F\x94\xAD", ur"""\U0001F52D"""
    ),
    u"CLOCK FACE ONE-THIRTY": (u"\xF0\x9F\x95\x9C", ur"""\U0001F55C"""
    ),
    u"CLOCK FACE TWO-THIRTY": (u"\xF0\x9F\x95\x9D", ur"""\U0001F55D"""
    ),
    u"CLOCK FACE THREE-THIRTY": (u"\xF0\x9F\x95\x9E", ur"""\U0001F55E"""
    ),
    u"CLOCK FACE FOUR-THIRTY": (u"\xF0\x9F\x95\x9F", ur"""\U0001F55F"""
    ),
    u"CLOCK FACE FIVE-THIRTY": (u"\xF0\x9F\x95\xA0", ur"""\U0001F560"""
    ),
    u"CLOCK FACE SIX-THIRTY": (u"\xF0\x9F\x95\xA1", ur"""\U0001F561"""
    ),
    u"CLOCK FACE SEVEN-THIRTY": (u"\xF0\x9F\x95\xA2", ur"""\U0001F562"""
    ),
    u"CLOCK FACE EIGHT-THIRTY": (u"\xF0\x9F\x95\xA3", ur"""\U0001F563"""
    ),
    u"CLOCK FACE NINE-THIRTY": (u"\xF0\x9F\x95\xA4", ur"""\U0001F564"""
    ),
    u"CLOCK FACE TEN-THIRTY": (u"\xF0\x9F\x95\xA5", ur"""\U0001F565"""
    ),
    u"CLOCK FACE ELEVEN-THIRTY": (u"\xF0\x9F\x95\xA6", ur"""\U0001F566"""
    ),
    u"CLOCK FACE TWELVE-THIRTY": (u"\xF0\x9F\x95\xA7", ur"""\U0001F567"""
    ),

}

REGEX_FEATURE_CONFIG_SMALL = {
    # ---- Emoticons ----
    u"Emoticon Happy": (u":-)", 
                        r"""[:=][o-]?[)}>\]]|               # :-) :o) :)
                        [({<\[][o-]?[:=]|                   # (-: (o: (:
                        \^(_*|[-oO]?)\^                     # ^^ ^-^
                        """
    ), 
    u"Emoticon Laughing": (u":-D", r"""([:=][-]?|x)[D]"""),   # :-D xD
    u"Emoticon Winking": (u";-)", 
                        r"""[;\*][-o]?[)}>\]]|              # ;-) ;o) ;)
                        [({<\[][-o]?[;\*]                   # (-; (
                        """
    ), 
    u"Emotion Tongue": (u":-P", 
                        r"""[:=][-]?[pqP](?!\w)|            # :-P :P
                        (?<!\w)[pqP][-]?[:=]                # q-: P-:
                        """
    ),  
    u"Emoticon Surprise": (u":-O", 
                            r"""(?<!\w|\.)                  # Boundary
                            ([:=]-?[oO0]|                   # :-O
                            [oO0]-?[:=]|                    # O-:
                            [oO](_*|\.)[oO])                # Oo O____o O.o
                            (?!\w)
                            """
    ), 
    u"Emoticon Dissatisfied": (u":-/", 
                                r"""(?<!\w)                 # Boundary
                                [:=][-o]?[\/\\|]|           # :-/ :-\ :-| :/
                                [\/\\|][-o]?[:=]|           # \-: \:
                                -_+-                        # -_- -___-
                                """
    ), 
    u"Emoticon Sad": (u":-(", 
                        r"""[:=][o-]?[({<\[]|               # :-( :(
                        (?<!(\w|%))                         # Boundary
                        [)}>\[][o-]?[:=]                    # )-: ): )o: 
                        """
    ), 
    u"Emoticon Crying": (u";-(", 
                        r"""(([:=]')|(;'?))[o-]?[({<\[]|    # ;-( :'(
                        (?<!(\w|%))                         # Boundary
                        [)}>\[][o-]?(('[:=])|('?;))         # )-; )-';
                        """
    ), 
      
    # ---- Punctuation----
    # u"AllPunctuation": (u"", r"""((\.{2,}|[?!]{2,})1*)"""),
    u"Question Mark": (u"??", r"""\?{2,}"""),                 # ??
    u"Exclamation Mark": (u"!!", r"""\!{2,}"""),              # !!
    u"Question and Exclamation Mark": (u"?!", r"""[\!\?]*((\?\!)+|              # ?!
                    (\!\?)+)[\!\?]*                         # !?
                    """
    ),                                          # Unicode interrobang: U+203D
    u"Ellipsis": (u"...", r"""\.{2,}|                         # .. ...
                \.(\ \.){2,}                                # . . .
                """
    ),                                          # Unicode Ellipsis: U+2026
    # ---- Markup----
#     u"Hashtag": (u"#", r"""\#(irony|ironic|sarcasm|sarcastic)"""),  # #irony
#     u"Pseudo-Tag": (u"Tag", 
#                     r"""([<\[][\/\\]
#                     (irony|ironic|sarcasm|sarcastic)        # </irony>
#                     [>\]])|                                 #
#                     ((?<!(\w|[<\[]))[\/\\]                  #
#                     (irony|ironic|sarcasm|sarcastic)        # /irony
#                     (?![>\]]))
#                     """
#     ),
  
    # ---- Acronyms, onomatopoeia ----
    u"Acroym for Laughter": (u"lol", 
                    r"""(?<!\w)                             # Boundary
                    (l(([oua]|aw)l)+([sz]?|wut)|            # lol, lawl, luls
                    rot?fl(mf?ao)?)|                        # rofl, roflmao
                    lmf?ao                                  # lmao, lmfao
                    (?!\w)                                  # Boundary
                    """
    ),                                    
    u"Acronym for Grin": (u"*g*", 
                        r"""\*([Gg]{1,2}|                   # *g* *gg*
                        grin)\*                             # *grin*
                        """
    ),
    u"Onomatopoeia for Laughter": (u"haha", 
                        r"""(?<!\w)                         # Boundary
                        (mu|ba)?                            # mu- ba-
                        (ha|h(e|3)|hi){2,}                  # haha, hehe, hihi
                        (?!\w)                              # Boundary
                        """
    ),
    u"Interjection": (u"ITJ", 
                        r"""(?<!\w)((a+h+a?)|               # ah, aha
                        (e+h+)|                             # eh
                        (u+g?h+)|                           # ugh
                        (huh)|                              # huh
                        ([uo]h( |-)h?[uo]h)|                # uh huh, 
                        (m*hm+)                             # hmm, mhm
                        |(h(u|r)?mp(h|f))|                  # hmpf
                        (ar+gh+)|                           # argh
                        (wow+))(?!\w)                       # wow
                        """
    ),
    u"GRINNING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x81", ur"""\U0001F601"""
    ),
    u"FACE WITH TEARS OF JOY": (u"\xF0\x9F\x98\x82", ur"""\U0001F602"""
    ),
    u"SMILING FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\x83", ur"""\U0001F603"""
    ),
    u"SMILING FACE WITH OPEN MOUTH AND SMILING EYES": (u"\xF0\x9F\x98\x84", ur"""\U0001F604"""
    ),
    u"SMILING FACE WITH OPEN MOUTH AND COLD SWEAT": (u"\xF0\x9F\x98\x85", ur"""\U0001F605"""
    ),
    u"SMILING FACE WITH OPEN MOUTH AND TIGHTLY-CLOSED EYES": (u"\xF0\x9F\x98\x86", ur"""\U0001F606"""
    ),
    u"WINKING FACE": (u"\xF0\x9F\x98\x89", ur"""\U0001F609"""
    ),
    u"SMILING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x8A", ur"""\U0001F60A"""
    ),
    u"FACE SAVOURING DELICIOUS FOOD": (u"\xF0\x9F\x98\x8B", ur"""\U0001F60B"""
    ),
    u"RELIEVED FACE": (u"\xF0\x9F\x98\x8C", ur"""\U0001F60C"""
    ),
    u"SMILING FACE WITH HEART-SHAPED EYES": (u"\xF0\x9F\x98\x8D", ur"""\U0001F60D"""
    ),
    u"SMIRKING FACE": (u"\xF0\x9F\x98\x8F", ur"""\U0001F60F"""
    ),
    u"UNAMUSED FACE": (u"\xF0\x9F\x98\x92", ur"""\U0001F612"""
    ),
    u"FACE WITH COLD SWEAT": (u"\xF0\x9F\x98\x93", ur"""\U0001F613"""
    ),
    u"PENSIVE FACE": (u"\xF0\x9F\x98\x94", ur"""\U0001F614"""
    ),
    u"CONFOUNDED FACE": (u"\xF0\x9F\x98\x96", ur"""\U0001F616"""
    ),
    u"FACE THROWING A KISS": (u"\xF0\x9F\x98\x98", ur"""\U0001F618"""
    ),
    u"KISSING FACE WITH CLOSED EYES": (u"\xF0\x9F\x98\x9A", ur"""\U0001F61A"""
    ),
    u"FACE WITH STUCK-OUT TONGUE AND WINKING EYE": (u"\xF0\x9F\x98\x9C", ur"""\U0001F61C"""
    ),
    u"FACE WITH STUCK-OUT TONGUE AND TIGHTLY-CLOSED EYES": (u"\xF0\x9F\x98\x9D", ur"""\U0001F61D"""
    ),
    u"DISAPPOINTED FACE": (u"\xF0\x9F\x98\x9E", ur"""\U0001F61E"""
    ),
    u"ANGRY FACE": (u"\xF0\x9F\x98\xA0", ur"""\U0001F620"""
    ),
    u"POUTING FACE": (u"\xF0\x9F\x98\xA1", ur"""\U0001F621"""
    ),
    u"CRYING FACE": (u"\xF0\x9F\x98\xA2", ur"""\U0001F622"""
    ),
    u"PERSEVERING FACE": (u"\xF0\x9F\x98\xA3", ur"""\U0001F623"""
    ),
    u"FACE WITH LOOK OF TRIUMPH": (u"\xF0\x9F\x98\xA4", ur"""\U0001F624"""
    ),
    u"DISAPPOINTED BUT RELIEVED FACE": (u"\xF0\x9F\x98\xA5", ur"""\U0001F625"""
    ),
    u"FEARFUL FACE": (u"\xF0\x9F\x98\xA8", ur"""\U0001F628"""
    ),
    u"WEARY FACE": (u"\xF0\x9F\x98\xA9", ur"""\U0001F629"""
    ),
    u"SLEEPY FACE": (u"\xF0\x9F\x98\xAA", ur"""\U0001F62A"""
    ),
    u"TIRED FACE": (u"\xF0\x9F\x98\xAB", ur"""\U0001F62B"""
    ),
    u"LOUDLY CRYING FACE": (u"\xF0\x9F\x98\xAD", ur"""\U0001F62D"""
    ),
    u"FACE WITH OPEN MOUTH AND COLD SWEAT": (u"\xF0\x9F\x98\xB0", ur"""\U0001F630"""
    ),
    u"FACE SCREAMING IN FEAR": (u"\xF0\x9F\x98\xB1", ur"""\U0001F631"""
    ),
    u"ASTONISHED FACE": (u"\xF0\x9F\x98\xB2", ur"""\U0001F632"""
    ),
    u"FLUSHED FACE": (u"\xF0\x9F\x98\xB3", ur"""\U0001F633"""
    ),
    u"DIZZY FACE": (u"\xF0\x9F\x98\xB5", ur"""\U0001F635"""
    ),
#     u"FACE WITH MEDICAL MASK": (u"\xF0\x9F\x98\xB7", ur"""\U0001F637"""
#     ),
#     u"GRINNING CAT FACE WITH SMILING EYES": (u"\xF0\x9F\x98\xB8", ur"""\U0001F638"""
#     ),
#     u"CAT FACE WITH TEARS OF JOY": (u"\xF0\x9F\x98\xB9", ur"""\U0001F639"""
#     ),
#     u"SMILING CAT FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xBA", ur"""\U0001F63A"""
#     ),
#     u"SMILING CAT FACE WITH HEART-SHAPED EYES": (u"\xF0\x9F\x98\xBB", ur"""\U0001F63B"""
#     ),
#     u"CAT FACE WITH WRY SMILE": (u"\xF0\x9F\x98\xBC", ur"""\U0001F63C"""
#     ),
#     u"KISSING CAT FACE WITH CLOSED EYES": (u"\xF0\x9F\x98\xBD", ur"""\U0001F63D"""
#     ),
#     u"POUTING CAT FACE": (u"\xF0\x9F\x98\xBE", ur"""\U0001F63E"""
#     ),
#     u"CRYING CAT FACE": (u"\xF0\x9F\x98\xBF", ur"""\U0001F63F"""
#     ),
#     u"WEARY CAT FACE": (u"\xF0\x9F\x99\x80", ur"""\U0001F640"""
#     ),
#     u"FACE WITH NO GOOD GESTURE": (u"\xF0\x9F\x99\x85", ur"""\U0001F645"""
#     ),
#     u"FACE WITH OK GESTURE": (u"\xF0\x9F\x99\x86", ur"""\U0001F646"""
#     ),
#     u"PERSON BOWING DEEPLY": (u"\xF0\x9F\x99\x87", ur"""\U0001F647"""
#     ),
#     u"SEE-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x88", ur"""\U0001F648"""
#     ),
#     u"HEAR-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x89", ur"""\U0001F649"""
#     ),
#     u"SPEAK-NO-EVIL MONKEY": (u"\xF0\x9F\x99\x8A", ur"""\U0001F64A"""
#     ),
#     u"HAPPY PERSON RAISING ONE HAND": (u"\xF0\x9F\x99\x8B", ur"""\U0001F64B"""
#     ),
#     u"PERSON RAISING BOTH HANDS IN CELEBRATION": (u"\xF0\x9F\x99\x8C", ur"""\U0001F64C"""
#     ),
#     u"PERSON FROWNING": (u"\xF0\x9F\x99\x8D", ur"""\U0001F64D"""
#     ),
#     u"PERSON WITH POUTING FACE": (u"\xF0\x9F\x99\x8E", ur"""\U0001F64E"""
#     ),
#     u"PERSON WITH FOLDED HANDS": (u"\xF0\x9F\x99\x8F", ur"""\U0001F64F"""
#     ),
#     u"BLACK SCISSORS": (u"\xE2\x9C\x82", ur"""\U00002702"""
#     ),
#     u"WHITE HEAVY CHECK MARK": (u"\xE2\x9C\x85", ur"""\U00002705"""
#     ),
#     u"AIRPLANE": (u"\xE2\x9C\x88", ur"""\U00002708"""
#     ),
#     u"ENVELOPE": (u"\xE2\x9C\x89", ur"""\U00002709"""
#     ),
#     u"RAISED FIST": (u"\xE2\x9C\x8A", ur"""\U0000270A"""
#     ),
#     u"RAISED HAND": (u"\xE2\x9C\x8B", ur"""\U0000270B"""
#     ),
#     u"VICTORY HAND": (u"\xE2\x9C\x8C", ur"""\U0000270C"""
#     ),
#     u"PENCIL": (u"\xE2\x9C\x8F", ur"""\U0000270F"""
#     ),
#     u"BLACK NIB": (u"\xE2\x9C\x92", ur"""\U00002712"""
#     ),
#     u"HEAVY CHECK MARK": (u"\xE2\x9C\x94", ur"""\U00002714"""
#     ),
#     u"HEAVY MULTIPLICATION X": (u"\xE2\x9C\x96", ur"""\U00002716"""
#     ),
#     u"SPARKLES": (u"\xE2\x9C\xA8", ur"""\U00002728"""
#     ),
#     u"EIGHT SPOKED ASTERISK": (u"\xE2\x9C\xB3", ur"""\U00002733"""
#     ),
#     u"EIGHT POINTED BLACK STAR": (u"\xE2\x9C\xB4", ur"""\U00002734"""
#     ),
#     u"SNOWFLAKE": (u"\xE2\x9D\x84", ur"""\U00002744"""
#     ),
#     u"SPARKLE": (u"\xE2\x9D\x87", ur"""\U00002747"""
#     ),
#     u"CROSS MARK": (u"\xE2\x9D\x8C", ur"""\U0000274C"""
#     ),
#     u"NEGATIVE SQUARED CROSS MARK": (u"\xE2\x9D\x8E", ur"""\U0000274E"""
#     ),
#     u"BLACK QUESTION MARK ORNAMENT": (u"\xE2\x9D\x93", ur"""\U00002753"""
#     ),
#     u"WHITE QUESTION MARK ORNAMENT": (u"\xE2\x9D\x94", ur"""\U00002754"""
#     ),
#     u"WHITE EXCLAMATION MARK ORNAMENT": (u"\xE2\x9D\x95", ur"""\U00002755"""
#     ),
#     u"HEAVY EXCLAMATION MARK SYMBOL": (u"\xE2\x9D\x97", ur"""\U00002757"""
#     ),
#     u"HEAVY BLACK HEART": (u"\xE2\x9D\xA4", ur"""\U00002764"""
#     ),
#     u"HEAVY PLUS SIGN": (u"\xE2\x9E\x95", ur"""\U00002795"""
#     ),
#     u"HEAVY MINUS SIGN": (u"\xE2\x9E\x96", ur"""\U00002796"""
#     ),
#     u"HEAVY DIVISION SIGN": (u"\xE2\x9E\x97", ur"""\U00002797"""
#     ),
#     u"BLACK RIGHTWARDS ARROW": (u"\xE2\x9E\xA1", ur"""\U000027A1"""
#     ),
#     u"CURLY LOOP": (u"\xE2\x9E\xB0", ur"""\U000027B0"""
#     ),
#     u"ROCKET": (u"\xF0\x9F\x9A\x80", ur"""\U0001F680"""
#     ),
#     u"RAILWAY CAR": (u"\xF0\x9F\x9A\x83", ur"""\U0001F683"""
#     ),
#     u"HIGH-SPEED TRAIN": (u"\xF0\x9F\x9A\x84", ur"""\U0001F684"""
#     ),
#     u"HIGH-SPEED TRAIN WITH BULLET NOSE": (u"\xF0\x9F\x9A\x85", ur"""\U0001F685"""
#     ),
#     u"METRO": (u"\xF0\x9F\x9A\x87", ur"""\U0001F687"""
#     ),
#     u"STATION": (u"\xF0\x9F\x9A\x89", ur"""\U0001F689"""
#     ),
#     u"BUS": (u"\xF0\x9F\x9A\x8C", ur"""\U0001F68C"""
#     ),
#     u"BUS STOP": (u"\xF0\x9F\x9A\x8F", ur"""\U0001F68F"""
#     ),
#     u"AMBULANCE": (u"\xF0\x9F\x9A\x91", ur"""\U0001F691"""
#     ),
#     u"FIRE ENGINE": (u"\xF0\x9F\x9A\x92", ur"""\U0001F692"""
#     ),
#     u"POLICE CAR": (u"\xF0\x9F\x9A\x93", ur"""\U0001F693"""
#     ),
#     u"TAXI": (u"\xF0\x9F\x9A\x95", ur"""\U0001F695"""
#     ),
#     u"AUTOMOBILE": (u"\xF0\x9F\x9A\x97", ur"""\U0001F697"""
#     ),
#     u"RECREATIONAL VEHICLE": (u"\xF0\x9F\x9A\x99", ur"""\U0001F699"""
#     ),
#     u"DELIVERY TRUCK": (u"\xF0\x9F\x9A\x9A", ur"""\U0001F69A"""
#     ),
#     u"SHIP": (u"\xF0\x9F\x9A\xA2", ur"""\U0001F6A2"""
#     ),
#     u"SPEEDBOAT": (u"\xF0\x9F\x9A\xA4", ur"""\U0001F6A4"""
#     ),
#     u"HORIZONTAL TRAFFIC LIGHT": (u"\xF0\x9F\x9A\xA5", ur"""\U0001F6A5"""
#     ),
#     u"CONSTRUCTION SIGN": (u"\xF0\x9F\x9A\xA7", ur"""\U0001F6A7"""
#     ),
#     u"POLICE CARS REVOLVING LIGHT": (u"\xF0\x9F\x9A\xA8", ur"""\U0001F6A8"""
#     ),
#     u"TRIANGULAR FLAG ON POST": (u"\xF0\x9F\x9A\xA9", ur"""\U0001F6A9"""
#     ),
#     u"DOOR": (u"\xF0\x9F\x9A\xAA", ur"""\U0001F6AA"""
#     ),
#     u"NO ENTRY SIGN": (u"\xF0\x9F\x9A\xAB", ur"""\U0001F6AB"""
#     ),
#     u"SMOKING SYMBOL": (u"\xF0\x9F\x9A\xAC", ur"""\U0001F6AC"""
#     ),
#     u"NO SMOKING SYMBOL": (u"\xF0\x9F\x9A\xAD", ur"""\U0001F6AD"""
#     ),
#     u"BICYCLE": (u"\xF0\x9F\x9A\xB2", ur"""\U0001F6B2"""
#     ),
#     u"PEDESTRIAN": (u"\xF0\x9F\x9A\xB6", ur"""\U0001F6B6"""
#     ),
#     u"MENS SYMBOL": (u"\xF0\x9F\x9A\xB9", ur"""\U0001F6B9"""
#     ),
#     u"WOMENS SYMBOL": (u"\xF0\x9F\x9A\xBA", ur"""\U0001F6BA"""
#     ),
#     u"RESTROOM": (u"\xF0\x9F\x9A\xBB", ur"""\U0001F6BB"""
#     ),
#     u"BABY SYMBOL": (u"\xF0\x9F\x9A\xBC", ur"""\U0001F6BC"""
#     ),
#     u"TOILET": (u"\xF0\x9F\x9A\xBD", ur"""\U0001F6BD"""
#     ),
#     u"WATER CLOSET": (u"\xF0\x9F\x9A\xBE", ur"""\U0001F6BE"""
#     ),
#     u"BATH": (u"\xF0\x9F\x9B\x80", ur"""\U0001F6C0"""
#     ),
#     u"CIRCLED LATIN CAPITAL LETTER M": (u"\xE2\x93\x82", ur"""\U000024C2"""
#     ),
    u"NEGATIVE SQUARED LATIN CAPITAL LETTER A": (u"\xF0\x9F\x85\xB0", ur"""\U0001F170"""
    ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER B": (u"\xF0\x9F\x85\xB1", ur"""\U0001F171"""
#     ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER O": (u"\xF0\x9F\x85\xBE", ur"""\U0001F17E"""
#     ),
#     u"NEGATIVE SQUARED LATIN CAPITAL LETTER P": (u"\xF0\x9F\x85\xBF", ur"""\U0001F17F"""
#     ),
#     u"NEGATIVE SQUARED AB": (u"\xF0\x9F\x86\x8E", ur"""\U0001F18E"""
#     ),
#     u"SQUARED CL": (u"\xF0\x9F\x86\x91", ur"""\U0001F191"""
#     ),
#     u"SQUARED COOL": (u"\xF0\x9F\x86\x92", ur"""\U0001F192"""
#     ),
#     u"SQUARED FREE": (u"\xF0\x9F\x86\x93", ur"""\U0001F193"""
#     ),
#     u"SQUARED ID": (u"\xF0\x9F\x86\x94", ur"""\U0001F194"""
#     ),
#     u"SQUARED NEW": (u"\xF0\x9F\x86\x95", ur"""\U0001F195"""
#     ),
#     u"SQUARED NG": (u"\xF0\x9F\x86\x96", ur"""\U0001F196"""
#     ),
#     u"SQUARED OK": (u"\xF0\x9F\x86\x97", ur"""\U0001F197"""
#     ),
#     u"SQUARED SOS": (u"\xF0\x9F\x86\x98", ur"""\U0001F198"""
#     ),
#     u"SQUARED UP WITH EXCLAMATION MARK": (u"\xF0\x9F\x86\x99", ur"""\U0001F199"""
#     ),
#     u"SQUARED VS": (u"\xF0\x9F\x86\x9A", ur"""\U0001F19A"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER D + REGIONAL INDICATOR SYMBOL LETTER E": (u"\xF0\x9F\x87\xA9\xF0\x9F\x87\xAA", ur"""\U0001F1E9 \U0001F1EA"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER G + REGIONAL INDICATOR SYMBOL LETTER B": (u"\xF0\x9F\x87\xAC\xF0\x9F\x87\xA7", ur"""\U0001F1EC \U0001F1E7"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER C + REGIONAL INDICATOR SYMBOL LETTER N": (u"\xF0\x9F\x87\xA8\xF0\x9F\x87\xB3", ur"""\U0001F1E8 \U0001F1F3"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER J + REGIONAL INDICATOR SYMBOL LETTER P": (u"\xF0\x9F\x87\xAF\xF0\x9F\x87\xB5", ur"""\U0001F1EF \U0001F1F5"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER K + REGIONAL INDICATOR SYMBOL LETTER R": (u"\xF0\x9F\x87\xB0\xF0\x9F\x87\xB7", ur"""\U0001F1F0 \U0001F1F7"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER F + REGIONAL INDICATOR SYMBOL LETTER R": (u"\xF0\x9F\x87\xAB\xF0\x9F\x87\xB7", ur"""\U0001F1EB \U0001F1F7"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER E + REGIONAL INDICATOR SYMBOL LETTER S": (u"\xF0\x9F\x87\xAA\xF0\x9F\x87\xB8", ur"""\U0001F1EA \U0001F1F8"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER I + REGIONAL INDICATOR SYMBOL LETTER T": (u"\xF0\x9F\x87\xAE\xF0\x9F\x87\xB9", ur"""\U0001F1EE \U0001F1F9"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER U + REGIONAL INDICATOR SYMBOL LETTER S": (u"\xF0\x9F\x87\xBA\xF0\x9F\x87\xB8", ur"""\U0001F1FA \U0001F1F8"""
#     ),
#     u"REGIONAL INDICATOR SYMBOL LETTER R + REGIONAL INDICATOR SYMBOL LETTER U": (u"\xF0\x9F\x87\xB7\xF0\x9F\x87\xBA", ur"""\U0001F1F7 \U0001F1FA"""
#     ),
#     u"SQUARED KATAKANA KOKO": (u"\xF0\x9F\x88\x81", ur"""\U0001F201"""
#     ),
#     u"SQUARED KATAKANA SA": (u"\xF0\x9F\x88\x82", ur"""\U0001F202"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7121": (u"\xF0\x9F\x88\x9A", ur"""\U0001F21A"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6307": (u"\xF0\x9F\x88\xAF", ur"""\U0001F22F"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7981": (u"\xF0\x9F\x88\xB2", ur"""\U0001F232"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7A7A": (u"\xF0\x9F\x88\xB3", ur"""\U0001F233"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-5408": (u"\xF0\x9F\x88\xB4", ur"""\U0001F234"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6E80": (u"\xF0\x9F\x88\xB5", ur"""\U0001F235"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6709": (u"\xF0\x9F\x88\xB6", ur"""\U0001F236"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-6708": (u"\xF0\x9F\x88\xB7", ur"""\U0001F237"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-7533": (u"\xF0\x9F\x88\xB8", ur"""\U0001F238"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-5272": (u"\xF0\x9F\x88\xB9", ur"""\U0001F239"""
#     ),
#     u"SQUARED CJK UNIFIED IDEOGRAPH-55B6": (u"\xF0\x9F\x88\xBA", ur"""\U0001F23A"""
#     ),
#     u"CIRCLED IDEOGRAPH ADVANTAGE": (u"\xF0\x9F\x89\x90", ur"""\U0001F250"""
#     ),
#     u"CIRCLED IDEOGRAPH ACCEPT": (u"\xF0\x9F\x89\x91", ur"""\U0001F251"""
#     ),
#     u"COPYRIGHT SIGN": (u"\xC2\xA9", ur"""\U000000A9"""
#     ),
#     u"REGISTERED SIGN": (u"\xC2\xAE", ur"""\U000000AE"""
#     ),
#     u"DOUBLE EXCLAMATION MARK": (u"\xE2\x80\xBC", ur"""\U0000203C"""
#     ),
#     u"EXCLAMATION QUESTION MARK": (u"\xE2\x81\x89", ur"""\U00002049"""
#     ),
#     u"DIGIT EIGHT + COMBINING ENCLOSING KEYCAP": (u"\x38\xE2\x83\xA3", ur"""\U00000038 \U000020E3"""
#     ),
#     u"DIGIT NINE + COMBINING ENCLOSING KEYCAP": (u"\x39\xE2\x83\xA3", ur"""\U00000039 \U000020E3"""
#     ),
#     u"DIGIT SEVEN + COMBINING ENCLOSING KEYCAP": (u"\x37\xE2\x83\xA3", ur"""\U00000037 \U000020E3"""
#     ),
#     u"DIGIT SIX + COMBINING ENCLOSING KEYCAP": (u"\x36\xE2\x83\xA3", ur"""\U00000036 \U000020E3"""
#     ),
#     u"DIGIT ONE + COMBINING ENCLOSING KEYCAP": (u"\x31\xE2\x83\xA3", ur"""\U00000031 \U000020E3"""
#     ),
#     u"DIGIT ZERO + COMBINING ENCLOSING KEYCAP": (u"\x30\xE2\x83\xA3", ur"""\U00000030 \U000020E3"""
#     ),
#     u"DIGIT TWO + COMBINING ENCLOSING KEYCAP": (u"\x32\xE2\x83\xA3", ur"""\U00000032 \U000020E3"""
#     ),
#     u"DIGIT THREE + COMBINING ENCLOSING KEYCAP": (u"\x33\xE2\x83\xA3", ur"""\U00000033 \U000020E3"""
#     ),
#     u"DIGIT FIVE + COMBINING ENCLOSING KEYCAP": (u"\x35\xE2\x83\xA3", ur"""\U00000035 \U000020E3"""
#     ),
#     u"DIGIT FOUR + COMBINING ENCLOSING KEYCAP": (u"\x34\xE2\x83\xA3", ur"""\U00000034 \U000020E3"""
#     ),
#     u"NUMBER SIGN + COMBINING ENCLOSING KEYCAP": (u"\x23\xE2\x83\xA3", ur"""\U00000023 \U000020E3"""
#     ),
#     u"TRADE MARK SIGN": (u"\xE2\x84\xA2", ur"""\U00002122"""
#     ),
#     u"INFORMATION SOURCE": (u"\xE2\x84\xB9", ur"""\U00002139"""
#     ),
#     u"LEFT RIGHT ARROW": (u"\xE2\x86\x94", ur"""\U00002194"""
#     ),
#     u"UP DOWN ARROW": (u"\xE2\x86\x95", ur"""\U00002195"""
#     ),
#     u"NORTH WEST ARROW": (u"\xE2\x86\x96", ur"""\U00002196"""
#     ),
#     u"NORTH EAST ARROW": (u"\xE2\x86\x97", ur"""\U00002197"""
#     ),
#     u"SOUTH EAST ARROW": (u"\xE2\x86\x98", ur"""\U00002198"""
#     ),
#     u"SOUTH WEST ARROW": (u"\xE2\x86\x99", ur"""\U00002199"""
#     ),
#     u"LEFTWARDS ARROW WITH HOOK": (u"\xE2\x86\xA9", ur"""\U000021A9"""
#     ),
#     u"RIGHTWARDS ARROW WITH HOOK": (u"\xE2\x86\xAA", ur"""\U000021AA"""
#     ),
#     u"WATCH": (u"\xE2\x8C\x9A", ur"""\U0000231A"""
#     ),
#     u"HOURGLASS": (u"\xE2\x8C\x9B", ur"""\U0000231B"""
#     ),
#     u"BLACK RIGHT-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xA9", ur"""\U000023E9"""
#     ),
#     u"BLACK LEFT-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAA", ur"""\U000023EA"""
#     ),
#     u"BLACK UP-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAB", ur"""\U000023EB"""
#     ),
#     u"BLACK DOWN-POINTING DOUBLE TRIANGLE": (u"\xE2\x8F\xAC", ur"""\U000023EC"""
#     ),
#     u"ALARM CLOCK": (u"\xE2\x8F\xB0", ur"""\U000023F0"""
#     ),
#     u"HOURGLASS WITH FLOWING SAND": (u"\xE2\x8F\xB3", ur"""\U000023F3"""
#     ),
#     u"BLACK SMALL SQUARE": (u"\xE2\x96\xAA", ur"""\U000025AA"""
#     ),
#     u"WHITE SMALL SQUARE": (u"\xE2\x96\xAB", ur"""\U000025AB"""
#     ),
#     u"BLACK RIGHT-POINTING TRIANGLE": (u"\xE2\x96\xB6", ur"""\U000025B6"""
#     ),
#     u"BLACK LEFT-POINTING TRIANGLE": (u"\xE2\x97\x80", ur"""\U000025C0"""
#     ),
#     u"WHITE MEDIUM SQUARE": (u"\xE2\x97\xBB", ur"""\U000025FB"""
#     ),
#     u"BLACK MEDIUM SQUARE": (u"\xE2\x97\xBC", ur"""\U000025FC"""
#     ),
#     u"WHITE MEDIUM SMALL SQUARE": (u"\xE2\x97\xBD", ur"""\U000025FD"""
#     ),
#     u"BLACK MEDIUM SMALL SQUARE": (u"\xE2\x97\xBE", ur"""\U000025FE"""
#     ),
#     u"BLACK SUN WITH RAYS": (u"\xE2\x98\x80", ur"""\U00002600"""
#     ),
#     u"CLOUD": (u"\xE2\x98\x81", ur"""\U00002601"""
#     ),
#     u"BLACK TELEPHONE": (u"\xE2\x98\x8E", ur"""\U0000260E"""
#     ),
#     u"BALLOT BOX WITH CHECK": (u"\xE2\x98\x91", ur"""\U00002611"""
#     ),
#     u"UMBRELLA WITH RAIN DROPS": (u"\xE2\x98\x94", ur"""\U00002614"""
#     ),
#     u"HOT BEVERAGE": (u"\xE2\x98\x95", ur"""\U00002615"""
#     ),
#     u"WHITE UP POINTING INDEX": (u"\xE2\x98\x9D", ur"""\U0000261D"""
#     ),
#     u"WHITE SMILING FACE": (u"\xE2\x98\xBA", ur"""\U0000263A"""
#     ),
#     u"ARIES": (u"\xE2\x99\x88", ur"""\U00002648"""
#     ),
#     u"TAURUS": (u"\xE2\x99\x89", ur"""\U00002649"""
#     ),
#     u"GEMINI": (u"\xE2\x99\x8A", ur"""\U0000264A"""
#     ),
#     u"CANCER": (u"\xE2\x99\x8B", ur"""\U0000264B"""
#     ),
#     u"LEO": (u"\xE2\x99\x8C", ur"""\U0000264C"""
#     ),
#     u"VIRGO": (u"\xE2\x99\x8D", ur"""\U0000264D"""
#     ),
#     u"LIBRA": (u"\xE2\x99\x8E", ur"""\U0000264E"""
#     ),
#     u"SCORPIUS": (u"\xE2\x99\x8F", ur"""\U0000264F"""
#     ),
#     u"SAGITTARIUS": (u"\xE2\x99\x90", ur"""\U00002650"""
#     ),
#     u"CAPRICORN": (u"\xE2\x99\x91", ur"""\U00002651"""
#     ),
#     u"AQUARIUS": (u"\xE2\x99\x92", ur"""\U00002652"""
#     ),
#     u"PISCES": (u"\xE2\x99\x93", ur"""\U00002653"""
#     ),
#     u"BLACK SPADE SUIT": (u"\xE2\x99\xA0", ur"""\U00002660"""
#     ),
#     u"BLACK CLUB SUIT": (u"\xE2\x99\xA3", ur"""\U00002663"""
#     ),
#     u"BLACK HEART SUIT": (u"\xE2\x99\xA5", ur"""\U00002665"""
#     ),
#     u"BLACK DIAMOND SUIT": (u"\xE2\x99\xA6", ur"""\U00002666"""
#     ),
#     u"HOT SPRINGS": (u"\xE2\x99\xA8", ur"""\U00002668"""
#     ),
#     u"BLACK UNIVERSAL RECYCLING SYMBOL": (u"\xE2\x99\xBB", ur"""\U0000267B"""
#     ),
#     u"WHEELCHAIR SYMBOL": (u"\xE2\x99\xBF", ur"""\U0000267F"""
#     ),
#     u"ANCHOR": (u"\xE2\x9A\x93", ur"""\U00002693"""
#     ),
#     u"WARNING SIGN": (u"\xE2\x9A\xA0", ur"""\U000026A0"""
#     ),
#     u"HIGH VOLTAGE SIGN": (u"\xE2\x9A\xA1", ur"""\U000026A1"""
#     ),
#     u"MEDIUM WHITE CIRCLE": (u"\xE2\x9A\xAA", ur"""\U000026AA"""
#     ),
#     u"MEDIUM BLACK CIRCLE": (u"\xE2\x9A\xAB", ur"""\U000026AB"""
#     ),
#     u"SOCCER BALL": (u"\xE2\x9A\xBD", ur"""\U000026BD"""
#     ),
#     u"BASEBALL": (u"\xE2\x9A\xBE", ur"""\U000026BE"""
#     ),
#     u"SNOWMAN WITHOUT SNOW": (u"\xE2\x9B\x84", ur"""\U000026C4"""
#     ),
#     u"SUN BEHIND CLOUD": (u"\xE2\x9B\x85", ur"""\U000026C5"""
#     ),
#     u"OPHIUCHUS": (u"\xE2\x9B\x8E", ur"""\U000026CE"""
#     ),
#     u"NO ENTRY": (u"\xE2\x9B\x94", ur"""\U000026D4"""
#     ),
#     u"CHURCH": (u"\xE2\x9B\xAA", ur"""\U000026EA"""
#     ),
#     u"FOUNTAIN": (u"\xE2\x9B\xB2", ur"""\U000026F2"""
#     ),
#     u"FLAG IN HOLE": (u"\xE2\x9B\xB3", ur"""\U000026F3"""
#     ),
#     u"SAILBOAT": (u"\xE2\x9B\xB5", ur"""\U000026F5"""
#     ),
#     u"TENT": (u"\xE2\x9B\xBA", ur"""\U000026FA"""
#     ),
#     u"FUEL PUMP": (u"\xE2\x9B\xBD", ur"""\U000026FD"""
#     ),
#     u"ARROW POINTING RIGHTWARDS THEN CURVING UPWARDS": (u"\xE2\xA4\xB4", ur"""\U00002934"""
#     ),
#     u"ARROW POINTING RIGHTWARDS THEN CURVING DOWNWARDS": (u"\xE2\xA4\xB5", ur"""\U00002935"""
#     ),
#     u"LEFTWARDS BLACK ARROW": (u"\xE2\xAC\x85", ur"""\U00002B05"""
#     ),
#     u"UPWARDS BLACK ARROW": (u"\xE2\xAC\x86", ur"""\U00002B06"""
#     ),
#     u"DOWNWARDS BLACK ARROW": (u"\xE2\xAC\x87", ur"""\U00002B07"""
#     ),
#     u"BLACK LARGE SQUARE": (u"\xE2\xAC\x9B", ur"""\U00002B1B"""
#     ),
#     u"WHITE LARGE SQUARE": (u"\xE2\xAC\x9C", ur"""\U00002B1C"""
#     ),
#     u"WHITE MEDIUM STAR": (u"\xE2\xAD\x90", ur"""\U00002B50"""
#     ),
#     u"HEAVY LARGE CIRCLE": (u"\xE2\xAD\x95", ur"""\U00002B55"""
#     ),
#     u"WAVY DASH": (u"\xE3\x80\xB0", ur"""\U00003030"""
#     ),
#     u"PART ALTERNATION MARK": (u"\xE3\x80\xBD", ur"""\U0000303D"""
#     ),
#     u"CIRCLED IDEOGRAPH CONGRATULATION": (u"\xE3\x8A\x97", ur"""\U00003297"""
#     ),
#     u"CIRCLED IDEOGRAPH SECRET": (u"\xE3\x8A\x99", ur"""\U00003299"""
#     ),
#     u"MAHJONG TILE RED DRAGON": (u"\xF0\x9F\x80\x84", ur"""\U0001F004"""
#     ),
#     u"PLAYING CARD BLACK JOKER": (u"\xF0\x9F\x83\x8F", ur"""\U0001F0CF"""
#     ),
#     u"CYCLONE": (u"\xF0\x9F\x8C\x80", ur"""\U0001F300"""
#     ),
#     u"FOGGY": (u"\xF0\x9F\x8C\x81", ur"""\U0001F301"""
#     ),
#     u"CLOSED UMBRELLA": (u"\xF0\x9F\x8C\x82", ur"""\U0001F302"""
#     ),
#     u"NIGHT WITH STARS": (u"\xF0\x9F\x8C\x83", ur"""\U0001F303"""
#     ),
#     u"SUNRISE OVER MOUNTAINS": (u"\xF0\x9F\x8C\x84", ur"""\U0001F304"""
#     ),
#     u"SUNRISE": (u"\xF0\x9F\x8C\x85", ur"""\U0001F305"""
#     ),
#     u"CITYSCAPE AT DUSK": (u"\xF0\x9F\x8C\x86", ur"""\U0001F306"""
#     ),
#     u"SUNSET OVER BUILDINGS": (u"\xF0\x9F\x8C\x87", ur"""\U0001F307"""
#     ),
#     u"RAINBOW": (u"\xF0\x9F\x8C\x88", ur"""\U0001F308"""
#     ),
#     u"BRIDGE AT NIGHT": (u"\xF0\x9F\x8C\x89", ur"""\U0001F309"""
#     ),
#     u"WATER WAVE": (u"\xF0\x9F\x8C\x8A", ur"""\U0001F30A"""
#     ),
#     u"VOLCANO": (u"\xF0\x9F\x8C\x8B", ur"""\U0001F30B"""
#     ),
#     u"MILKY WAY": (u"\xF0\x9F\x8C\x8C", ur"""\U0001F30C"""
#     ),
#     u"EARTH GLOBE ASIA-AUSTRALIA": (u"\xF0\x9F\x8C\x8F", ur"""\U0001F30F"""
#     ),
#     u"NEW MOON SYMBOL": (u"\xF0\x9F\x8C\x91", ur"""\U0001F311"""
#     ),
#     u"FIRST QUARTER MOON SYMBOL": (u"\xF0\x9F\x8C\x93", ur"""\U0001F313"""
#     ),
#     u"WAXING GIBBOUS MOON SYMBOL": (u"\xF0\x9F\x8C\x94", ur"""\U0001F314"""
#     ),
#     u"FULL MOON SYMBOL": (u"\xF0\x9F\x8C\x95", ur"""\U0001F315"""
#     ),
#     u"CRESCENT MOON": (u"\xF0\x9F\x8C\x99", ur"""\U0001F319"""
#     ),
#     u"FIRST QUARTER MOON WITH FACE": (u"\xF0\x9F\x8C\x9B", ur"""\U0001F31B"""
#     ),
#     u"GLOWING STAR": (u"\xF0\x9F\x8C\x9F", ur"""\U0001F31F"""
#     ),
#     u"SHOOTING STAR": (u"\xF0\x9F\x8C\xA0", ur"""\U0001F320"""
#     ),
#     u"CHESTNUT": (u"\xF0\x9F\x8C\xB0", ur"""\U0001F330"""
#     ),
#     u"SEEDLING": (u"\xF0\x9F\x8C\xB1", ur"""\U0001F331"""
#     ),
#     u"PALM TREE": (u"\xF0\x9F\x8C\xB4", ur"""\U0001F334"""
#     ),
#     u"CACTUS": (u"\xF0\x9F\x8C\xB5", ur"""\U0001F335"""
#     ),
#     u"TULIP": (u"\xF0\x9F\x8C\xB7", ur"""\U0001F337"""
#     ),
#     u"CHERRY BLOSSOM": (u"\xF0\x9F\x8C\xB8", ur"""\U0001F338"""
#     ),
#     u"ROSE": (u"\xF0\x9F\x8C\xB9", ur"""\U0001F339"""
#     ),
#     u"HIBISCUS": (u"\xF0\x9F\x8C\xBA", ur"""\U0001F33A"""
#     ),
#     u"SUNFLOWER": (u"\xF0\x9F\x8C\xBB", ur"""\U0001F33B"""
#     ),
#     u"BLOSSOM": (u"\xF0\x9F\x8C\xBC", ur"""\U0001F33C"""
#     ),
#     u"EAR OF MAIZE": (u"\xF0\x9F\x8C\xBD", ur"""\U0001F33D"""
#     ),
#     u"EAR OF RICE": (u"\xF0\x9F\x8C\xBE", ur"""\U0001F33E"""
#     ),
#     u"HERB": (u"\xF0\x9F\x8C\xBF", ur"""\U0001F33F"""
#     ),
#     u"FOUR LEAF CLOVER": (u"\xF0\x9F\x8D\x80", ur"""\U0001F340"""
#     ),
#     u"MAPLE LEAF": (u"\xF0\x9F\x8D\x81", ur"""\U0001F341"""
#     ),
#     u"FALLEN LEAF": (u"\xF0\x9F\x8D\x82", ur"""\U0001F342"""
#     ),
#     u"LEAF FLUTTERING IN WIND": (u"\xF0\x9F\x8D\x83", ur"""\U0001F343"""
#     ),
#     u"MUSHROOM": (u"\xF0\x9F\x8D\x84", ur"""\U0001F344"""
#     ),
#     u"TOMATO": (u"\xF0\x9F\x8D\x85", ur"""\U0001F345"""
#     ),
#     u"AUBERGINE": (u"\xF0\x9F\x8D\x86", ur"""\U0001F346"""
#     ),
#     u"GRAPES": (u"\xF0\x9F\x8D\x87", ur"""\U0001F347"""
#     ),
#     u"MELON": (u"\xF0\x9F\x8D\x88", ur"""\U0001F348"""
#     ),
#     u"WATERMELON": (u"\xF0\x9F\x8D\x89", ur"""\U0001F349"""
#     ),
#     u"TANGERINE": (u"\xF0\x9F\x8D\x8A", ur"""\U0001F34A"""
#     ),
#     u"BANANA": (u"\xF0\x9F\x8D\x8C", ur"""\U0001F34C"""
#     ),
#     u"PINEAPPLE": (u"\xF0\x9F\x8D\x8D", ur"""\U0001F34D"""
#     ),
#     u"RED APPLE": (u"\xF0\x9F\x8D\x8E", ur"""\U0001F34E"""
#     ),
#     u"GREEN APPLE": (u"\xF0\x9F\x8D\x8F", ur"""\U0001F34F"""
#     ),
#     u"PEACH": (u"\xF0\x9F\x8D\x91", ur"""\U0001F351"""
#     ),
#     u"CHERRIES": (u"\xF0\x9F\x8D\x92", ur"""\U0001F352"""
#     ),
#     u"STRAWBERRY": (u"\xF0\x9F\x8D\x93", ur"""\U0001F353"""
#     ),
#     u"HAMBURGER": (u"\xF0\x9F\x8D\x94", ur"""\U0001F354"""
#     ),
#     u"SLICE OF PIZZA": (u"\xF0\x9F\x8D\x95", ur"""\U0001F355"""
#     ),
#     u"MEAT ON BONE": (u"\xF0\x9F\x8D\x96", ur"""\U0001F356"""
#     ),
#     u"POULTRY LEG": (u"\xF0\x9F\x8D\x97", ur"""\U0001F357"""
#     ),
#     u"RICE CRACKER": (u"\xF0\x9F\x8D\x98", ur"""\U0001F358"""
#     ),
#     u"RICE BALL": (u"\xF0\x9F\x8D\x99", ur"""\U0001F359"""
#     ),
#     u"COOKED RICE": (u"\xF0\x9F\x8D\x9A", ur"""\U0001F35A"""
#     ),
#     u"CURRY AND RICE": (u"\xF0\x9F\x8D\x9B", ur"""\U0001F35B"""
#     ),
#     u"STEAMING BOWL": (u"\xF0\x9F\x8D\x9C", ur"""\U0001F35C"""
#     ),
#     u"SPAGHETTI": (u"\xF0\x9F\x8D\x9D", ur"""\U0001F35D"""
#     ),
#     u"BREAD": (u"\xF0\x9F\x8D\x9E", ur"""\U0001F35E"""
#     ),
#     u"FRENCH FRIES": (u"\xF0\x9F\x8D\x9F", ur"""\U0001F35F"""
#     ),
#     u"ROASTED SWEET POTATO": (u"\xF0\x9F\x8D\xA0", ur"""\U0001F360"""
#     ),
#     u"DANGO": (u"\xF0\x9F\x8D\xA1", ur"""\U0001F361"""
#     ),
#     u"ODEN": (u"\xF0\x9F\x8D\xA2", ur"""\U0001F362"""
#     ),
#     u"SUSHI": (u"\xF0\x9F\x8D\xA3", ur"""\U0001F363"""
#     ),
#     u"FRIED SHRIMP": (u"\xF0\x9F\x8D\xA4", ur"""\U0001F364"""
#     ),
#     u"FISH CAKE WITH SWIRL DESIGN": (u"\xF0\x9F\x8D\xA5", ur"""\U0001F365"""
#     ),
#     u"SOFT ICE CREAM": (u"\xF0\x9F\x8D\xA6", ur"""\U0001F366"""
#     ),
#     u"SHAVED ICE": (u"\xF0\x9F\x8D\xA7", ur"""\U0001F367"""
#     ),
#     u"ICE CREAM": (u"\xF0\x9F\x8D\xA8", ur"""\U0001F368"""
#     ),
#     u"DOUGHNUT": (u"\xF0\x9F\x8D\xA9", ur"""\U0001F369"""
#     ),
#     u"COOKIE": (u"\xF0\x9F\x8D\xAA", ur"""\U0001F36A"""
#     ),
#     u"CHOCOLATE BAR": (u"\xF0\x9F\x8D\xAB", ur"""\U0001F36B"""
#     ),
#     u"CANDY": (u"\xF0\x9F\x8D\xAC", ur"""\U0001F36C"""
#     ),
#     u"LOLLIPOP": (u"\xF0\x9F\x8D\xAD", ur"""\U0001F36D"""
#     ),
#     u"CUSTARD": (u"\xF0\x9F\x8D\xAE", ur"""\U0001F36E"""
#     ),
#     u"HONEY POT": (u"\xF0\x9F\x8D\xAF", ur"""\U0001F36F"""
#     ),
#     u"SHORTCAKE": (u"\xF0\x9F\x8D\xB0", ur"""\U0001F370"""
#     ),
#     u"BENTO BOX": (u"\xF0\x9F\x8D\xB1", ur"""\U0001F371"""
#     ),
#     u"POT OF FOOD": (u"\xF0\x9F\x8D\xB2", ur"""\U0001F372"""
#     ),
#     u"COOKING": (u"\xF0\x9F\x8D\xB3", ur"""\U0001F373"""
#     ),
#     u"FORK AND KNIFE": (u"\xF0\x9F\x8D\xB4", ur"""\U0001F374"""
#     ),
#     u"TEACUP WITHOUT HANDLE": (u"\xF0\x9F\x8D\xB5", ur"""\U0001F375"""
#     ),
#     u"SAKE BOTTLE AND CUP": (u"\xF0\x9F\x8D\xB6", ur"""\U0001F376"""
#     ),
#     u"WINE GLASS": (u"\xF0\x9F\x8D\xB7", ur"""\U0001F377"""
#     ),
#     u"COCKTAIL GLASS": (u"\xF0\x9F\x8D\xB8", ur"""\U0001F378"""
#     ),
#     u"TROPICAL DRINK": (u"\xF0\x9F\x8D\xB9", ur"""\U0001F379"""
#     ),
#     u"BEER MUG": (u"\xF0\x9F\x8D\xBA", ur"""\U0001F37A"""
#     ),
#     u"CLINKING BEER MUGS": (u"\xF0\x9F\x8D\xBB", ur"""\U0001F37B"""
#     ),
#     u"RIBBON": (u"\xF0\x9F\x8E\x80", ur"""\U0001F380"""
#     ),
#     u"WRAPPED PRESENT": (u"\xF0\x9F\x8E\x81", ur"""\U0001F381"""
#     ),
#     u"BIRTHDAY CAKE": (u"\xF0\x9F\x8E\x82", ur"""\U0001F382"""
#     ),
#     u"JACK-O-LANTERN": (u"\xF0\x9F\x8E\x83", ur"""\U0001F383"""
#     ),
#     u"CHRISTMAS TREE": (u"\xF0\x9F\x8E\x84", ur"""\U0001F384"""
#     ),
#     u"FATHER CHRISTMAS": (u"\xF0\x9F\x8E\x85", ur"""\U0001F385"""
#     ),
#     u"FIREWORKS": (u"\xF0\x9F\x8E\x86", ur"""\U0001F386"""
#     ),
#     u"FIREWORK SPARKLER": (u"\xF0\x9F\x8E\x87", ur"""\U0001F387"""
#     ),
#     u"BALLOON": (u"\xF0\x9F\x8E\x88", ur"""\U0001F388"""
#     ),
#     u"PARTY POPPER": (u"\xF0\x9F\x8E\x89", ur"""\U0001F389"""
#     ),
#     u"CONFETTI BALL": (u"\xF0\x9F\x8E\x8A", ur"""\U0001F38A"""
#     ),
#     u"TANABATA TREE": (u"\xF0\x9F\x8E\x8B", ur"""\U0001F38B"""
#     ),
#     u"CROSSED FLAGS": (u"\xF0\x9F\x8E\x8C", ur"""\U0001F38C"""
#     ),
#     u"PINE DECORATION": (u"\xF0\x9F\x8E\x8D", ur"""\U0001F38D"""
#     ),
#     u"JAPANESE DOLLS": (u"\xF0\x9F\x8E\x8E", ur"""\U0001F38E"""
#     ),
#     u"CARP STREAMER": (u"\xF0\x9F\x8E\x8F", ur"""\U0001F38F"""
#     ),
#     u"WIND CHIME": (u"\xF0\x9F\x8E\x90", ur"""\U0001F390"""
#     ),
#     u"MOON VIEWING CEREMONY": (u"\xF0\x9F\x8E\x91", ur"""\U0001F391"""
#     ),
#     u"SCHOOL SATCHEL": (u"\xF0\x9F\x8E\x92", ur"""\U0001F392"""
#     ),
#     u"GRADUATION CAP": (u"\xF0\x9F\x8E\x93", ur"""\U0001F393"""
#     ),
#     u"CAROUSEL HORSE": (u"\xF0\x9F\x8E\xA0", ur"""\U0001F3A0"""
#     ),
#     u"FERRIS WHEEL": (u"\xF0\x9F\x8E\xA1", ur"""\U0001F3A1"""
#     ),
#     u"ROLLER COASTER": (u"\xF0\x9F\x8E\xA2", ur"""\U0001F3A2"""
#     ),
#     u"FISHING POLE AND FISH": (u"\xF0\x9F\x8E\xA3", ur"""\U0001F3A3"""
#     ),
#     u"MICROPHONE": (u"\xF0\x9F\x8E\xA4", ur"""\U0001F3A4"""
#     ),
#     u"MOVIE CAMERA": (u"\xF0\x9F\x8E\xA5", ur"""\U0001F3A5"""
#     ),
#     u"CINEMA": (u"\xF0\x9F\x8E\xA6", ur"""\U0001F3A6"""
#     ),
#     u"HEADPHONE": (u"\xF0\x9F\x8E\xA7", ur"""\U0001F3A7"""
#     ),
#     u"ARTIST PALETTE": (u"\xF0\x9F\x8E\xA8", ur"""\U0001F3A8"""
#     ),
#     u"TOP HAT": (u"\xF0\x9F\x8E\xA9", ur"""\U0001F3A9"""
#     ),
#     u"CIRCUS TENT": (u"\xF0\x9F\x8E\xAA", ur"""\U0001F3AA"""
#     ),
#     u"TICKET": (u"\xF0\x9F\x8E\xAB", ur"""\U0001F3AB"""
#     ),
#     u"CLAPPER BOARD": (u"\xF0\x9F\x8E\xAC", ur"""\U0001F3AC"""
#     ),
#     u"PERFORMING ARTS": (u"\xF0\x9F\x8E\xAD", ur"""\U0001F3AD"""
#     ),
#     u"VIDEO GAME": (u"\xF0\x9F\x8E\xAE", ur"""\U0001F3AE"""
#     ),
#     u"DIRECT HIT": (u"\xF0\x9F\x8E\xAF", ur"""\U0001F3AF"""
#     ),
#     u"SLOT MACHINE": (u"\xF0\x9F\x8E\xB0", ur"""\U0001F3B0"""
#     ),
#     u"BILLIARDS": (u"\xF0\x9F\x8E\xB1", ur"""\U0001F3B1"""
#     ),
#     u"GAME DIE": (u"\xF0\x9F\x8E\xB2", ur"""\U0001F3B2"""
#     ),
#     u"BOWLING": (u"\xF0\x9F\x8E\xB3", ur"""\U0001F3B3"""
#     ),
#     u"FLOWER PLAYING CARDS": (u"\xF0\x9F\x8E\xB4", ur"""\U0001F3B4"""
#     ),
#     u"MUSICAL NOTE": (u"\xF0\x9F\x8E\xB5", ur"""\U0001F3B5"""
#     ),
#     u"MULTIPLE MUSICAL NOTES": (u"\xF0\x9F\x8E\xB6", ur"""\U0001F3B6"""
#     ),
#     u"SAXOPHONE": (u"\xF0\x9F\x8E\xB7", ur"""\U0001F3B7"""
#     ),
#     u"GUITAR": (u"\xF0\x9F\x8E\xB8", ur"""\U0001F3B8"""
#     ),
#     u"MUSICAL KEYBOARD": (u"\xF0\x9F\x8E\xB9", ur"""\U0001F3B9"""
#     ),
#     u"TRUMPET": (u"\xF0\x9F\x8E\xBA", ur"""\U0001F3BA"""
#     ),
#     u"VIOLIN": (u"\xF0\x9F\x8E\xBB", ur"""\U0001F3BB"""
#     ),
#     u"MUSICAL SCORE": (u"\xF0\x9F\x8E\xBC", ur"""\U0001F3BC"""
#     ),
#     u"RUNNING SHIRT WITH SASH": (u"\xF0\x9F\x8E\xBD", ur"""\U0001F3BD"""
#     ),
#     u"TENNIS RACQUET AND BALL": (u"\xF0\x9F\x8E\xBE", ur"""\U0001F3BE"""
#     ),
#     u"SKI AND SKI BOOT": (u"\xF0\x9F\x8E\xBF", ur"""\U0001F3BF"""
#     ),
#     u"BASKETBALL AND HOOP": (u"\xF0\x9F\x8F\x80", ur"""\U0001F3C0"""
#     ),
#     u"CHEQUERED FLAG": (u"\xF0\x9F\x8F\x81", ur"""\U0001F3C1"""
#     ),
#     u"SNOWBOARDER": (u"\xF0\x9F\x8F\x82", ur"""\U0001F3C2"""
#     ),
#     u"RUNNER": (u"\xF0\x9F\x8F\x83", ur"""\U0001F3C3"""
#     ),
#     u"SURFER": (u"\xF0\x9F\x8F\x84", ur"""\U0001F3C4"""
#     ),
#     u"TROPHY": (u"\xF0\x9F\x8F\x86", ur"""\U0001F3C6"""
#     ),
#     u"AMERICAN FOOTBALL": (u"\xF0\x9F\x8F\x88", ur"""\U0001F3C8"""
#     ),
#     u"SWIMMER": (u"\xF0\x9F\x8F\x8A", ur"""\U0001F3CA"""
#     ),
#     u"HOUSE BUILDING": (u"\xF0\x9F\x8F\xA0", ur"""\U0001F3E0"""
#     ),
#     u"HOUSE WITH GARDEN": (u"\xF0\x9F\x8F\xA1", ur"""\U0001F3E1"""
#     ),
#     u"OFFICE BUILDING": (u"\xF0\x9F\x8F\xA2", ur"""\U0001F3E2"""
#     ),
#     u"JAPANESE POST OFFICE": (u"\xF0\x9F\x8F\xA3", ur"""\U0001F3E3"""
#     ),
#     u"HOSPITAL": (u"\xF0\x9F\x8F\xA5", ur"""\U0001F3E5"""
#     ),
#     u"BANK": (u"\xF0\x9F\x8F\xA6", ur"""\U0001F3E6"""
#     ),
#     u"AUTOMATED TELLER MACHINE": (u"\xF0\x9F\x8F\xA7", ur"""\U0001F3E7"""
#     ),
#     u"HOTEL": (u"\xF0\x9F\x8F\xA8", ur"""\U0001F3E8"""
#     ),
#     u"LOVE HOTEL": (u"\xF0\x9F\x8F\xA9", ur"""\U0001F3E9"""
#     ),
#     u"CONVENIENCE STORE": (u"\xF0\x9F\x8F\xAA", ur"""\U0001F3EA"""
#     ),
#     u"SCHOOL": (u"\xF0\x9F\x8F\xAB", ur"""\U0001F3EB"""
#     ),
#     u"DEPARTMENT STORE": (u"\xF0\x9F\x8F\xAC", ur"""\U0001F3EC"""
#     ),
#     u"FACTORY": (u"\xF0\x9F\x8F\xAD", ur"""\U0001F3ED"""
#     ),
#     u"IZAKAYA LANTERN": (u"\xF0\x9F\x8F\xAE", ur"""\U0001F3EE"""
#     ),
#     u"JAPANESE CASTLE": (u"\xF0\x9F\x8F\xAF", ur"""\U0001F3EF"""
#     ),
#     u"EUROPEAN CASTLE": (u"\xF0\x9F\x8F\xB0", ur"""\U0001F3F0"""
#     ),
#     u"SNAIL": (u"\xF0\x9F\x90\x8C", ur"""\U0001F40C"""
#     ),
#     u"SNAKE": (u"\xF0\x9F\x90\x8D", ur"""\U0001F40D"""
#     ),
#     u"HORSE": (u"\xF0\x9F\x90\x8E", ur"""\U0001F40E"""
#     ),
#     u"SHEEP": (u"\xF0\x9F\x90\x91", ur"""\U0001F411"""
#     ),
#     u"MONKEY": (u"\xF0\x9F\x90\x92", ur"""\U0001F412"""
#     ),
#     u"CHICKEN": (u"\xF0\x9F\x90\x94", ur"""\U0001F414"""
#     ),
#     u"BOAR": (u"\xF0\x9F\x90\x97", ur"""\U0001F417"""
#     ),
#     u"ELEPHANT": (u"\xF0\x9F\x90\x98", ur"""\U0001F418"""
#     ),
#     u"OCTOPUS": (u"\xF0\x9F\x90\x99", ur"""\U0001F419"""
#     ),
#     u"SPIRAL SHELL": (u"\xF0\x9F\x90\x9A", ur"""\U0001F41A"""
#     ),
#     u"BUG": (u"\xF0\x9F\x90\x9B", ur"""\U0001F41B"""
#     ),
#     u"ANT": (u"\xF0\x9F\x90\x9C", ur"""\U0001F41C"""
#     ),
#     u"HONEYBEE": (u"\xF0\x9F\x90\x9D", ur"""\U0001F41D"""
#     ),
#     u"LADY BEETLE": (u"\xF0\x9F\x90\x9E", ur"""\U0001F41E"""
#     ),
#     u"FISH": (u"\xF0\x9F\x90\x9F", ur"""\U0001F41F"""
#     ),
#     u"TROPICAL FISH": (u"\xF0\x9F\x90\xA0", ur"""\U0001F420"""
#     ),
#     u"BLOWFISH": (u"\xF0\x9F\x90\xA1", ur"""\U0001F421"""
#     ),
#     u"TURTLE": (u"\xF0\x9F\x90\xA2", ur"""\U0001F422"""
#     ),
#     u"HATCHING CHICK": (u"\xF0\x9F\x90\xA3", ur"""\U0001F423"""
#     ),
#     u"BABY CHICK": (u"\xF0\x9F\x90\xA4", ur"""\U0001F424"""
#     ),
#     u"FRONT-FACING BABY CHICK": (u"\xF0\x9F\x90\xA5", ur"""\U0001F425"""
#     ),
#     u"BIRD": (u"\xF0\x9F\x90\xA6", ur"""\U0001F426"""
#     ),
#     u"PENGUIN": (u"\xF0\x9F\x90\xA7", ur"""\U0001F427"""
#     ),
#     u"KOALA": (u"\xF0\x9F\x90\xA8", ur"""\U0001F428"""
#     ),
#     u"POODLE": (u"\xF0\x9F\x90\xA9", ur"""\U0001F429"""
#     ),
#     u"BACTRIAN CAMEL": (u"\xF0\x9F\x90\xAB", ur"""\U0001F42B"""
#     ),
#     u"DOLPHIN": (u"\xF0\x9F\x90\xAC", ur"""\U0001F42C"""
#     ),
#     u"MOUSE FACE": (u"\xF0\x9F\x90\xAD", ur"""\U0001F42D"""
#     ),
#     u"COW FACE": (u"\xF0\x9F\x90\xAE", ur"""\U0001F42E"""
#     ),
#     u"TIGER FACE": (u"\xF0\x9F\x90\xAF", ur"""\U0001F42F"""
#     ),
#     u"RABBIT FACE": (u"\xF0\x9F\x90\xB0", ur"""\U0001F430"""
#     ),
#     u"CAT FACE": (u"\xF0\x9F\x90\xB1", ur"""\U0001F431"""
#     ),
#     u"DRAGON FACE": (u"\xF0\x9F\x90\xB2", ur"""\U0001F432"""
#     ),
#     u"SPOUTING WHALE": (u"\xF0\x9F\x90\xB3", ur"""\U0001F433"""
#     ),
#     u"HORSE FACE": (u"\xF0\x9F\x90\xB4", ur"""\U0001F434"""
#     ),
#     u"MONKEY FACE": (u"\xF0\x9F\x90\xB5", ur"""\U0001F435"""
#     ),
#     u"DOG FACE": (u"\xF0\x9F\x90\xB6", ur"""\U0001F436"""
#     ),
#     u"PIG FACE": (u"\xF0\x9F\x90\xB7", ur"""\U0001F437"""
#     ),
#     u"FROG FACE": (u"\xF0\x9F\x90\xB8", ur"""\U0001F438"""
#     ),
#     u"HAMSTER FACE": (u"\xF0\x9F\x90\xB9", ur"""\U0001F439"""
#     ),
#     u"WOLF FACE": (u"\xF0\x9F\x90\xBA", ur"""\U0001F43A"""
#     ),
#     u"BEAR FACE": (u"\xF0\x9F\x90\xBB", ur"""\U0001F43B"""
#     ),
#     u"PANDA FACE": (u"\xF0\x9F\x90\xBC", ur"""\U0001F43C"""
#     ),
#     u"PIG NOSE": (u"\xF0\x9F\x90\xBD", ur"""\U0001F43D"""
#     ),
#     u"PAW PRINTS": (u"\xF0\x9F\x90\xBE", ur"""\U0001F43E"""
#     ),
#     u"EYES": (u"\xF0\x9F\x91\x80", ur"""\U0001F440"""
#     ),
#     u"EAR": (u"\xF0\x9F\x91\x82", ur"""\U0001F442"""
#     ),
#     u"NOSE": (u"\xF0\x9F\x91\x83", ur"""\U0001F443"""
#     ),
#     u"MOUTH": (u"\xF0\x9F\x91\x84", ur"""\U0001F444"""
#     ),
#     u"TONGUE": (u"\xF0\x9F\x91\x85", ur"""\U0001F445"""
#     ),
#     u"WHITE UP POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x86", ur"""\U0001F446"""
#     ),
#     u"WHITE DOWN POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x87", ur"""\U0001F447"""
#     ),
#     u"WHITE LEFT POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x88", ur"""\U0001F448"""
#     ),
#     u"WHITE RIGHT POINTING BACKHAND INDEX": (u"\xF0\x9F\x91\x89", ur"""\U0001F449"""
#     ),
#     u"FISTED HAND SIGN": (u"\xF0\x9F\x91\x8A", ur"""\U0001F44A"""
#     ),
#     u"WAVING HAND SIGN": (u"\xF0\x9F\x91\x8B", ur"""\U0001F44B"""
#     ),
#     u"OK HAND SIGN": (u"\xF0\x9F\x91\x8C", ur"""\U0001F44C"""
#     ),
#     u"THUMBS UP SIGN": (u"\xF0\x9F\x91\x8D", ur"""\U0001F44D"""
#     ),
#     u"THUMBS DOWN SIGN": (u"\xF0\x9F\x91\x8E", ur"""\U0001F44E"""
#     ),
#     u"CLAPPING HANDS SIGN": (u"\xF0\x9F\x91\x8F", ur"""\U0001F44F"""
#     ),
#     u"OPEN HANDS SIGN": (u"\xF0\x9F\x91\x90", ur"""\U0001F450"""
#     ),
    u"CROWN": (u"\xF0\x9F\x91\x91", ur"""\U0001F451"""
    ),
#     u"WOMANS HAT": (u"\xF0\x9F\x91\x92", ur"""\U0001F452"""
#     ),
#     u"EYEGLASSES": (u"\xF0\x9F\x91\x93", ur"""\U0001F453"""
#     ),
#     u"NECKTIE": (u"\xF0\x9F\x91\x94", ur"""\U0001F454"""
#     ),
#     u"T-SHIRT": (u"\xF0\x9F\x91\x95", ur"""\U0001F455"""
#     ),
#     u"JEANS": (u"\xF0\x9F\x91\x96", ur"""\U0001F456"""
#     ),
#     u"DRESS": (u"\xF0\x9F\x91\x97", ur"""\U0001F457"""
#     ),
#     u"KIMONO": (u"\xF0\x9F\x91\x98", ur"""\U0001F458"""
#     ),
#     u"BIKINI": (u"\xF0\x9F\x91\x99", ur"""\U0001F459"""
#     ),
#     u"WOMANS CLOTHES": (u"\xF0\x9F\x91\x9A", ur"""\U0001F45A"""
#     ),
#     u"PURSE": (u"\xF0\x9F\x91\x9B", ur"""\U0001F45B"""
#     ),
#     u"HANDBAG": (u"\xF0\x9F\x91\x9C", ur"""\U0001F45C"""
#     ),
#     u"POUCH": (u"\xF0\x9F\x91\x9D", ur"""\U0001F45D"""
#     ),
#     u"MANS SHOE": (u"\xF0\x9F\x91\x9E", ur"""\U0001F45E"""
#     ),
#     u"ATHLETIC SHOE": (u"\xF0\x9F\x91\x9F", ur"""\U0001F45F"""
#     ),
#     u"HIGH-HEELED SHOE": (u"\xF0\x9F\x91\xA0", ur"""\U0001F460"""
#     ),
#     u"WOMANS SANDAL": (u"\xF0\x9F\x91\xA1", ur"""\U0001F461"""
#     ),
#     u"WOMANS BOOTS": (u"\xF0\x9F\x91\xA2", ur"""\U0001F462"""
#     ),
#     u"FOOTPRINTS": (u"\xF0\x9F\x91\xA3", ur"""\U0001F463"""
#     ),
#     u"BUST IN SILHOUETTE": (u"\xF0\x9F\x91\xA4", ur"""\U0001F464"""
#     ),
#     u"BOY": (u"\xF0\x9F\x91\xA6", ur"""\U0001F466"""
#     ),
#     u"GIRL": (u"\xF0\x9F\x91\xA7", ur"""\U0001F467"""
#     ),
#     u"MAN": (u"\xF0\x9F\x91\xA8", ur"""\U0001F468"""
#     ),
#     u"WOMAN": (u"\xF0\x9F\x91\xA9", ur"""\U0001F469"""
#     ),
#     u"FAMILY": (u"\xF0\x9F\x91\xAA", ur"""\U0001F46A"""
#     ),
#     u"MAN AND WOMAN HOLDING HANDS": (u"\xF0\x9F\x91\xAB", ur"""\U0001F46B"""
#     ),
#     u"POLICE OFFICER": (u"\xF0\x9F\x91\xAE", ur"""\U0001F46E"""
#     ),
#     u"WOMAN WITH BUNNY EARS": (u"\xF0\x9F\x91\xAF", ur"""\U0001F46F"""
#     ),
#     u"BRIDE WITH VEIL": (u"\xF0\x9F\x91\xB0", ur"""\U0001F470"""
#     ),
#     u"PERSON WITH BLOND HAIR": (u"\xF0\x9F\x91\xB1", ur"""\U0001F471"""
#     ),
#     u"MAN WITH GUA PI MAO": (u"\xF0\x9F\x91\xB2", ur"""\U0001F472"""
#     ),
#     u"MAN WITH TURBAN": (u"\xF0\x9F\x91\xB3", ur"""\U0001F473"""
#     ),
#     u"OLDER MAN": (u"\xF0\x9F\x91\xB4", ur"""\U0001F474"""
#     ),
#     u"OLDER WOMAN": (u"\xF0\x9F\x91\xB5", ur"""\U0001F475"""
#     ),
#     u"BABY": (u"\xF0\x9F\x91\xB6", ur"""\U0001F476"""
#     ),
#     u"CONSTRUCTION WORKER": (u"\xF0\x9F\x91\xB7", ur"""\U0001F477"""
#     ),
#     u"PRINCESS": (u"\xF0\x9F\x91\xB8", ur"""\U0001F478"""
#     ),
#     u"JAPANESE OGRE": (u"\xF0\x9F\x91\xB9", ur"""\U0001F479"""
#     ),
#     u"JAPANESE GOBLIN": (u"\xF0\x9F\x91\xBA", ur"""\U0001F47A"""
#     ),
#     u"GHOST": (u"\xF0\x9F\x91\xBB", ur"""\U0001F47B"""
#     ),
#     u"BABY ANGEL": (u"\xF0\x9F\x91\xBC", ur"""\U0001F47C"""
#     ),
#     u"EXTRATERRESTRIAL ALIEN": (u"\xF0\x9F\x91\xBD", ur"""\U0001F47D"""
#     ),
#     u"ALIEN MONSTER": (u"\xF0\x9F\x91\xBE", ur"""\U0001F47E"""
#     ),
#     u"IMP": (u"\xF0\x9F\x91\xBF", ur"""\U0001F47F"""
#     ),
#     u"SKULL": (u"\xF0\x9F\x92\x80", ur"""\U0001F480"""
#     ),
#     u"INFORMATION DESK PERSON": (u"\xF0\x9F\x92\x81", ur"""\U0001F481"""
#     ),
#     u"GUARDSMAN": (u"\xF0\x9F\x92\x82", ur"""\U0001F482"""
#     ),
#     u"DANCER": (u"\xF0\x9F\x92\x83", ur"""\U0001F483"""
#     ),
#     u"LIPSTICK": (u"\xF0\x9F\x92\x84", ur"""\U0001F484"""
#     ),
#     u"NAIL POLISH": (u"\xF0\x9F\x92\x85", ur"""\U0001F485"""
#     ),
#     u"FACE MASSAGE": (u"\xF0\x9F\x92\x86", ur"""\U0001F486"""
#     ),
#     u"HAIRCUT": (u"\xF0\x9F\x92\x87", ur"""\U0001F487"""
#     ),
#     u"BARBER POLE": (u"\xF0\x9F\x92\x88", ur"""\U0001F488"""
#     ),
#     u"SYRINGE": (u"\xF0\x9F\x92\x89", ur"""\U0001F489"""
#     ),
#     u"PILL": (u"\xF0\x9F\x92\x8A", ur"""\U0001F48A"""
#     ),
#     u"KISS MARK": (u"\xF0\x9F\x92\x8B", ur"""\U0001F48B"""
#     ),
#     u"LOVE LETTER": (u"\xF0\x9F\x92\x8C", ur"""\U0001F48C"""
#     ),
#     u"RING": (u"\xF0\x9F\x92\x8D", ur"""\U0001F48D"""
#     ),
#     u"GEM STONE": (u"\xF0\x9F\x92\x8E", ur"""\U0001F48E"""
#     ),
#     u"KISS": (u"\xF0\x9F\x92\x8F", ur"""\U0001F48F"""
#     ),
#     u"BOUQUET": (u"\xF0\x9F\x92\x90", ur"""\U0001F490"""
#     ),
#     u"COUPLE WITH HEART": (u"\xF0\x9F\x92\x91", ur"""\U0001F491"""
#     ),
#     u"WEDDING": (u"\xF0\x9F\x92\x92", ur"""\U0001F492"""
#     ),
#     u"BEATING HEART": (u"\xF0\x9F\x92\x93", ur"""\U0001F493"""
#     ),
#     u"BROKEN HEART": (u"\xF0\x9F\x92\x94", ur"""\U0001F494"""
#     ),
#     u"TWO HEARTS": (u"\xF0\x9F\x92\x95", ur"""\U0001F495"""
#     ),
#     u"SPARKLING HEART": (u"\xF0\x9F\x92\x96", ur"""\U0001F496"""
#     ),
#     u"GROWING HEART": (u"\xF0\x9F\x92\x97", ur"""\U0001F497"""
#     ),
#     u"HEART WITH ARROW": (u"\xF0\x9F\x92\x98", ur"""\U0001F498"""
#     ),
#     u"BLUE HEART": (u"\xF0\x9F\x92\x99", ur"""\U0001F499"""
#     ),
#     u"GREEN HEART": (u"\xF0\x9F\x92\x9A", ur"""\U0001F49A"""
#     ),
#     u"YELLOW HEART": (u"\xF0\x9F\x92\x9B", ur"""\U0001F49B"""
#     ),
#     u"PURPLE HEART": (u"\xF0\x9F\x92\x9C", ur"""\U0001F49C"""
#     ),
#     u"HEART WITH RIBBON": (u"\xF0\x9F\x92\x9D", ur"""\U0001F49D"""
#     ),
#     u"REVOLVING HEARTS": (u"\xF0\x9F\x92\x9E", ur"""\U0001F49E"""
#     ),
#     u"HEART DECORATION": (u"\xF0\x9F\x92\x9F", ur"""\U0001F49F"""
#     ),
#     u"DIAMOND SHAPE WITH A DOT INSIDE": (u"\xF0\x9F\x92\xA0", ur"""\U0001F4A0"""
#     ),
#     u"ELECTRIC LIGHT BULB": (u"\xF0\x9F\x92\xA1", ur"""\U0001F4A1"""
#     ),
#     u"ANGER SYMBOL": (u"\xF0\x9F\x92\xA2", ur"""\U0001F4A2"""
#     ),
#     u"BOMB": (u"\xF0\x9F\x92\xA3", ur"""\U0001F4A3"""
#     ),
#     u"SLEEPING SYMBOL": (u"\xF0\x9F\x92\xA4", ur"""\U0001F4A4"""
#     ),
#     u"COLLISION SYMBOL": (u"\xF0\x9F\x92\xA5", ur"""\U0001F4A5"""
#     ),
#     u"SPLASHING SWEAT SYMBOL": (u"\xF0\x9F\x92\xA6", ur"""\U0001F4A6"""
#     ),
#     u"DROPLET": (u"\xF0\x9F\x92\xA7", ur"""\U0001F4A7"""
#     ),
#     u"DASH SYMBOL": (u"\xF0\x9F\x92\xA8", ur"""\U0001F4A8"""
#     ),
#     u"PILE OF POO": (u"\xF0\x9F\x92\xA9", ur"""\U0001F4A9"""
#     ),
#     u"FLEXED BICEPS": (u"\xF0\x9F\x92\xAA", ur"""\U0001F4AA"""
#     ),
#     u"DIZZY SYMBOL": (u"\xF0\x9F\x92\xAB", ur"""\U0001F4AB"""
#     ),
#     u"SPEECH BALLOON": (u"\xF0\x9F\x92\xAC", ur"""\U0001F4AC"""
#     ),
#     u"WHITE FLOWER": (u"\xF0\x9F\x92\xAE", ur"""\U0001F4AE"""
#     ),
#     u"HUNDRED POINTS SYMBOL": (u"\xF0\x9F\x92\xAF", ur"""\U0001F4AF"""
#     ),
    u"MONEY BAG": (u"\xF0\x9F\x92\xB0", ur"""\U0001F4B0"""
    ),
#     u"CURRENCY EXCHANGE": (u"\xF0\x9F\x92\xB1", ur"""\U0001F4B1"""
#     ),
#     u"HEAVY DOLLAR SIGN": (u"\xF0\x9F\x92\xB2", ur"""\U0001F4B2"""
#     ),
#     u"CREDIT CARD": (u"\xF0\x9F\x92\xB3", ur"""\U0001F4B3"""
#     ),
#     u"BANKNOTE WITH YEN SIGN": (u"\xF0\x9F\x92\xB4", ur"""\U0001F4B4"""
#     ),
#     u"BANKNOTE WITH DOLLAR SIGN": (u"\xF0\x9F\x92\xB5", ur"""\U0001F4B5"""
#     ),
#     u"MONEY WITH WINGS": (u"\xF0\x9F\x92\xB8", ur"""\U0001F4B8"""
#     ),
#     u"CHART WITH UPWARDS TREND AND YEN SIGN": (u"\xF0\x9F\x92\xB9", ur"""\U0001F4B9"""
#     ),
#     u"SEAT": (u"\xF0\x9F\x92\xBA", ur"""\U0001F4BA"""
#     ),
#     u"PERSONAL COMPUTER": (u"\xF0\x9F\x92\xBB", ur"""\U0001F4BB"""
#     ),
#     u"BRIEFCASE": (u"\xF0\x9F\x92\xBC", ur"""\U0001F4BC"""
#     ),
#     u"MINIDISC": (u"\xF0\x9F\x92\xBD", ur"""\U0001F4BD"""
#     ),
#     u"FLOPPY DISK": (u"\xF0\x9F\x92\xBE", ur"""\U0001F4BE"""
#     ),
#     u"OPTICAL DISC": (u"\xF0\x9F\x92\xBF", ur"""\U0001F4BF"""
#     ),
#     u"DVD": (u"\xF0\x9F\x93\x80", ur"""\U0001F4C0"""
#     ),
#     u"FILE FOLDER": (u"\xF0\x9F\x93\x81", ur"""\U0001F4C1"""
#     ),
#     u"OPEN FILE FOLDER": (u"\xF0\x9F\x93\x82", ur"""\U0001F4C2"""
#     ),
#     u"PAGE WITH CURL": (u"\xF0\x9F\x93\x83", ur"""\U0001F4C3"""
#     ),
#     u"PAGE FACING UP": (u"\xF0\x9F\x93\x84", ur"""\U0001F4C4"""
#     ),
#     u"CALENDAR": (u"\xF0\x9F\x93\x85", ur"""\U0001F4C5"""
#     ),
#     u"TEAR-OFF CALENDAR": (u"\xF0\x9F\x93\x86", ur"""\U0001F4C6"""
#     ),
#     u"CARD INDEX": (u"\xF0\x9F\x93\x87", ur"""\U0001F4C7"""
#     ),
#     u"CHART WITH UPWARDS TREND": (u"\xF0\x9F\x93\x88", ur"""\U0001F4C8"""
#     ),
#     u"CHART WITH DOWNWARDS TREND": (u"\xF0\x9F\x93\x89", ur"""\U0001F4C9"""
#     ),
#     u"BAR CHART": (u"\xF0\x9F\x93\x8A", ur"""\U0001F4CA"""
#     ),
#     u"CLIPBOARD": (u"\xF0\x9F\x93\x8B", ur"""\U0001F4CB"""
#     ),
#     u"PUSHPIN": (u"\xF0\x9F\x93\x8C", ur"""\U0001F4CC"""
#     ),
#     u"ROUND PUSHPIN": (u"\xF0\x9F\x93\x8D", ur"""\U0001F4CD"""
#     ),
#     u"PAPERCLIP": (u"\xF0\x9F\x93\x8E", ur"""\U0001F4CE"""
#     ),
#     u"STRAIGHT RULER": (u"\xF0\x9F\x93\x8F", ur"""\U0001F4CF"""
#     ),
#     u"TRIANGULAR RULER": (u"\xF0\x9F\x93\x90", ur"""\U0001F4D0"""
#     ),
#     u"BOOKMARK TABS": (u"\xF0\x9F\x93\x91", ur"""\U0001F4D1"""
#     ),
#     u"LEDGER": (u"\xF0\x9F\x93\x92", ur"""\U0001F4D2"""
#     ),
#     u"NOTEBOOK": (u"\xF0\x9F\x93\x93", ur"""\U0001F4D3"""
#     ),
#     u"NOTEBOOK WITH DECORATIVE COVER": (u"\xF0\x9F\x93\x94", ur"""\U0001F4D4"""
#     ),
#     u"CLOSED BOOK": (u"\xF0\x9F\x93\x95", ur"""\U0001F4D5"""
#     ),
#     u"OPEN BOOK": (u"\xF0\x9F\x93\x96", ur"""\U0001F4D6"""
#     ),
#     u"GREEN BOOK": (u"\xF0\x9F\x93\x97", ur"""\U0001F4D7"""
#     ),
#     u"BLUE BOOK": (u"\xF0\x9F\x93\x98", ur"""\U0001F4D8"""
#     ),
#     u"ORANGE BOOK": (u"\xF0\x9F\x93\x99", ur"""\U0001F4D9"""
#     ),
#     u"BOOKS": (u"\xF0\x9F\x93\x9A", ur"""\U0001F4DA"""
#     ),
#     u"NAME BADGE": (u"\xF0\x9F\x93\x9B", ur"""\U0001F4DB"""
#     ),
#     u"SCROLL": (u"\xF0\x9F\x93\x9C", ur"""\U0001F4DC"""
#     ),
#     u"MEMO": (u"\xF0\x9F\x93\x9D", ur"""\U0001F4DD"""
#     ),
#     u"TELEPHONE RECEIVER": (u"\xF0\x9F\x93\x9E", ur"""\U0001F4DE"""
#     ),
#     u"PAGER": (u"\xF0\x9F\x93\x9F", ur"""\U0001F4DF"""
#     ),
#     u"FAX MACHINE": (u"\xF0\x9F\x93\xA0", ur"""\U0001F4E0"""
#     ),
#     u"SATELLITE ANTENNA": (u"\xF0\x9F\x93\xA1", ur"""\U0001F4E1"""
#     ),
#     u"PUBLIC ADDRESS LOUDSPEAKER": (u"\xF0\x9F\x93\xA2", ur"""\U0001F4E2"""
#     ),
#     u"CHEERING MEGAPHONE": (u"\xF0\x9F\x93\xA3", ur"""\U0001F4E3"""
#     ),
#     u"OUTBOX TRAY": (u"\xF0\x9F\x93\xA4", ur"""\U0001F4E4"""
#     ),
#     u"INBOX TRAY": (u"\xF0\x9F\x93\xA5", ur"""\U0001F4E5"""
#     ),
#     u"PACKAGE": (u"\xF0\x9F\x93\xA6", ur"""\U0001F4E6"""
#     ),
#     u"E-MAIL SYMBOL": (u"\xF0\x9F\x93\xA7", ur"""\U0001F4E7"""
#     ),
#     u"INCOMING ENVELOPE": (u"\xF0\x9F\x93\xA8", ur"""\U0001F4E8"""
#     ),
#     u"ENVELOPE WITH DOWNWARDS ARROW ABOVE": (u"\xF0\x9F\x93\xA9", ur"""\U0001F4E9"""
#     ),
#     u"CLOSED MAILBOX WITH LOWERED FLAG": (u"\xF0\x9F\x93\xAA", ur"""\U0001F4EA"""
#     ),
#     u"CLOSED MAILBOX WITH RAISED FLAG": (u"\xF0\x9F\x93\xAB", ur"""\U0001F4EB"""
#     ),
#     u"POSTBOX": (u"\xF0\x9F\x93\xAE", ur"""\U0001F4EE"""
#     ),
#     u"NEWSPAPER": (u"\xF0\x9F\x93\xB0", ur"""\U0001F4F0"""
#     ),
#     u"MOBILE PHONE": (u"\xF0\x9F\x93\xB1", ur"""\U0001F4F1"""
#     ),
#     u"MOBILE PHONE WITH RIGHTWARDS ARROW AT LEFT": (u"\xF0\x9F\x93\xB2", ur"""\U0001F4F2"""
#     ),
#     u"VIBRATION MODE": (u"\xF0\x9F\x93\xB3", ur"""\U0001F4F3"""
#     ),
#     u"MOBILE PHONE OFF": (u"\xF0\x9F\x93\xB4", ur"""\U0001F4F4"""
#     ),
#     u"ANTENNA WITH BARS": (u"\xF0\x9F\x93\xB6", ur"""\U0001F4F6"""
#     ),
#     u"CAMERA": (u"\xF0\x9F\x93\xB7", ur"""\U0001F4F7"""
#     ),
#     u"VIDEO CAMERA": (u"\xF0\x9F\x93\xB9", ur"""\U0001F4F9"""
#     ),
#     u"TELEVISION": (u"\xF0\x9F\x93\xBA", ur"""\U0001F4FA"""
#     ),
#     u"RADIO": (u"\xF0\x9F\x93\xBB", ur"""\U0001F4FB"""
#     ),
#     u"VIDEOCASSETTE": (u"\xF0\x9F\x93\xBC", ur"""\U0001F4FC"""
#     ),
#     u"CLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x83", ur"""\U0001F503"""
#     ),
#     u"SPEAKER WITH THREE SOUND WAVES": (u"\xF0\x9F\x94\x8A", ur"""\U0001F50A"""
#     ),
#     u"BATTERY": (u"\xF0\x9F\x94\x8B", ur"""\U0001F50B"""
#     ),
#     u"ELECTRIC PLUG": (u"\xF0\x9F\x94\x8C", ur"""\U0001F50C"""
#     ),
#     u"LEFT-POINTING MAGNIFYING GLASS": (u"\xF0\x9F\x94\x8D", ur"""\U0001F50D"""
#     ),
#     u"RIGHT-POINTING MAGNIFYING GLASS": (u"\xF0\x9F\x94\x8E", ur"""\U0001F50E"""
#     ),
#     u"LOCK WITH INK PEN": (u"\xF0\x9F\x94\x8F", ur"""\U0001F50F"""
#     ),
#     u"CLOSED LOCK WITH KEY": (u"\xF0\x9F\x94\x90", ur"""\U0001F510"""
#     ),
#     u"KEY": (u"\xF0\x9F\x94\x91", ur"""\U0001F511"""
#     ),
#     u"LOCK": (u"\xF0\x9F\x94\x92", ur"""\U0001F512"""
#     ),
#     u"OPEN LOCK": (u"\xF0\x9F\x94\x93", ur"""\U0001F513"""
#     ),
#     u"BELL": (u"\xF0\x9F\x94\x94", ur"""\U0001F514"""
#     ),
#     u"BOOKMARK": (u"\xF0\x9F\x94\x96", ur"""\U0001F516"""
#     ),
#     u"LINK SYMBOL": (u"\xF0\x9F\x94\x97", ur"""\U0001F517"""
#     ),
#     u"RADIO BUTTON": (u"\xF0\x9F\x94\x98", ur"""\U0001F518"""
#     ),
#     u"BACK WITH LEFTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x99", ur"""\U0001F519"""
#     ),
#     u"END WITH LEFTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9A", ur"""\U0001F51A"""
#     ),
#     u"ON WITH EXCLAMATION MARK WITH LEFT RIGHT ARROW ABOVE": (u"\xF0\x9F\x94\x9B", ur"""\U0001F51B"""
#     ),
#     u"SOON WITH RIGHTWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9C", ur"""\U0001F51C"""
#     ),
#     u"TOP WITH UPWARDS ARROW ABOVE": (u"\xF0\x9F\x94\x9D", ur"""\U0001F51D"""
#     ),
#     u"NO ONE UNDER EIGHTEEN SYMBOL": (u"\xF0\x9F\x94\x9E", ur"""\U0001F51E"""
#     ),
#     u"KEYCAP TEN": (u"\xF0\x9F\x94\x9F", ur"""\U0001F51F"""
#     ),
#     u"INPUT SYMBOL FOR LATIN CAPITAL LETTERS": (u"\xF0\x9F\x94\xA0", ur"""\U0001F520"""
#     ),
#     u"INPUT SYMBOL FOR LATIN SMALL LETTERS": (u"\xF0\x9F\x94\xA1", ur"""\U0001F521"""
#     ),
#     u"INPUT SYMBOL FOR NUMBERS": (u"\xF0\x9F\x94\xA2", ur"""\U0001F522"""
#     ),
#     u"INPUT SYMBOL FOR SYMBOLS": (u"\xF0\x9F\x94\xA3", ur"""\U0001F523"""
#     ),
#     u"INPUT SYMBOL FOR LATIN LETTERS": (u"\xF0\x9F\x94\xA4", ur"""\U0001F524"""
#     ),
#     u"FIRE": (u"\xF0\x9F\x94\xA5", ur"""\U0001F525"""
#     ),
#     u"ELECTRIC TORCH": (u"\xF0\x9F\x94\xA6", ur"""\U0001F526"""
#     ),
#     u"WRENCH": (u"\xF0\x9F\x94\xA7", ur"""\U0001F527"""
#     ),
#     u"HAMMER": (u"\xF0\x9F\x94\xA8", ur"""\U0001F528"""
#     ),
#     u"NUT AND BOLT": (u"\xF0\x9F\x94\xA9", ur"""\U0001F529"""
#     ),
#     u"HOCHO": (u"\xF0\x9F\x94\xAA", ur"""\U0001F52A"""
#     ),
#     u"PISTOL": (u"\xF0\x9F\x94\xAB", ur"""\U0001F52B"""
#     ),
#     u"CRYSTAL BALL": (u"\xF0\x9F\x94\xAE", ur"""\U0001F52E"""
#     ),
#     u"SIX POINTED STAR WITH MIDDLE DOT": (u"\xF0\x9F\x94\xAF", ur"""\U0001F52F"""
#     ),
#     u"JAPANESE SYMBOL FOR BEGINNER": (u"\xF0\x9F\x94\xB0", ur"""\U0001F530"""
#     ),
#     u"TRIDENT EMBLEM": (u"\xF0\x9F\x94\xB1", ur"""\U0001F531"""
#     ),
#     u"BLACK SQUARE BUTTON": (u"\xF0\x9F\x94\xB2", ur"""\U0001F532"""
#     ),
#     u"WHITE SQUARE BUTTON": (u"\xF0\x9F\x94\xB3", ur"""\U0001F533"""
#     ),
#     u"LARGE RED CIRCLE": (u"\xF0\x9F\x94\xB4", ur"""\U0001F534"""
#     ),
#     u"LARGE BLUE CIRCLE": (u"\xF0\x9F\x94\xB5", ur"""\U0001F535"""
#     ),
#     u"LARGE ORANGE DIAMOND": (u"\xF0\x9F\x94\xB6", ur"""\U0001F536"""
#     ),
#     u"LARGE BLUE DIAMOND": (u"\xF0\x9F\x94\xB7", ur"""\U0001F537"""
#     ),
#     u"SMALL ORANGE DIAMOND": (u"\xF0\x9F\x94\xB8", ur"""\U0001F538"""
#     ),
#     u"SMALL BLUE DIAMOND": (u"\xF0\x9F\x94\xB9", ur"""\U0001F539"""
#     ),
#     u"UP-POINTING RED TRIANGLE": (u"\xF0\x9F\x94\xBA", ur"""\U0001F53A"""
#     ),
#     u"DOWN-POINTING RED TRIANGLE": (u"\xF0\x9F\x94\xBB", ur"""\U0001F53B"""
#     ),
#     u"UP-POINTING SMALL RED TRIANGLE": (u"\xF0\x9F\x94\xBC", ur"""\U0001F53C"""
#     ),
#     u"DOWN-POINTING SMALL RED TRIANGLE": (u"\xF0\x9F\x94\xBD", ur"""\U0001F53D"""
#     ),
#     u"CLOCK FACE ONE OCLOCK": (u"\xF0\x9F\x95\x90", ur"""\U0001F550"""
#     ),
#     u"CLOCK FACE TWO OCLOCK": (u"\xF0\x9F\x95\x91", ur"""\U0001F551"""
#     ),
#     u"CLOCK FACE THREE OCLOCK": (u"\xF0\x9F\x95\x92", ur"""\U0001F552"""
#     ),
#     u"CLOCK FACE FOUR OCLOCK": (u"\xF0\x9F\x95\x93", ur"""\U0001F553"""
#     ),
#     u"CLOCK FACE FIVE OCLOCK": (u"\xF0\x9F\x95\x94", ur"""\U0001F554"""
#     ),
#     u"CLOCK FACE SIX OCLOCK": (u"\xF0\x9F\x95\x95", ur"""\U0001F555"""
#     ),
#     u"CLOCK FACE SEVEN OCLOCK": (u"\xF0\x9F\x95\x96", ur"""\U0001F556"""
#     ),
#     u"CLOCK FACE EIGHT OCLOCK": (u"\xF0\x9F\x95\x97", ur"""\U0001F557"""
#     ),
#     u"CLOCK FACE NINE OCLOCK": (u"\xF0\x9F\x95\x98", ur"""\U0001F558"""
#     ),
#     u"CLOCK FACE TEN OCLOCK": (u"\xF0\x9F\x95\x99", ur"""\U0001F559"""
#     ),
#     u"CLOCK FACE ELEVEN OCLOCK": (u"\xF0\x9F\x95\x9A", ur"""\U0001F55A"""
#     ),
#     u"CLOCK FACE TWELVE OCLOCK": (u"\xF0\x9F\x95\x9B", ur"""\U0001F55B"""
#     ),
#     u"MOUNT FUJI": (u"\xF0\x9F\x97\xBB", ur"""\U0001F5FB"""
#     ),
#     u"TOKYO TOWER": (u"\xF0\x9F\x97\xBC", ur"""\U0001F5FC"""
#     ),
#     u"STATUE OF LIBERTY": (u"\xF0\x9F\x97\xBD", ur"""\U0001F5FD"""
#     ),
#     u"SILHOUETTE OF JAPAN": (u"\xF0\x9F\x97\xBE", ur"""\U0001F5FE"""
#     ),
#     u"MOYAI": (u"\xF0\x9F\x97\xBF", ur"""\U0001F5FF"""
#     ),
#     u"GRINNING FACE": (u"\xF0\x9F\x98\x80", ur"""\U0001F600"""
#     ),
#     u"SMILING FACE WITH HALO": (u"\xF0\x9F\x98\x87", ur"""\U0001F607"""
#     ),
#     u"SMILING FACE WITH HORNS": (u"\xF0\x9F\x98\x88", ur"""\U0001F608"""
#     ),
#     u"SMILING FACE WITH SUNGLASSES": (u"\xF0\x9F\x98\x8E", ur"""\U0001F60E"""
#     ),
#     u"NEUTRAL FACE": (u"\xF0\x9F\x98\x90", ur"""\U0001F610"""
#     ),
#     u"EXPRESSIONLESS FACE": (u"\xF0\x9F\x98\x91", ur"""\U0001F611"""
#     ),
#     u"CONFUSED FACE": (u"\xF0\x9F\x98\x95", ur"""\U0001F615"""
#     ),
#     u"KISSING FACE": (u"\xF0\x9F\x98\x97", ur"""\U0001F617"""
#     ),
#     u"KISSING FACE WITH SMILING EYES": (u"\xF0\x9F\x98\x99", ur"""\U0001F619"""
#     ),
#     u"FACE WITH STUCK-OUT TONGUE": (u"\xF0\x9F\x98\x9B", ur"""\U0001F61B"""
#     ),
#     u"WORRIED FACE": (u"\xF0\x9F\x98\x9F", ur"""\U0001F61F"""
#     ),
#     u"FROWNING FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xA6", ur"""\U0001F626"""
#     ),
#     u"ANGUISHED FACE": (u"\xF0\x9F\x98\xA7", ur"""\U0001F627"""
#     ),
#     u"GRIMACING FACE": (u"\xF0\x9F\x98\xAC", ur"""\U0001F62C"""
#     ),
#     u"FACE WITH OPEN MOUTH": (u"\xF0\x9F\x98\xAE", ur"""\U0001F62E"""
#     ),
#     u"HUSHED FACE": (u"\xF0\x9F\x98\xAF", ur"""\U0001F62F"""
#     ),
#     u"SLEEPING FACE": (u"\xF0\x9F\x98\xB4", ur"""\U0001F634"""
#     ),
#     u"FACE WITHOUT MOUTH": (u"\xF0\x9F\x98\xB6", ur"""\U0001F636"""
#     ),
#     u"HELICOPTER": (u"\xF0\x9F\x9A\x81", ur"""\U0001F681"""
#     ),
#     u"STEAM LOCOMOTIVE": (u"\xF0\x9F\x9A\x82", ur"""\U0001F682"""
#     ),
#     u"TRAIN": (u"\xF0\x9F\x9A\x86", ur"""\U0001F686"""
#     ),
#     u"LIGHT RAIL": (u"\xF0\x9F\x9A\x88", ur"""\U0001F688"""
#     ),
#     u"TRAM": (u"\xF0\x9F\x9A\x8A", ur"""\U0001F68A"""
#     ),
#     u"ONCOMING BUS": (u"\xF0\x9F\x9A\x8D", ur"""\U0001F68D"""
#     ),
#     u"TROLLEYBUS": (u"\xF0\x9F\x9A\x8E", ur"""\U0001F68E"""
#     ),
#     u"MINIBUS": (u"\xF0\x9F\x9A\x90", ur"""\U0001F690"""
#     ),
#     u"ONCOMING POLICE CAR": (u"\xF0\x9F\x9A\x94", ur"""\U0001F694"""
#     ),
#     u"ONCOMING TAXI": (u"\xF0\x9F\x9A\x96", ur"""\U0001F696"""
#     ),
#     u"ONCOMING AUTOMOBILE": (u"\xF0\x9F\x9A\x98", ur"""\U0001F698"""
#     ),
#     u"ARTICULATED LORRY": (u"\xF0\x9F\x9A\x9B", ur"""\U0001F69B"""
#     ),
#     u"TRACTOR": (u"\xF0\x9F\x9A\x9C", ur"""\U0001F69C"""
#     ),
#     u"MONORAIL": (u"\xF0\x9F\x9A\x9D", ur"""\U0001F69D"""
#     ),
#     u"MOUNTAIN RAILWAY": (u"\xF0\x9F\x9A\x9E", ur"""\U0001F69E"""
#     ),
#     u"SUSPENSION RAILWAY": (u"\xF0\x9F\x9A\x9F", ur"""\U0001F69F"""
#     ),
#     u"MOUNTAIN CABLEWAY": (u"\xF0\x9F\x9A\xA0", ur"""\U0001F6A0"""
#     ),
#     u"AERIAL TRAMWAY": (u"\xF0\x9F\x9A\xA1", ur"""\U0001F6A1"""
#     ),
#     u"ROWBOAT": (u"\xF0\x9F\x9A\xA3", ur"""\U0001F6A3"""
#     ),
#     u"VERTICAL TRAFFIC LIGHT": (u"\xF0\x9F\x9A\xA6", ur"""\U0001F6A6"""
#     ),
#     u"PUT LITTER IN ITS PLACE SYMBOL": (u"\xF0\x9F\x9A\xAE", ur"""\U0001F6AE"""
#     ),
#     u"DO NOT LITTER SYMBOL": (u"\xF0\x9F\x9A\xAF", ur"""\U0001F6AF"""
#     ),
#     u"POTABLE WATER SYMBOL": (u"\xF0\x9F\x9A\xB0", ur"""\U0001F6B0"""
#     ),
#     u"NON-POTABLE WATER SYMBOL": (u"\xF0\x9F\x9A\xB1", ur"""\U0001F6B1"""
#     ),
#     u"NO BICYCLES": (u"\xF0\x9F\x9A\xB3", ur"""\U0001F6B3"""
#     ),
#     u"BICYCLIST": (u"\xF0\x9F\x9A\xB4", ur"""\U0001F6B4"""
#     ),
#     u"MOUNTAIN BICYCLIST": (u"\xF0\x9F\x9A\xB5", ur"""\U0001F6B5"""
#     ),
#     u"NO PEDESTRIANS": (u"\xF0\x9F\x9A\xB7", ur"""\U0001F6B7"""
#     ),
#     u"CHILDREN CROSSING": (u"\xF0\x9F\x9A\xB8", ur"""\U0001F6B8"""
#     ),
#     u"SHOWER": (u"\xF0\x9F\x9A\xBF", ur"""\U0001F6BF"""
#     ),
#     u"BATHTUB": (u"\xF0\x9F\x9B\x81", ur"""\U0001F6C1"""
#     ),
#     u"PASSPORT CONTROL": (u"\xF0\x9F\x9B\x82", ur"""\U0001F6C2"""
#     ),
#     u"CUSTOMS": (u"\xF0\x9F\x9B\x83", ur"""\U0001F6C3"""
#     ),
#     u"BAGGAGE CLAIM": (u"\xF0\x9F\x9B\x84", ur"""\U0001F6C4"""
#     ),
#     u"LEFT LUGGAGE": (u"\xF0\x9F\x9B\x85", ur"""\U0001F6C5"""
#     ),
#     u"EARTH GLOBE EUROPE-AFRICA": (u"\xF0\x9F\x8C\x8D", ur"""\U0001F30D"""
#     ),
#     u"EARTH GLOBE AMERICAS": (u"\xF0\x9F\x8C\x8E", ur"""\U0001F30E"""
#     ),
#     u"GLOBE WITH MERIDIANS": (u"\xF0\x9F\x8C\x90", ur"""\U0001F310"""
#     ),
#     u"WAXING CRESCENT MOON SYMBOL": (u"\xF0\x9F\x8C\x92", ur"""\U0001F312"""
#     ),
#     u"WANING GIBBOUS MOON SYMBOL": (u"\xF0\x9F\x8C\x96", ur"""\U0001F316"""
#     ),
#     u"LAST QUARTER MOON SYMBOL": (u"\xF0\x9F\x8C\x97", ur"""\U0001F317"""
#     ),
#     u"WANING CRESCENT MOON SYMBOL": (u"\xF0\x9F\x8C\x98", ur"""\U0001F318"""
#     ),
#     u"NEW MOON WITH FACE": (u"\xF0\x9F\x8C\x9A", ur"""\U0001F31A"""
#     ),
#     u"LAST QUARTER MOON WITH FACE": (u"\xF0\x9F\x8C\x9C", ur"""\U0001F31C"""
#     ),
#     u"FULL MOON WITH FACE": (u"\xF0\x9F\x8C\x9D", ur"""\U0001F31D"""
#     ),
#     u"SUN WITH FACE": (u"\xF0\x9F\x8C\x9E", ur"""\U0001F31E"""
#     ),
#     u"EVERGREEN TREE": (u"\xF0\x9F\x8C\xB2", ur"""\U0001F332"""
#     ),
#     u"DECIDUOUS TREE": (u"\xF0\x9F\x8C\xB3", ur"""\U0001F333"""
#     ),
#     u"LEMON": (u"\xF0\x9F\x8D\x8B", ur"""\U0001F34B"""
#     ),
#     u"PEAR": (u"\xF0\x9F\x8D\x90", ur"""\U0001F350"""
#     ),
#     u"BABY BOTTLE": (u"\xF0\x9F\x8D\xBC", ur"""\U0001F37C"""
#     ),
#     u"HORSE RACING": (u"\xF0\x9F\x8F\x87", ur"""\U0001F3C7"""
#     ),
#     u"RUGBY FOOTBALL": (u"\xF0\x9F\x8F\x89", ur"""\U0001F3C9"""
#     ),
#     u"EUROPEAN POST OFFICE": (u"\xF0\x9F\x8F\xA4", ur"""\U0001F3E4"""
#     ),
#     u"RAT": (u"\xF0\x9F\x90\x80", ur"""\U0001F400"""
#     ),
#     u"MOUSE": (u"\xF0\x9F\x90\x81", ur"""\U0001F401"""
#     ),
#     u"OX": (u"\xF0\x9F\x90\x82", ur"""\U0001F402"""
#     ),
#     u"WATER BUFFALO": (u"\xF0\x9F\x90\x83", ur"""\U0001F403"""
#     ),
#     u"COW": (u"\xF0\x9F\x90\x84", ur"""\U0001F404"""
#     ),
#     u"TIGER": (u"\xF0\x9F\x90\x85", ur"""\U0001F405"""
#     ),
#     u"LEOPARD": (u"\xF0\x9F\x90\x86", ur"""\U0001F406"""
#     ),
#     u"RABBIT": (u"\xF0\x9F\x90\x87", ur"""\U0001F407"""
#     ),
#     u"CAT": (u"\xF0\x9F\x90\x88", ur"""\U0001F408"""
#     ),
#     u"DRAGON": (u"\xF0\x9F\x90\x89", ur"""\U0001F409"""
#     ),
#     u"CROCODILE": (u"\xF0\x9F\x90\x8A", ur"""\U0001F40A"""
#     ),
#     u"WHALE": (u"\xF0\x9F\x90\x8B", ur"""\U0001F40B"""
#     ),
#     u"RAM": (u"\xF0\x9F\x90\x8F", ur"""\U0001F40F"""
#     ),
#     u"GOAT": (u"\xF0\x9F\x90\x90", ur"""\U0001F410"""
#     ),
#     u"ROOSTER": (u"\xF0\x9F\x90\x93", ur"""\U0001F413"""
#     ),
#     u"DOG": (u"\xF0\x9F\x90\x95", ur"""\U0001F415"""
#     ),
#     u"PIG": (u"\xF0\x9F\x90\x96", ur"""\U0001F416"""
#     ),
#     u"DROMEDARY CAMEL": (u"\xF0\x9F\x90\xAA", ur"""\U0001F42A"""
#     ),
#     u"BUSTS IN SILHOUETTE": (u"\xF0\x9F\x91\xA5", ur"""\U0001F465"""
#     ),
#     u"TWO MEN HOLDING HANDS": (u"\xF0\x9F\x91\xAC", ur"""\U0001F46C"""
#     ),
#     u"TWO WOMEN HOLDING HANDS": (u"\xF0\x9F\x91\xAD", ur"""\U0001F46D"""
#     ),
#     u"THOUGHT BALLOON": (u"\xF0\x9F\x92\xAD", ur"""\U0001F4AD"""
#     ),
#     u"BANKNOTE WITH EURO SIGN": (u"\xF0\x9F\x92\xB6", ur"""\U0001F4B6"""
#     ),
#     u"BANKNOTE WITH POUND SIGN": (u"\xF0\x9F\x92\xB7", ur"""\U0001F4B7"""
#     ),
#     u"OPEN MAILBOX WITH RAISED FLAG": (u"\xF0\x9F\x93\xAC", ur"""\U0001F4EC"""
#     ),
#     u"OPEN MAILBOX WITH LOWERED FLAG": (u"\xF0\x9F\x93\xAD", ur"""\U0001F4ED"""
#     ),
#     u"POSTAL HORN": (u"\xF0\x9F\x93\xAF", ur"""\U0001F4EF"""
#     ),
#     u"NO MOBILE PHONES": (u"\xF0\x9F\x93\xB5", ur"""\U0001F4F5"""
#     ),
#     u"TWISTED RIGHTWARDS ARROWS": (u"\xF0\x9F\x94\x80", ur"""\U0001F500"""
#     ),
#     u"CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x81", ur"""\U0001F501"""
#     ),
#     u"CLOCKWISE RIGHTWARDS AND LEFTWARDS OPEN CIRCLE ARROWS WITH CIRCLED ONE OVERLAY": (u"\xF0\x9F\x94\x82", ur"""\U0001F502"""
#     ),
#     u"ANTICLOCKWISE DOWNWARDS AND UPWARDS OPEN CIRCLE ARROWS": (u"\xF0\x9F\x94\x84", ur"""\U0001F504"""
#     ),
#     u"LOW BRIGHTNESS SYMBOL": (u"\xF0\x9F\x94\x85", ur"""\U0001F505"""
#     ),
#     u"HIGH BRIGHTNESS SYMBOL": (u"\xF0\x9F\x94\x86", ur"""\U0001F506"""
#     ),
#     u"SPEAKER WITH CANCELLATION STROKE": (u"\xF0\x9F\x94\x87", ur"""\U0001F507"""
#     ),
#     u"SPEAKER WITH ONE SOUND WAVE": (u"\xF0\x9F\x94\x89", ur"""\U0001F509"""
#     ),
#     u"BELL WITH CANCELLATION STROKE": (u"\xF0\x9F\x94\x95", ur"""\U0001F515"""
#     ),
#     u"MICROSCOPE": (u"\xF0\x9F\x94\xAC", ur"""\U0001F52C"""
#     ),
#     u"TELESCOPE": (u"\xF0\x9F\x94\xAD", ur"""\U0001F52D"""
#     ),
#     u"CLOCK FACE ONE-THIRTY": (u"\xF0\x9F\x95\x9C", ur"""\U0001F55C"""
#     ),
#     u"CLOCK FACE TWO-THIRTY": (u"\xF0\x9F\x95\x9D", ur"""\U0001F55D"""
#     ),
#     u"CLOCK FACE THREE-THIRTY": (u"\xF0\x9F\x95\x9E", ur"""\U0001F55E"""
#     ),
#     u"CLOCK FACE FOUR-THIRTY": (u"\xF0\x9F\x95\x9F", ur"""\U0001F55F"""
#     ),
#     u"CLOCK FACE FIVE-THIRTY": (u"\xF0\x9F\x95\xA0", ur"""\U0001F560"""
#     ),
#     u"CLOCK FACE SIX-THIRTY": (u"\xF0\x9F\x95\xA1", ur"""\U0001F561"""
#     ),
#     u"CLOCK FACE SEVEN-THIRTY": (u"\xF0\x9F\x95\xA2", ur"""\U0001F562"""
#     ),
#     u"CLOCK FACE EIGHT-THIRTY": (u"\xF0\x9F\x95\xA3", ur"""\U0001F563"""
#     ),
#     u"CLOCK FACE NINE-THIRTY": (u"\xF0\x9F\x95\xA4", ur"""\U0001F564"""
#     ),
#     u"CLOCK FACE TEN-THIRTY": (u"\xF0\x9F\x95\xA5", ur"""\U0001F565"""
#     ),
#     u"CLOCK FACE ELEVEN-THIRTY": (u"\xF0\x9F\x95\xA6", ur"""\U0001F566"""
#     ),
#     u"CLOCK FACE TWELVE-THIRTY": (u"\xF0\x9F\x95\xA7", ur"""\U0001F567"""
#     ),

}
