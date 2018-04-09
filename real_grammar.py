import nltk
import numpy as np
import six
import pdb


gram="""ROOT -> S
S -> SP VP
S -> SP VP ADJP
S -> SP VP OP
S -> SP VP NP NP
S -> CP S | S CP

SP -> NP | INF
OP -> NP | INF
CP -> ADVP | PP

NP -> DT NP | ADJP NP | NP INF | NP PP
NP -> PRP | CD | VBG | NN | NNS | NNP | NNPS

VP -> MD VP | V VP
VP -> V

ADJP -> ADVP ADJP | ADJP ADJP
ADJP -> JJ | JJR | JJS | PRPR | VBG | VBN 

ADVP -> RB | RBR | RBS

PP -> IN NP

INF -> INF1 INF2
INF1 -> TO | TO 'be' | TO 'have' | TO 'have' 'been'
INF2 -> ADVP INF2 | INF2 NP
INF2 -> VB | VBN | VBG

V -> ADVP V
V -> VB | VBD | VBG | VBN | VBP | VBZ

CC -> 'CC'
CD -> 'CD'
DT -> 'DT'
EX -> 'EX'
FW -> 'FW'
IN -> 'IN'
JJ -> 'JJ'
JJR -> 'JJR'
JJS -> 'JJS'
LS -> 'LS'
MD -> 'MD'
NN -> 'NN'
NNS -> 'NNS'
NNP -> 'NNP'
NNPS -> 'NNPS'
PDT -> 'PDT'
POS -> 'POS'
PRP -> 'PRP'
PRPR -> 'PRPR'
RB -> 'RB'
RBR -> 'RBR'
RBS -> 'RBS'
RP -> 'RP'
SYM -> 'SYM'
UH -> 'UH'
VB -> 'VB'
VBD -> 'VBD'
VBG -> 'VBG'
VBN -> 'VBN'
VBP -> 'VBP'
VBZ -> 'VBZ'
WDT -> 'WDT'
WP -> 'WP'
WPR -> 'WPR'
WRB -> 'WRB'

VB -> 'be'
VBD -> 'was' | 'were'
VBG -> 'being'
VBN -> 'been'
VBP -> 'am' | 'are'
VPZ -> 'is'

VB -> 'have'
VBD -> 'had'
VBG -> 'having'
VBN -> 'had'
VBP -> 'have'
VPZ -> 'has'

VB -> 'do'
VBD -> 'did'
VBG -> 'doing'
VBN -> 'done'
VBP -> 'do'
VPZ -> 'does'

RPR -> 'i'

PRPR -> 'her'

DT -> 'the'
IN -> 'of'
CC -> 'and'
TO -> 'to'
DT -> 'a'
IN -> 'in'
DT -> 'that'
PRP -> 'he'
IN -> 'for'
PRP -> 'it'
IN -> 'with'
IN -> 'as'
PRPR -> 'his'
IN -> 'on'
IN -> 'at'
IN -> 'by'
DT -> 'this'
RB -> 'not'
CC -> 'but'
IN -> 'from'
CC -> 'or'
DT -> 'an'
PRP -> 'they'
WDT -> 'which'
PRP -> 'you'
PRP -> 'her'
DT -> 'all'
PRP -> 'she'
EX -> 'there'
MD -> 'would'
PRPR -> 'their'
PRP -> 'we'
PRP -> 'him'
WRB -> 'when'
WP -> 'who'
MD -> 'will'
RBR -> 'more'
IN -> 'if'
DT -> 'no'
RB -> 'so'
WP -> 'what'
RP -> 'up'
PRPR -> 'its'
IN -> 'about'
IN -> 'into'
IN -> 'than'
PRP -> 'them'
MD -> 'can'
RB -> 'only'
JJ -> 'other'
JJ -> 'new'
DT -> 'some'
MD -> 'could'
DT -> 'these'
MD -> 'may'
RB -> 'then'
RB -> 'first'
DT -> 'any'
PRPR -> 'my'
RB -> 'now'
JJ -> 'such'
IN -> 'like'
PRPR -> 'our'
IN -> 'over'
PRP -> 'me'
RB -> 'even'
RBS -> 'most'
RB -> 'also'
RB -> 'after'
JJ -> 'many'
IN -> 'before'
MD -> 'must'
IN -> 'through'
JJ -> 'back'
RB -> 'much'
WRB -> 'where'
PRPR -> 'your'
RB -> 'well'
RB -> 'down'
MD -> 'should'
IN -> 'because'
DT -> 'each'
RB -> 'just'
DT -> 'those'
RB -> 'too'
WRB -> 'how'
JJ -> 'little'
JJ -> 'good'
JJ -> 'very'
RB -> 'still'
JJ -> 'own'
RB -> 'long'
RB -> 'here'
IN -> 'between'
DT -> 'both'
IN -> 'under'
RB -> 'never'
JJ -> 'same'
DT -> 'another'
IN -> 'while'
JJ -> 'last'
PRP -> 'us'
MD -> 'might'
JJ -> 'great'
JJ -> 'old'
RB -> 'off'
IN -> 'since'
IN -> 'against'
RB -> 'right'
PRP -> 'himself'
JJ -> 'few'
IN -> 'during'
IN -> 'without'
RB -> 'again'
IN -> 'around'
RB -> 'however'
JJ -> 'small'
RB -> 'once'
JJ -> 'general'
IN -> 'upon'
DT -> 'every'
JJ -> 'united'
JJ -> 'left'
IN -> 'until'
RB -> 'always'
RB -> 'away'
IN -> 'though'
JJR -> 'less'
RB -> 'almost'
RB -> 'enough'
RB -> 'far'
CC -> 'yet'
RB -> 'better'
WRB -> 'why'
JJ -> 'asked'
JJ -> 'later'
JJ -> 'next'
IN -> 'toward'
JJ -> 'young'
JJ -> 'social'
JJ -> 'several'
JJ -> 'present'
JJ -> 'national'
JJ -> 'possible'
RB -> 'rather'
JJ -> 'second'
IN -> 'per'
IN -> 'among'
JJ -> 'important'
RB -> 'often'
JJ -> 'early'
JJ -> 'white'
JJ -> 'large'
JJ -> 'big'
IN -> 'within'
IN -> 'along'
JJS -> 'best'
RB -> 'ever'
JJS -> 'least'
JJ -> 'light'
IN -> 'although'
JJ -> 'open'
JJ -> 'certain'
RB -> 'thus'
JJ -> 'different'
JJ -> 'whole'
RB -> 'perhaps'
PRP -> 'itself'
JJ -> 'human'
IN -> 'above'
JJ -> 'local'
IN -> 'whether'
CC -> 'either'
IN -> 'across'
RB -> 'quite'
RB -> 'really'
RB -> 'already'
PRP -> 'themselves'
MD -> 'shall'
RB -> 'together'
RB -> 'sure'
RB -> 'probably'
JJ -> 'free'
IN -> 'behind'
JJ -> 'cannot'
JJ -> 'political'
WPR -> 'whose'
JJ -> 'special'
JJ -> 'major'
JJ -> 'federal'
RB -> 'ago'
JJ -> 'available'
JJ -> 'known'
JJ -> 'economic'
JJ -> 'south'
JJ -> 'individual'
RB -> 'west'
RB -> 'close'
JJ -> 'true'
JJ -> 'full'
JJ -> 'future'
JJ -> 'common'
JJ -> 'necessary'
RB -> 'sometimes'
JJ -> 'front'
JJ -> 'clear'
JJ -> 'further'
JJ -> 'able'
JJ -> 'short'
JJ -> 'military'
JJ -> 'total'
IN -> 'outside'
RB -> 'usually'
RB -> 'north'
RB -> 'therefore'
JJ -> 'sound'
JJ -> 'top'
JJ -> 'black'
JJ -> 'strong'
JJ -> 'hard'
JJ -> 'various'
RB -> 'soon'
JJ -> 'mean'
IN -> 'near'
JJ -> 'modern'
JJ -> 'red'
JJ -> 'personal'
CC -> 'nor'
JJ -> 'alone'
JJ -> 'english'
JJR -> 'longer'
JJ -> 'private'
RB -> 'finally'
JJ -> 'third'
JJR -> 'greater'
JJ -> 'needed'
JJ -> 'dark'
JJ -> 'east'
JJ -> 'complete'
IN -> 'except'
JJ -> 'recent'
JJ -> 'late'
JJ -> 'particular'
RB -> 'live'
RB -> 'else'
JJ -> 'brown'
IN -> 'beyond'
JJ -> 'inside'
RB -> 'heart'
JJ -> 'low'
RB -> 'instead'
JJ -> 'single'
JJ -> 'basic'
JJ -> 'cold'
RB -> 'simply'
JJ -> 'tried'
RB -> 'actually'
JJ -> 'religious'
JJ -> 'central'
JJ -> 'received'
RB -> 'indeed'
JJ -> 'medical'
RB -> 'especially'
JJ -> 'difficult'
JJ -> 'subject'
JJ -> 'fine'
JJR -> 'higher'
JJ -> 'simple'
JJ -> 'foreign'
JJ -> 'similar'
JJ -> 'natural'
JJ -> 'final'
JJ -> 'international'
RB -> 'suddenly'
JJ -> 'likely'
JJ -> 'entire'
RB -> 'earlier'
RB -> 'particularly'
WP -> 'whom'
IN -> 'below'
RB -> 'yes'
JJ -> 'christian'
JJ -> 'blue'
RB -> 'square'
RB -> 'certainly'
JJ -> 'ready'
JJ -> 'industrial'
JJ -> 'due'
JJ -> 'moral'
CC -> 'neither'
RB -> 'throughout'
RB -> 'directly'
RB -> 'nearly'
JJ -> 'french'
JJ -> 'western'
JJ -> 'southern'
JJ -> 'normal'
RB -> 'merely'
JJ -> 'concerned'
RB -> 'maybe'
JJ -> 'continued'
RB -> 'generally'
JJ -> 'former'
JJ -> 'average'
JJ -> 'hot'
JJ -> 'wrong'
JJ -> 'direct'
JJ -> 'effective'
JJ -> 'soviet'
PRP -> 'myself'
RB -> 'clearly'
JJ -> 'beautiful'
RB -> 'somewhat'
PRP -> 'herself'
RB -> 'apparently'
JJ -> 'wide'
JJ -> 'easy'
JJR -> 'larger'
RB -> 'recently'
JJR -> 'lower'
RB -> 'immediately'
IN -> 'de'
UH -> 'oh'
JJ -> 'daily'
JJ -> 'additional'
JJ -> 'technical'
JJ -> 'fiscal'
JJ -> 'main'
JJ -> 'chief'
IN -> 'aj'
JJ -> 'middle'
JJ -> 'british'
JJ -> 'green'
JJ -> 'serious'
JJ -> 'nuclear'
RB -> 'forward'
JJ -> 'specific'
RB -> 'slowly'
RB -> 'obviously'
JJ -> 'straight'
JJ -> 'born'
JJ -> 'poor'
WDT -> 'whatever'
JJ -> 'heavy'
RB -> 'completely'
RB -> 'ahead'
JJ -> 'deep'
JJ -> 'democratic'
JJ -> 'established'
JJ -> 'pretty'
RB -> 'easily'
RB -> 'negro'
RB -> 'hardly'
JJ -> 'limited'
JJ -> 'professional'
JJ -> 'interested'
IN -> 'despite'
JJ -> 'covered'
JJ -> 'original'
RB -> 'exactly'
JJ -> 'related'
IN -> 'unless'
JJ -> 'actual'
JJ -> 'popular'
JJ -> 'happy'
JJ -> 'communist'
JJ -> 'christ'
JJ -> 'considerable'
JJ -> 'primary'
JJ -> 'usual'
JJ -> 'successful'
JJ -> 'proper'
JJ -> 'worth'
RB -> 'highly'
JJR -> 'older'
JJ -> 'annual'
JJ -> 'principal'
JJ -> 'obvious'
JJ -> 'thin'
RB -> 'entirely'
JJ -> 'objective'
RB -> 'frequently'
JJ -> 'civil'
JJ -> 'equal'
JJ -> 'famous'
RB -> 'quickly'
RB -> 'moreover'
JJS -> 'greatest'
JJ -> 'active'
JJ -> 'key'
RB -> 'carefully'
JJ -> 'bright'
JJ -> 'finished'
JJ -> 'mary'
JJ -> 'financial'
JJ -> 'significant'
JJ -> 'previous'
JJ -> 'allowed'
JJ -> 'scientific'
RB -> 'otherwise'
JJ -> 'musical'
JJ -> 'german'
RB -> 'relatively'
JJ -> 'marked'
JJ -> 'broad'
JJ -> 'impossible'
JJ -> 'aware'
JJ -> 'strange'
JJ -> 'catholic'
JJ -> 'regular'
RB -> 'slightly'
JJ -> 'remembered'
JJ -> 'interesting'
JJ -> 'fresh'
JJ -> 'germany'
JJ -> 'immediate'
JJ -> 'essential'
JJ -> 'forced'
RB -> 'fully'
JJ -> 'russian'
JJ -> 'gray'
JJ -> 'maximum'
JJ -> 'separate'
JJ -> 'literary'
IN -> 'beside'
JJ -> 'traditional'
JJ -> 'fair'
JJ -> 'secret'
JJ -> 'fast'
JJR -> 'smaller'
JJ -> 'vocational'
RB -> 'solid'
JJ -> 'formed'
JJ -> 'quiet'
JJ -> 'nice'
JJ -> 'junior'
JJ -> 'rich'
JJ -> 'fourth'
JJ -> 'positive'
JJ -> 'jewish'
JJ -> 'pointed'
RB -> 'twice'
JJ -> 'interior'
RB -> 'nevertheless'
JJ -> 'brief'
JJ -> 'legal'
RB -> 'somehow'
"""

gram="""    ROOT -> S
    S -> SP VP
    S -> SP VP ADJP
    S -> SP VP OP
    S -> SP VP NP NP
    S -> CP S
    S -> S CP
    SP -> NP
    SP -> INF
    OP -> NP
    OP -> INF
    CP -> ADVP
    CP -> PP
    NP -> DT NP
    NP -> ADJP NP
    NP -> NP INF
    NP -> NP PP
    NP -> PRP
    NP -> CD
    NP -> VBG
    NP -> NN
    NP -> NNS
    NP -> NNP
    NP -> NNPS
    VP -> MD VP
    VP -> V VP
    VP -> V
    ADJP -> ADVP ADJP
    ADJP -> ADJP ADJP
    ADJP -> JJ
    ADJP -> JJR
    ADJP -> JJS
    ADJP -> PRPR
    ADJP -> VBG
    ADJP -> VBN
    ADVP -> RB
    ADVP -> RBR
    ADVP -> RBS
    PP -> IN NP
    INF -> INF1 INF2
    INF1 -> TO
    INF1 -> TO 'be'
    INF1 -> TO 'have'
    INF1 -> TO 'have' 'been'
    INF2 -> ADVP INF2
    INF2 -> INF2 NP
    INF2 -> VB
    INF2 -> VBN
    INF2 -> VBG
    V -> ADVP V
    V -> VB
    V -> VBD
    V -> VBG
    V -> VBN
    V -> VBP
    V -> VBZ
    CC -> 'CC'
    CD -> 'CD'
    DT -> 'DT'
    EX -> 'EX'
    FW -> 'FW'
    IN -> 'IN'
    JJ -> 'JJ'
    JJR -> 'JJR'
    JJS -> 'JJS'
    LS -> 'LS'
    MD -> 'MD'
    NN -> 'NN'
    NNS -> 'NNS'
    NNP -> 'NNP'
    NNPS -> 'NNPS'
    PDT -> 'PDT'
    POS -> 'POS'
    PRP -> 'PRP'
    PRPR -> 'PRPR'
    RB -> 'RB'
    RBR -> 'RBR'
    RBS -> 'RBS'
    RP -> 'RP'
    SYM -> 'SYM'
    UH -> 'UH'
    VB -> 'VB'
    VBD -> 'VBD'
    VBG -> 'VBG'
    VBN -> 'VBN'
    VBP -> 'VBP'
    VBZ -> 'VBZ'
    WDT -> 'WDT'
    WP -> 'WP'
    WPR -> 'WPR'
    WRB -> 'WRB'
    VB -> 'be'
    VBD -> 'was'
    VBD -> 'were'
    VBG -> 'being'
    VBN -> 'been'
    VBP -> 'am'
    VBP -> 'are'
    VPZ -> 'is'
    VB -> 'have'
    VBD -> 'had'
    VBG -> 'having'
    VBN -> 'had'
    VBP -> 'have'
    VPZ -> 'has'
    VB -> 'do'
    VBD -> 'did'
    VBG -> 'doing'
    VBN -> 'done'
    VBP -> 'do'
    VPZ -> 'does'
    RPR -> 'i'
    PRPR -> 'her'
    DT -> 'the'
    IN -> 'of'
    CC -> 'and'
    TO -> 'to'
    DT -> 'a'
    IN -> 'in'
    DT -> 'that'
    PRP -> 'he'
    IN -> 'for'
    PRP -> 'it'
    IN -> 'with'
    IN -> 'as'
    PRPR -> 'his'
    IN -> 'on'
    IN -> 'at'
    IN -> 'by'
    DT -> 'this'
    RB -> 'not'
    CC -> 'but'
    IN -> 'from'
    CC -> 'or'
    DT -> 'an'
    PRP -> 'they'
    WDT -> 'which'
    PRP -> 'you'
    PRP -> 'her'
    DT -> 'all'
    PRP -> 'she'
    EX -> 'there'
    MD -> 'would'
    PRPR -> 'their'
    PRP -> 'we'
    PRP -> 'him'
    WRB -> 'when'
    WP -> 'who'
    MD -> 'will'
    RBR -> 'more'
    IN -> 'if'
    DT -> 'no'
    RB -> 'so'
    WP -> 'what'
    RP -> 'up'
    PRPR -> 'its'
    IN -> 'about'
    IN -> 'into'
    IN -> 'than'
    PRP -> 'them'
    MD -> 'can'
    RB -> 'only'
    JJ -> 'other'
    JJ -> 'new'
    DT -> 'some'
    MD -> 'could'
    DT -> 'these'
    MD -> 'may'
    RB -> 'then'
    RB -> 'first'
    DT -> 'any'
    PRPR -> 'my'
    RB -> 'now'
    JJ -> 'such'
    IN -> 'like'
    PRPR -> 'our'
    IN -> 'over'
    PRP -> 'me'
    RB -> 'even'
    RBS -> 'most'
    RB -> 'also'
    RB -> 'after'
    JJ -> 'many'
    IN -> 'before'
    MD -> 'must'
    IN -> 'through'
    JJ -> 'back'
    RB -> 'much'
    WRB -> 'where'
    PRPR -> 'your'
    RB -> 'well'
    RB -> 'down'
    MD -> 'should'
    IN -> 'because'
    DT -> 'each'
    RB -> 'just'
    DT -> 'those'
    RB -> 'too'
    WRB -> 'how'
    JJ -> 'little'
    JJ -> 'good'
    JJ -> 'very'
    RB -> 'still'
    JJ -> 'own'
    RB -> 'long'
    RB -> 'here'
    IN -> 'between'
    DT -> 'both'
    IN -> 'under'
    RB -> 'never'
    JJ -> 'same'
    DT -> 'another'
    IN -> 'while'
    JJ -> 'last'
    PRP -> 'us'
    MD -> 'might'
    JJ -> 'great'
    JJ -> 'old'
    RB -> 'off'
    IN -> 'since'
    IN -> 'against'
    RB -> 'right'
    PRP -> 'himself'
    JJ -> 'few'
    IN -> 'during'
    IN -> 'without'
    RB -> 'again'
    IN -> 'around'
    RB -> 'however'
    JJ -> 'small'
    RB -> 'once'
    JJ -> 'general'
    IN -> 'upon'
    DT -> 'every'
    JJ -> 'united'
    JJ -> 'left'
    IN -> 'until'
    RB -> 'always'
    RB -> 'away'
    IN -> 'though'
    JJR -> 'less'
    RB -> 'almost'
    RB -> 'enough'
    RB -> 'far'
    CC -> 'yet'
    RB -> 'better'
    WRB -> 'why'
    JJ -> 'asked'
    JJ -> 'later'
    JJ -> 'next'
    IN -> 'toward'
    JJ -> 'young'
    JJ -> 'social'
    JJ -> 'several'
    JJ -> 'present'
    JJ -> 'national'
    JJ -> 'possible'
    RB -> 'rather'
    JJ -> 'second'
    IN -> 'per'
    IN -> 'among'
    JJ -> 'important'
    RB -> 'often'
    JJ -> 'early'
    JJ -> 'white'
    JJ -> 'large'
    JJ -> 'big'
    IN -> 'within'
    IN -> 'along'
    JJS -> 'best'
    RB -> 'ever'
    JJS -> 'least'
    JJ -> 'light'
    IN -> 'although'
    JJ -> 'open'
    JJ -> 'certain'
    RB -> 'thus'
    JJ -> 'different'
    JJ -> 'whole'
    RB -> 'perhaps'
    PRP -> 'itself'
    JJ -> 'human'
    IN -> 'above'
    JJ -> 'local'
    IN -> 'whether'
    CC -> 'either'
    IN -> 'across'
    RB -> 'quite'
    RB -> 'really'
    RB -> 'already'
    PRP -> 'themselves'
    MD -> 'shall'
    RB -> 'together'
    RB -> 'sure'
    RB -> 'probably'
    JJ -> 'free'
    IN -> 'behind'
    JJ -> 'cannot'
    JJ -> 'political'
    WPR -> 'whose'
    JJ -> 'special'
    JJ -> 'major'
    JJ -> 'federal'
    RB -> 'ago'
    JJ -> 'available'
    JJ -> 'known'
    JJ -> 'economic'
    JJ -> 'south'
    JJ -> 'individual'
    RB -> 'west'
    RB -> 'close'
    JJ -> 'true'
    JJ -> 'full'
    JJ -> 'future'
    JJ -> 'common'
    JJ -> 'necessary'
    RB -> 'sometimes'
    JJ -> 'front'
    JJ -> 'clear'
    JJ -> 'further'
    JJ -> 'able'
    JJ -> 'short'
    JJ -> 'military'
    JJ -> 'total'
    IN -> 'outside'
    RB -> 'usually'
    RB -> 'north'
    RB -> 'therefore'
    JJ -> 'sound'
    JJ -> 'top'
    JJ -> 'black'
    JJ -> 'strong'
    JJ -> 'hard'
    JJ -> 'various'
    RB -> 'soon'
    JJ -> 'mean'
    IN -> 'near'
    JJ -> 'modern'
    JJ -> 'red'
    JJ -> 'personal'
    CC -> 'nor'
    JJ -> 'alone'
    JJ -> 'english'
    JJR -> 'longer'
    JJ -> 'private'
    RB -> 'finally'
    JJ -> 'third'
    JJR -> 'greater'
    JJ -> 'needed'
    JJ -> 'dark'
    JJ -> 'east'
    JJ -> 'complete'
    IN -> 'except'
    JJ -> 'recent'
    JJ -> 'late'
    JJ -> 'particular'
    RB -> 'live'
    RB -> 'else'
    JJ -> 'brown'
    IN -> 'beyond'
    JJ -> 'inside'
    RB -> 'heart'
    JJ -> 'low'
    RB -> 'instead'
    JJ -> 'single'
    JJ -> 'basic'
    JJ -> 'cold'
    RB -> 'simply'
    JJ -> 'tried'
    RB -> 'actually'
    JJ -> 'religious'
    JJ -> 'central'
    JJ -> 'received'
    RB -> 'indeed'
    JJ -> 'medical'
    RB -> 'especially'
    JJ -> 'difficult'
    JJ -> 'subject'
    JJ -> 'fine'
    JJR -> 'higher'
    JJ -> 'simple'
    JJ -> 'foreign'
    JJ -> 'similar'
    JJ -> 'natural'
    JJ -> 'final'
    JJ -> 'international'
    RB -> 'suddenly'
    JJ -> 'likely'
    JJ -> 'entire'
    RB -> 'earlier'
    RB -> 'particularly'
    WP -> 'whom'
    IN -> 'below'
    RB -> 'yes'
    JJ -> 'christian'
    JJ -> 'blue'
    RB -> 'square'
    RB -> 'certainly'
    JJ -> 'ready'
    JJ -> 'industrial'
    JJ -> 'due'
    JJ -> 'moral'
    CC -> 'neither'
    RB -> 'throughout'
    RB -> 'directly'
    RB -> 'nearly'
    JJ -> 'french'
    JJ -> 'western'
    JJ -> 'southern'
    JJ -> 'normal'
    RB -> 'merely'
    JJ -> 'concerned'
    RB -> 'maybe'
    JJ -> 'continued'
    RB -> 'generally'
    JJ -> 'former'
    JJ -> 'average'
    JJ -> 'hot'
    JJ -> 'wrong'
    JJ -> 'direct'
    JJ -> 'effective'
    JJ -> 'soviet'
    PRP -> 'myself'
    RB -> 'clearly'
    JJ -> 'beautiful'
    RB -> 'somewhat'
    PRP -> 'herself'
    RB -> 'apparently'
    JJ -> 'wide'
    JJ -> 'easy'
    JJR -> 'larger'
    RB -> 'recently'
    JJR -> 'lower'
    RB -> 'immediately'
    IN -> 'de'
    UH -> 'oh'
    JJ -> 'daily'
    JJ -> 'additional'
    JJ -> 'technical'
    JJ -> 'fiscal'
    JJ -> 'main'
    JJ -> 'chief'
    IN -> 'aj'
    JJ -> 'middle'
    JJ -> 'british'
    JJ -> 'green'
    JJ -> 'serious'
    JJ -> 'nuclear'
    RB -> 'forward'
    JJ -> 'specific'
    RB -> 'slowly'
    RB -> 'obviously'
    JJ -> 'straight'
    JJ -> 'born'
    JJ -> 'poor'
    WDT -> 'whatever'
    JJ -> 'heavy'
    RB -> 'completely'
    RB -> 'ahead'
    JJ -> 'deep'
    JJ -> 'democratic'
    JJ -> 'established'
    JJ -> 'pretty'
    RB -> 'easily'
    RB -> 'negro'
    RB -> 'hardly'
    JJ -> 'limited'
    JJ -> 'professional'
    JJ -> 'interested'
    IN -> 'despite'
    JJ -> 'covered'
    JJ -> 'original'
    RB -> 'exactly'
    JJ -> 'related'
    IN -> 'unless'
    JJ -> 'actual'
    JJ -> 'popular'
    JJ -> 'happy'
    JJ -> 'communist'
    JJ -> 'christ'
    JJ -> 'considerable'
    JJ -> 'primary'
    JJ -> 'usual'
    JJ -> 'successful'
    JJ -> 'proper'
    JJ -> 'worth'
    RB -> 'highly'
    JJR -> 'older'
    JJ -> 'annual'
    JJ -> 'principal'
    JJ -> 'obvious'
    JJ -> 'thin'
    RB -> 'entirely'
    JJ -> 'objective'
    RB -> 'frequently'
    JJ -> 'civil'
    JJ -> 'equal'
    JJ -> 'famous'
    RB -> 'quickly'
    RB -> 'moreover'
    JJS -> 'greatest'
    JJ -> 'active'
    JJ -> 'key'
    RB -> 'carefully'
    JJ -> 'bright'
    JJ -> 'finished'
    JJ -> 'mary'
    JJ -> 'financial'
    JJ -> 'significant'
    JJ -> 'previous'
    JJ -> 'allowed'
    JJ -> 'scientific'
    RB -> 'otherwise'
    JJ -> 'musical'
    JJ -> 'german'
    RB -> 'relatively'
    JJ -> 'marked'
    JJ -> 'broad'
    JJ -> 'impossible'
    JJ -> 'aware'
    JJ -> 'strange'
    JJ -> 'catholic'
    JJ -> 'regular'
    RB -> 'slightly'
    JJ -> 'remembered'
    JJ -> 'interesting'
    JJ -> 'fresh'
    JJ -> 'germany'
    JJ -> 'immediate'
    JJ -> 'essential'
    JJ -> 'forced'
    RB -> 'fully'
    JJ -> 'russian'
    JJ -> 'gray'
    JJ -> 'maximum'
    JJ -> 'separate'
    JJ -> 'literary'
    IN -> 'beside'
    JJ -> 'traditional'
    JJ -> 'fair'
    JJ -> 'secret'
    JJ -> 'fast'
    JJR -> 'smaller'
    JJ -> 'vocational'
    RB -> 'solid'
    JJ -> 'formed'
    JJ -> 'quiet'
    JJ -> 'nice'
    JJ -> 'junior'
    JJ -> 'rich'
    JJ -> 'fourth'
    JJ -> 'positive'
    JJ -> 'jewish'
    JJ -> 'pointed'
    RB -> 'twice'
    JJ -> 'interior'
    RB -> 'nevertheless'
    JJ -> 'brief'
    JJ -> 'legal'
    RB -> 'somehow'
    """
# form the CFG and get the start symbol
GCFG = nltk.CFG.fromstring(gram)
start_index = GCFG.productions()[0].lhs()

print(GCFG)
print(start_index)

# collect all lhs symbols, and the unique set of them
all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(GCFG.productions())

print(all_lhs)
print(lhs_list)
print(D)


# this map tells us the rhs symbol indices for each production rule
rhs_map = [None]*D
count = 0
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b,six.string_types):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list),D))
count = 0

print(rhs_map)


# this tells us for each lhs symbol which productions rules should be masked
for sym in lhs_list:
    is_in = np.array([a == sym for a in all_lhs], dtype=int).reshape(1,-1)
    masks[count] = is_in
    count = count + 1

print(masks)

# this tells us the indices where the masks are equal to 1
index_array = []
for i in range(masks.shape[1]):
    index_array.append(np.where(masks[:,i]==1)[0][0])
ind_of_ind = np.array(index_array)

max_rhs = max([len(l) for l in rhs_map])

