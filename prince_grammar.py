import nltk
import numpy as np
import six
import pdb

# the toy grammar
gram =     """ S   -> NP VP
        NP  -> Det N | Det Adj N
        VP  -> V NP | V NP PP
        PP  -> P Det Pla
        Det -> 'a' | 'the' | 'my' | 'your'
        NP  -> 'bob' | 'kevin' | 'kyle' 
        N   -> 'man' | 'dog' | 'cat' | 'chicken' | 'bird' | 'pig' | 'lion' | 'bear'
        N   -> 'turkey' | 'wolf' | 'rabbit' | 'duck' | 'monkey'
        V   -> 'saw' | "killed" | 'caught' | 'chased' | 'played'
        P   -> 'in' | 'by' 
        Pla ->  'park' | 'school' | 'forest'
        Adj  -> 'angry' | 'frightened' |  'little' | 'wild' | 'big'
        """

gram = """ROOT -> FRAG
ROOT -> NP
ROOT -> S
ROOT -> SINV
ADJP -> ADJP CC ADJP
ADJP -> ADJP SBAR
ADJP -> ADVP JJ
ADJP -> ADVP VBN
ADJP -> DT JJ CC JJ
ADJP -> DT JJ JJ
ADJP -> DT JJR
ADJP -> JJ
ADJP -> JJ CC JJ
ADJP -> JJ CC JJ CC JJ
ADJP -> JJ JJ
ADJP -> JJ NN
ADJP -> JJ PP
ADJP -> JJ RB S
ADJP -> JJ RB SBAR
ADJP -> JJ S
ADJP -> JJR
ADJP -> JJR SBAR
ADJP -> NN CC JJ
ADJP -> NN JJ
ADJP -> NN PP
ADJP -> NP JJR
ADJP -> RB JJ
ADJP -> RB JJ CC JJ
ADJP -> RB JJ PP
ADJP -> RB JJ S
ADJP -> RB JJ SBAR
ADJP -> RB JJR
ADJP -> RB NN
ADJP -> RB RB
ADJP -> RB RB JJR
ADJP -> RB RBR JJ
ADJP -> RB SBAR
ADJP -> RB VBN
ADJP -> RB VBN PP
ADJP -> RBR JJ
ADJP -> RBR S
ADJP -> RBR VBN
ADJP -> RBS JJ
ADJP -> VBN NP
ADJP -> VBN PP
ADJP -> WHADVP JJ
ADVP -> ADVP CC ADVP
ADVP -> ADVP SBAR
ADVP -> CC
ADVP -> DT CC
ADVP -> DT IN RB
ADVP -> DT NN
ADVP -> DT RB
ADVP -> DT RBR
ADVP -> IN DT
ADVP -> IN JJS
ADVP -> IN NN
ADVP -> IN PP
ADVP -> IN RB
ADVP -> JJ
ADVP -> JJ CC JJ
ADVP -> NN
ADVP -> NP IN
ADVP -> NP RB
ADVP -> PRP
ADVP -> RB
ADVP -> RB CC RB
ADVP -> RB JJ
ADVP -> RB NP
ADVP -> RB PP
ADVP -> RB RB
ADVP -> RB RB RB
ADVP -> RB RBR
ADVP -> RBR
ADVP -> RBR CC RBR
ADVP -> RBR RB
ADVP -> RP CC RP
ADVP -> WRB
CC -> 'and'
CC -> 'but'
CC -> 'nor'
CC -> 'not'
CC -> 'or'
CD -> 'CD'
CONJP -> CC RB
CONJP -> RB IN
DT -> 'a'
DT -> 'all'
DT -> 'an'
DT -> 'another'
DT -> 'any'
DT -> 'both'
DT -> 'each'
DT -> 'either'
DT -> 'every'
DT -> 'many'
DT -> 'neither'
DT -> 'no'
DT -> 'some'
DT -> 'that'
DT -> 'the'
DT -> 'these'
DT -> 'this'
DT -> 'those'
EX -> 'there'
FRAG -> ADJP
FRAG -> ADJP S
FRAG -> ADJP SBAR
FRAG -> ADVP NP
FRAG -> CC PP
FRAG -> CC RB ADVP PP
FRAG -> CC SBAR
FRAG -> IN SBAR
FRAG -> NP
FRAG -> RB PP
FW -> 'FW'
IN -> 'IN'
IN -> 'about'
IN -> 'above'
IN -> 'across'
IN -> 'after'
IN -> 'against'
IN -> 'along'
IN -> 'among'
IN -> 'around'
IN -> 'as'
IN -> 'at'
IN -> 'because'
IN -> 'before'
IN -> 'behind'
IN -> 'between'
IN -> 'beyond'
IN -> 'by'
IN -> 'down'
IN -> 'during'
IN -> 'except'
IN -> 'for'
IN -> 'from'
IN -> 'if'
IN -> 'in'
IN -> 'inside'
IN -> 'into'
IN -> 'like'
IN -> 'near'
IN -> 'of'
IN -> 'off'
IN -> 'on'
IN -> 'out'
IN -> 'outside'
IN -> 'over'
IN -> 'since'
IN -> 'so'
IN -> 'than'
IN -> 'that'
IN -> 'though'
IN -> 'through'
IN -> 'toward'
IN -> 'under'
IN -> 'until'
IN -> 'up'
IN -> 'upon'
IN -> 'whether'
IN -> 'while'
IN -> 'with'
IN -> 'within'
IN -> 'without'
JJ -> 'JJ'
JJ -> 'able'
JJ -> 'back'
JJ -> 'bad'
JJ -> 'beautiful'
JJ -> 'big'
JJ -> 'black'
JJ -> 'blue'
JJ -> 'brown'
JJ -> 'certain'
JJ -> 'chief'
JJ -> 'close'
JJ -> 'cold'
JJ -> 'common'
JJ -> 'dark'
JJ -> 'dead'
JJ -> 'deep'
JJ -> 'different'
JJ -> 'difficult'
JJ -> 'due'
JJ -> 'early'
JJ -> 'east'
JJ -> 'easy'
JJ -> 'else'
JJ -> 'enough'
JJ -> 'far'
JJ -> 'federal'
JJ -> 'few'
JJ -> 'final'
JJ -> 'first'
JJ -> 'forward'
JJ -> 'free'
JJ -> 'front'
JJ -> 'full'
JJ -> 'good'
JJ -> 'great'
JJ -> 'green'
JJ -> 'hard'
JJ -> 'heavy'
JJ -> 'high'
JJ -> 'hot'
JJ -> 'human'
JJ -> 'important'
JJ -> 'large'
JJ -> 'last'
JJ -> 'like'
JJ -> 'little'
JJ -> 'live'
JJ -> 'local'
JJ -> 'long'
JJ -> 'many'
JJ -> 'mean'
JJ -> 'moral'
JJ -> 'much'
JJ -> 'natural'
JJ -> 'necessary'
JJ -> 'new'
JJ -> 'next'
JJ -> 'old'
JJ -> 'only'
JJ -> 'open'
JJ -> 'other'
JJ -> 'own'
JJ -> 'personal'
JJ -> 'poor'
JJ -> 'possible'
JJ -> 'present'
JJ -> 'ready'
JJ -> 'real'
JJ -> 'red'
JJ -> 'religious'
JJ -> 'right'
JJ -> 'same'
JJ -> 'second'
JJ -> 'several'
JJ -> 'short'
JJ -> 'simple'
JJ -> 'small'
JJ -> 'special'
JJ -> 'straight'
JJ -> 'such'
JJ -> 'sure'
JJ -> 'technical'
JJ -> 'third'
JJ -> 'true'
JJ -> 'very'
JJ -> 'western'
JJ -> 'white'
JJ -> 'whole'
JJ -> 'wrong'
JJ -> 'young'
JJR -> 'JJR'
JJR -> 'better'
JJR -> 'greater'
JJR -> 'higher'
JJR -> 'larger'
JJR -> 'more'
JJS -> 'JJS'
JJS -> 'best'
JJS -> 'least'
JJS -> 'most'
MD -> 'can'
MD -> 'could'
MD -> 'may'
MD -> 'might'
MD -> 'must'
MD -> 'should'
MD -> 'will'
MD -> 'would'
NN -> 'NN'
NN -> 'account'
NN -> 'action'
NN -> 'activity'
NN -> 'air'
NN -> 'answer'
NN -> 'anyone'
NN -> 'anything'
NN -> 'area'
NN -> 'army'
NN -> 'art'
NN -> 'attention'
NN -> 'audience'
NN -> 'back'
NN -> 'ball'
NN -> 'bed'
NN -> 'bill'
NN -> 'body'
NN -> 'boy'
NN -> 'business'
NN -> 'call'
NN -> 'car'
NN -> 'care'
NN -> 'case'
NN -> 'cause'
NN -> 'center'
NN -> 'century'
NN -> 'chance'
NN -> 'charge'
NN -> 'chief'
NN -> 'child'
NN -> 'church'
NN -> 'city'
NN -> 'close'
NN -> 'club'
NN -> 'color'
NN -> 'company'
NN -> 'congress'
NN -> 'corner'
NN -> 'cost'
NN -> 'country'
NN -> 'course'
NN -> 'day'
NN -> 'dead'
NN -> 'deal'
NN -> 'death'
NN -> 'decision'
NN -> 'direction'
NN -> 'distance'
NN -> 'door'
NN -> 'doubt'
NN -> 'earth'
NN -> 'effort'
NN -> 'end'
NN -> 'english'
NN -> 'evening'
NN -> 'everything'
NN -> 'existence'
NN -> 'experience'
NN -> 'eye'
NN -> 'face'
NN -> 'fact'
NN -> 'faith'
NN -> 'family'
NN -> 'father'
NN -> 'fear'
NN -> 'feeling'
NN -> 'field'
NN -> 'figure'
NN -> 'fire'
NN -> 'firm'
NN -> 'floor'
NN -> 'food'
NN -> 'freedom'
NN -> 'friend'
NN -> 'front'
NN -> 'game'
NN -> 'george'
NN -> 'girl'
NN -> 'god'
NN -> 'government'
NN -> 'ground'
NN -> 'group'
NN -> 'gun'
NN -> 'hair'
NN -> 'half'
NN -> 'hall'
NN -> 'hand'
NN -> 'head'
NN -> 'heart'
NN -> 'home'
NN -> 'hope'
NN -> 'horse'
NN -> 'hour'
NN -> 'house'
NN -> 'husband'
NN -> 'i'
NN -> 'idea'
NN -> 'inside'
NN -> 'interest'
NN -> 'island'
NN -> 'job'
NN -> 'john'
NN -> 'kind'
NN -> 'labor'
NN -> 'lack'
NN -> 'land'
NN -> 'letter'
NN -> 'level'
NN -> 'life'
NN -> 'light'
NN -> 'line'
NN -> 'list'
NN -> 'living'
NN -> 'look'
NN -> 'lot'
NN -> 'man'
NN -> 'manner'
NN -> 'material'
NN -> 'matter'
NN -> 'meaning'
NN -> 'meeting'
NN -> 'member'
NN -> 'mind'
NN -> 'moment'
NN -> 'money'
NN -> 'morning'
NN -> 'mother'
NN -> 'move'
NN -> 'movement'
NN -> 'music'
NN -> 'name'
NN -> 'nature'
NN -> 'need'
NN -> 'night'
NN -> 'none'
NN -> 'note'
NN -> 'nothing'
NN -> 'number'
NN -> 'office'
NN -> 'one'
NN -> 'opportunity'
NN -> 'order'
NN -> 'out'
NN -> 'outside'
NN -> 'paper'
NN -> 'part'
NN -> 'pattern'
NN -> 'performance'
NN -> 'person'
NN -> 'piece'
NN -> 'place'
NN -> 'plan'
NN -> 'play'
NN -> 'point'
NN -> 'position'
NN -> 'power'
NN -> 'president'
NN -> 'press'
NN -> 'problem'
NN -> 'program'
NN -> 'purpose'
NN -> 'quality'
NN -> 'question'
NN -> 'reaction'
NN -> 'reason'
NN -> 'record'
NN -> 'religion'
NN -> 'rest'
NN -> 'right'
NN -> 'river'
NN -> 'road'
NN -> 'room'
NN -> 'school'
NN -> 'secretary'
NN -> 'section'
NN -> 'sense'
NN -> 'service'
NN -> 'side'
NN -> 'situation'
NN -> 'size'
NN -> 'something'
NN -> 'son'
NN -> 'sound'
NN -> 'square'
NN -> 'stage'
NN -> 'state'
NN -> 'step'
NN -> 'street'
NN -> 'student'
NN -> 'study'
NN -> 'subject'
NN -> 'summer'
NN -> 'surface'
NN -> 'table'
NN -> 'thing'
NN -> 'thinking'
NN -> 'thought'
NN -> 'time'
NN -> 'today'
NN -> 'top'
NN -> 'town'
NN -> 'trouble'
NN -> 'truth'
NN -> 'turn'
NN -> 'union'
NN -> 'view'
NN -> 'visit'
NN -> 'voice'
NN -> 'wall'
NN -> 'war'
NN -> 'water'
NN -> 'way'
NN -> 'week'
NN -> 'well'
NN -> 'west'
NN -> 'while'
NN -> 'will'
NN -> 'william'
NN -> 'window'
NN -> 'woman'
NN -> 'word'
NN -> 'work'
NN -> 'world'
NN -> 'writing'
NN -> 'york'
NNP -> 'NNP'
NNS -> 'NNS'
NNS -> 'arms'
NNS -> 'cars'
NNS -> 'children'
NNS -> 'days'
NNS -> 'efforts'
NNS -> 'eyes'
NNS -> 'feet'
NNS -> 'forces'
NNS -> 'forms'
NNS -> 'friends'
NNS -> 'girls'
NNS -> 'hands'
NNS -> 'hours'
NNS -> 'i'
NNS -> 'letters'
NNS -> 'lines'
NNS -> 'men'
NNS -> 'miles'
NNS -> 'minutes'
NNS -> 'months'
NNS -> 'ones'
NNS -> 'others'
NNS -> 'parts'
NNS -> 'people'
NNS -> 'plans'
NNS -> 'points'
NNS -> 'problems'
NNS -> 'questions'
NNS -> 'services'
NNS -> 'steps'
NNS -> 'students'
NNS -> 'things'
NNS -> 'times'
NNS -> 'ways'
NNS -> 'weeks'
NNS -> 'women'
NNS -> 'words'
NNS -> 'works'
NNS -> 'years'
NP -> ADJP JJ NN
NP -> ADJP NN
NP -> ADJP NNS
NP -> ADVP
NP -> ADVP DT NN
NP -> ADVP JJ NN
NP -> ADVP RB RB
NP -> CC NP
NP -> CC NP CC NP
NP -> CD
NP -> CD ADJP NNS
NP -> CD CC CD NNS
NP -> CD DT
NP -> CD JJ NNS
NP -> CD NN
NP -> CD NN NNS
NP -> CD NNS
NP -> CD NNS CC NNS
NP -> CD VBN NNS
NP -> DT
NP -> DT ADJP JJ NN
NP -> DT ADJP NN
NP -> DT ADJP NNS
NP -> DT CD
NP -> DT CD JJ NNS
NP -> DT CD NN
NP -> DT CD NNS
NP -> DT CD VBN NNS
NP -> DT FW
NP -> DT FW FW
NP -> DT JJ
NP -> DT JJ CC JJ NN
NP -> DT JJ CC JJ NNS
NP -> DT JJ JJ JJ JJ NN
NP -> DT JJ JJ NN
NP -> DT JJ JJ NN
NP -> DT JJ JJ NN NN
NP -> DT JJ JJ NNS
NP -> DT JJ NN
NP -> DT JJ NN NN
NP -> DT JJ NN NNS
NP -> DT JJ NNS
NP -> DT JJ NNS IN
NP -> DT JJ NNS NN
NP -> DT JJ VBG NN
NP -> DT JJR
NP -> DT JJR NN
NP -> DT JJS
NP -> DT JJS JJ NN
NP -> DT JJS NN
NP -> DT NN
NP -> DT NN CC NN
NP -> DT NN NN
NP -> DT NN NN NN
NP -> DT NN NNS
NP -> DT NN NNS CC NN NN
NP -> DT NN PRP
NP -> DT NN S
NP -> DT NNS
NP -> DT NNS NN
NP -> DT NNS PRP
NP -> DT PRP
NP -> DT QP
NP -> DT RB
NP -> DT VBG
NP -> DT VBG CC VBG
NP -> DT VBG NN
NP -> DT VBG NNS
NP -> DT VBN NN
NP -> EX
NP -> FW
NP -> FW FW NN
NP -> JJ
NP -> JJ CC JJ NNS
NP -> JJ DT NN
NP -> JJ JJ NN
NP -> JJ JJ NNS
NP -> JJ NN
NP -> JJ NN CC JJ NN NNS
NP -> JJ NN CC NN
NP -> JJ NN NN
NP -> JJ NN NNS
NP -> JJ NNS
NP -> JJ NNS CC NNS
NP -> JJ VBG NNS
NP -> JJR
NP -> JJR CC JJR
NP -> JJR NN
NP -> JJR NN NNS
NP -> JJR NNS
NP -> JJS
NP -> JJS NN
NP -> NN
NP -> NN CC NN
NP -> NN CC NN NN
NP -> NN CC NN NNS
NP -> NN CC NNS
NP -> NN CD
NP -> NN NN
NP -> NN NN CC NN
NP -> NN NNS
NP -> NN NNS CC NN NNS
NP -> NN RB
NP -> NN VBG
NP -> NNP
NP -> NNP NN
NP -> NNS
NP -> NNS CC NN
NP -> NNS CC NN NNS
NP -> NNS CC NNS
NP -> NNS S
NP -> NP ADJP
NP -> NP ADJP PP SBAR
NP -> NP ADVP
NP -> NP CC ADVP PP
NP -> NP CC NP
NP -> NP NP
NP -> NP NP
NP -> NP NP PP
NP -> NP PP
NP -> NP PP
NP -> NP PP ADVP
NP -> NP PP PP
NP -> NP PP PP PP PP
NP -> NP PP S
NP -> NP PP SBAR
NP -> NP PP UCP
NP -> NP QP
NP -> NP SBAR
NP -> NP UCP
NP -> NP VP
NP -> PDT DT JJ NN
NP -> PDT DT NN
NP -> PDT DT NNS
NP -> PDT DT NNS CC NNS
NP -> PRP
NP -> PRP CC NNS
NP -> PRP CC PRP
NP -> PRP NNS
NP -> PRPS ADJP NN
NP -> PRPS JJ
NP -> PRPS JJ ADJP NN
NP -> PRPS JJ JJ NN NNS NN
NP -> PRPS JJ JJ NNS
NP -> PRPS JJ NN
NP -> PRPS JJ NN NNS
NP -> PRPS JJ NNS
NP -> PRPS JJS NN
NP -> PRPS JJS NNS
NP -> PRPS NN
NP -> PRPS NN CC NNS
NP -> PRPS NN NN
NP -> PRPS NN NN CC NN
NP -> PRPS NN NNS
NP -> PRPS NN S
NP -> PRPS NNS
NP -> PRPS NNS CC NNS
NP -> PRPS NNS S
NP -> PRPS VBG
NP -> PRPS VBG NN
NP -> PRPS VBG NNS
NP -> QP DT NN
NP -> QP NN
NP -> QP NNS
NP -> RB
NP -> RB DT JJ NN
NP -> RB DT JJ NN
NP -> RB DT NN
NP -> RB DT NNS
NP -> RB JJ
NP -> RB JJ NN
NP -> RB NN
NP -> RB RB
NP -> RB S
NP -> RB WDT
NP -> SBAR CC NP
NP -> UCP NNS
NP -> VBG NN
NP -> VBG NNS
NP -> WP NN
NP -> WPS NN
NP -> WRB
PDT -> 'all'
PDT -> 'half'
PDT -> 'such'
PP -> ADVP IN NP
PP -> CC NP
PP -> DT IN
PP -> DT IN NP
PP -> IN
PP -> IN ADJP
PP -> IN ADVP
PP -> IN IN NP
PP -> IN NN
PP -> IN NP
PP -> IN NP
PP -> IN PP
PP -> IN S
PP -> IN SBAR
PP -> JJ IN NP
PP -> PP CC PP
PP -> RB ADVP
PP -> RB IN S
PP -> TO
PP -> TO NP
PP -> TO S
PP -> TO SBAR
PRP -> 'PRP'
PRP -> 'he'
PRP -> 'her'
PRP -> 'herself'
PRP -> 'him'
PRP -> 'himself'
PRP -> 'it'
PRP -> 'itself'
PRP -> 'me'
PRP -> 'myself'
PRP -> 'one'
PRP -> 'she'
PRP -> 'them'
PRP -> 'themselves'
PRP -> 'they'
PRP -> 'us'
PRP -> 'we'
PRP -> 'you'
PRPS -> 'her'
PRPS -> 'his'
PRPS -> 'its'
PRPS -> 'my'
PRPS -> 'our'
PRPS -> 'their'
PRPS -> 'your'
PRT -> RP
PRT -> RP CC RP
QP -> CC RB
QP -> DT NN
QP -> DT PDT
QP -> JJR IN DT
QP -> RB CD
RB -> 'RB'
RB -> 'about'
RB -> 'above'
RB -> 'again'
RB -> 'ago'
RB -> 'ahead'
RB -> 'all'
RB -> 'almost'
RB -> 'alone'
RB -> 'already'
RB -> 'also'
RB -> 'always'
RB -> 'around'
RB -> 'as'
RB -> 'away'
RB -> 'back'
RB -> 'before'
RB -> 'behind'
RB -> 'below'
RB -> 'certainly'
RB -> 'close'
RB -> 'completely'
RB -> 'daily'
RB -> 'down'
RB -> 'else'
RB -> 'enough'
RB -> 'even'
RB -> 'ever'
RB -> 'far'
RB -> 'finally'
RB -> 'first'
RB -> 'forward'
RB -> 'generally'
RB -> 'hard'
RB -> 'here'
RB -> 'high'
RB -> 'home'
RB -> 'i'
RB -> 'immediately'
RB -> 'indeed'
RB -> 'inside'
RB -> 'instead'
RB -> 'just'
RB -> 'late'
RB -> 'later'
RB -> 'little'
RB -> 'long'
RB -> 'longer'
RB -> 'maybe'
RB -> 'merely'
RB -> 'much'
RB -> 'nearly'
RB -> 'never'
RB -> 'next'
RB -> 'no'
RB -> 'not'
RB -> 'now'
RB -> 'obviously'
RB -> 'off'
RB -> 'often'
RB -> 'oh'
RB -> 'once'
RB -> 'only'
RB -> 'out'
RB -> 'outside'
RB -> 'over'
RB -> 'perhaps'
RB -> 'pretty'
RB -> 'quite'
RB -> 'rather'
RB -> 'real'
RB -> 'really'
RB -> 'right'
RB -> 'simply'
RB -> 'slowly'
RB -> 'so'
RB -> 'sometimes'
RB -> 'soon'
RB -> 'still'
RB -> 'straight'
RB -> 'suddenly'
RB -> 'that'
RB -> 'then'
RB -> 'there'
RB -> 'together'
RB -> 'too'
RB -> 'up'
RB -> 'usually'
RB -> 'very'
RB -> 'well'
RB -> 'yet'
RBR -> 'RBR'
RBR -> 'better'
RBR -> 'further'
RBR -> 'less'
RBR -> 'longer'
RBR -> 'more'
RBS -> 'most'
RP -> 'around'
RP -> 'away'
RP -> 'back'
RP -> 'down'
RP -> 'in'
RP -> 'off'
RP -> 'on'
RP -> 'out'
RP -> 'over'
RP -> 'together'
RP -> 'up'
S -> ADJP
S -> ADJP NP VP
S -> ADVP ADVP NP VP
S -> ADVP NP ADVP VP
S -> ADVP NP VP
S -> ADVP NP VP
S -> ADVP PP NP VP
S -> ADVP VP
S -> CC ADVP NP VP
S -> CC NP ADVP VP
S -> CC NP VP
S -> CC NP VP
S -> CC PP ADVP VP
S -> CC PP NP VP
S -> CC SBAR NP VP
S -> CC VP
S -> IN ADVP NP VP
S -> IN NP VP
S -> NP ADJP
S -> NP ADJP ADVP
S -> NP ADVP PP
S -> NP ADVP VP
S -> NP ADVP VP
S -> NP DT VP
S -> NP NP
S -> NP NP VP
S -> NP VP
S -> NP VP
S -> PP NP ADVP VP
S -> PP NP VP
S -> PP NP VP
S -> PP S CC S
S -> RB NP VP
S -> RB VP
S -> RB VP
S -> S CC ADVP S
S -> S CC S
S -> S CC S
S -> S CC S ADVP
S -> S CC S SBAR CC S
S -> S NP VP
S -> S VP
S -> S VP
S -> SBAR NP VP
S -> SBAR VP
S -> VP
S -> VP
SBAR -> ADVP IN S
SBAR -> IN FRAG
SBAR -> IN NN S
SBAR -> IN S
SBAR -> IN S CC RB
SBAR -> IN SBAR
SBAR -> RB IN S
SBAR -> RB S
SBAR -> S
SBAR -> SBAR CC SBAR
SBAR -> SINV
SBAR -> WHADJP S
SBAR -> WHADVP S
SBAR -> WHNP S
SBAR -> WHPP S
SBAR -> X S
SINV -> CC ADVP VP NP
SINV -> MD NP VP
SINV -> PP VP NP
TO -> 'to'
UCP -> ADJP CC ADVP
UCP -> ADJP CC NP
UCP -> ADJP CC PP
UCP -> ADJP CC VP
UCP -> DT ADVP CC PP
VB -> 'VB'
VB -> 'act'
VB -> 'answer'
VB -> 'appear'
VB -> 'ask'
VB -> 'be'
VB -> 'become'
VB -> 'believe'
VB -> 'care'
VB -> 'change'
VB -> 'come'
VB -> 'do'
VB -> 'end'
VB -> 'expect'
VB -> 'face'
VB -> 'feel'
VB -> 'find'
VB -> 'get'
VB -> 'give'
VB -> 'go'
VB -> 'have'
VB -> 'hear'
VB -> 'help'
VB -> 'hold'
VB -> 'increase'
VB -> 'keep'
VB -> 'know'
VB -> 'leave'
VB -> 'let'
VB -> 'like'
VB -> 'live'
VB -> 'look'
VB -> 'lower'
VB -> 'make'
VB -> 'mean'
VB -> 'miss'
VB -> 'move'
VB -> 'name'
VB -> 'need'
VB -> 'pay'
VB -> 'police'
VB -> 'provide'
VB -> 'put'
VB -> 'question'
VB -> 'remember'
VB -> 'report'
VB -> 'return'
VB -> 'run'
VB -> 'say'
VB -> 'see'
VB -> 'seem'
VB -> 'show'
VB -> 'sound'
VB -> 'speak'
VB -> 'stand'
VB -> 'start'
VB -> 'stay'
VB -> 'stop'
VB -> 'take'
VB -> 'tell'
VB -> 'test'
VB -> 'think'
VB -> 'try'
VB -> 'understand'
VB -> 'visit'
VB -> 'want'
VB -> 'washington'
VB -> 'work'
VBD -> 'VBD'
VBD -> 'added'
VBD -> 'appeared'
VBD -> 'asked'
VBD -> 'became'
VBD -> 'began'
VBD -> 'brought'
VBD -> 'called'
VBD -> 'came'
VBD -> 'carried'
VBD -> 'continued'
VBD -> 'cut'
VBD -> 'described'
VBD -> 'did'
VBD -> 'felt'
VBD -> 'found'
VBD -> 'gave'
VBD -> 'got'
VBD -> 'ground'
VBD -> 'had'
VBD -> 'happened'
VBD -> 'heard'
VBD -> 'held'
VBD -> 'indicated'
VBD -> 'kept'
VBD -> 'knew'
VBD -> 'lay'
VBD -> 'learned'
VBD -> 'led'
VBD -> 'left'
VBD -> 'lived'
VBD -> 'looked'
VBD -> 'made'
VBD -> 'met'
VBD -> 'moved'
VBD -> 'needed'
VBD -> 'opened'
VBD -> 'paid'
VBD -> 'passed'
VBD -> 'placed'
VBD -> 'put'
VBD -> 'ran'
VBD -> 'reached'
VBD -> 'read'
VBD -> 'required'
VBD -> 'returned'
VBD -> 'said'
VBD -> 'sat'
VBD -> 'saw'
VBD -> 'seemed'
VBD -> 'sent'
VBD -> 'served'
VBD -> 'set'
VBD -> 'started'
VBD -> 'stood'
VBD -> 'stopped'
VBD -> 'thought'
VBD -> 'told'
VBD -> 'took'
VBD -> 'tried'
VBD -> 'turned'
VBD -> 'used'
VBD -> 'walked'
VBD -> 'wanted'
VBD -> 'was'
VBD -> 'went'
VBD -> 'were'
VBD -> 'worked'
VBD -> 'wrote'
VBG -> 'VBG'
VBG -> 'beginning'
VBG -> 'being'
VBG -> 'coming'
VBG -> 'doing'
VBG -> 'feeling'
VBG -> 'getting'
VBG -> 'going'
VBG -> 'growing'
VBG -> 'having'
VBG -> 'looking'
VBG -> 'making'
VBG -> 'moving'
VBG -> 'reading'
VBG -> 'running'
VBG -> 'saying'
VBG -> 'thinking'
VBG -> 'trying'
VBG -> 'using'
VBG -> 'waiting'
VBG -> 'working'
VBN -> 'VBN'
VBN -> 'asked'
VBN -> 'been'
VBN -> 'born'
VBN -> 'brought'
VBN -> 'called'
VBN -> 'come'
VBN -> 'concerned'
VBN -> 'decided'
VBN -> 'determined'
VBN -> 'developed'
VBN -> 'done'
VBN -> 'established'
VBN -> 'expected'
VBN -> 'felt'
VBN -> 'found'
VBN -> 'gone'
VBN -> 'had'
VBN -> 'heard'
VBN -> 'involved'
VBN -> 'known'
VBN -> 'led'
VBN -> 'left'
VBN -> 'lived'
VBN -> 'lost'
VBN -> 'made'
VBN -> 'moved'
VBN -> 'placed'
VBN -> 'put'
VBN -> 'run'
VBN -> 'said'
VBN -> 'sat'
VBN -> 'seen'
VBN -> 'sent'
VBN -> 'served'
VBN -> 'shown'
VBN -> 'started'
VBN -> 'stood'
VBN -> 'stopped'
VBN -> 'taken'
VBN -> 'thought'
VBN -> 'told'
VBN -> 'used'
VBN -> 'wanted'
VBN -> 'written'
VBP -> 'VBP'
VBP -> 'are'
VBP -> 'do'
VBP -> 'face'
VBP -> 'feel'
VBP -> 'have'
VBP -> 'hear'
VBP -> 'know'
VBP -> 'let'
VBP -> 'like'
VBP -> 'mean'
VBP -> 'remember'
VBP -> 'see'
VBP -> 'use'
VBP -> 'want'
VBP -> 'wish'
VBP -> 'work'
VBZ -> 'VBZ'
VBZ -> 'has'
VBZ -> 'is'
VBZ -> 'says'
VBZ -> 'steps'
VP -> ADVP ADVP VBD NP
VP -> ADVP VB NP
VP -> ADVP VB NP SBAR
VP -> ADVP VBD ADVP
VP -> ADVP VBD ADVP PP
VP -> ADVP VBD CC VBD PP
VP -> ADVP VBD NP
VP -> ADVP VBD NP PP
VP -> ADVP VBD PP
VP -> ADVP VBN
VP -> ADVP VBN NP PP SBAR
VP -> ADVP VBN PP
VP -> ADVP VBN PP S
VP -> ADVP VBN PP SBAR
VP -> ADVP VP CC VP
VP -> JJ
VP -> MD
VP -> MD ADVP VP
VP -> MD PP
VP -> MD RB ADVP VP
VP -> MD RB VP
VP -> MD VP
VP -> NN
VP -> NN S
VP -> NN SBAR
VP -> NP ADVP
VP -> PP
VP -> PP VBD NP
VP -> RB VP IN VP
VP -> SBAR
VP -> TO
VP -> TO VP
VP -> VB
VP -> VB ADJP
VP -> VB ADJP ADVP
VP -> VB ADVP
VP -> VB ADVP NP
VP -> VB ADVP S
VP -> VB ADVP SBAR
VP -> VB CC RB VB NP
VP -> VB CC VB
VP -> VB CC VB NP
VP -> VB CC VB PP
VP -> VB CC VB PP SBAR
VP -> VB CC VB SBAR
VP -> VB NP
VP -> VB NP ADVP
VP -> VB NP ADVP PP
VP -> VB NP ADVP S
VP -> VB NP ADVP SBAR
VP -> VB NP NP
VP -> VB NP PP
VP -> VB NP PP PP
VP -> VB NP PP SBAR
VP -> VB NP PRT
VP -> VB NP PRT ADVP
VP -> VB NP PRT PP
VP -> VB NP S
VP -> VB NP SBAR
VP -> VB PP
VP -> VB PP ADVP
VP -> VB PP PP
VP -> VB PP PP NP
VP -> VB PP SBAR
VP -> VB PRT
VP -> VB PRT ADVP
VP -> VB PRT NP
VP -> VB PRT NP PP
VP -> VB PRT NP SBAR
VP -> VB PRT PP
VP -> VB PRT PP NP
VP -> VB PRT PP PP
VP -> VB PRT SBAR
VP -> VB S
VP -> VB SBAR
VP -> VB VP
VP -> VBD
VP -> VBD ADJP
VP -> VBD ADJP ADVP
VP -> VBD ADJP NP
VP -> VBD ADJP PP
VP -> VBD ADJP PP SBAR
VP -> VBD ADJP SBAR
VP -> VBD ADVP
VP -> VBD ADVP ADJP
VP -> VBD ADVP ADJP NP
VP -> VBD ADVP ADJP PP
VP -> VBD ADVP ADJP SBAR
VP -> VBD ADVP ADVP
VP -> VBD ADVP ADVP PP
VP -> VBD ADVP NP
VP -> VBD ADVP PP
VP -> VBD ADVP PP PP
VP -> VBD ADVP S
VP -> VBD ADVP SBAR
VP -> VBD ADVP VP
VP -> VBD CC VBD NP
VP -> VBD CC VBD NP PP SBAR
VP -> VBD CC VBD PP
VP -> VBD NP
VP -> VBD NP ADVP
VP -> VBD NP ADVP ADJP PP
VP -> VBD NP ADVP PP
VP -> VBD NP ADVP S
VP -> VBD NP ADVP SBAR
VP -> VBD NP NP
VP -> VBD NP NP PP
VP -> VBD NP PP
VP -> VBD NP PP PP
VP -> VBD NP PP S
VP -> VBD NP PP SBAR
VP -> VBD NP PRT
VP -> VBD NP PRT ADVP
VP -> VBD NP PRT PP
VP -> VBD NP S
VP -> VBD NP SBAR
VP -> VBD PP
VP -> VBD PP ADVP
VP -> VBD PP ADVP SBAR
VP -> VBD PP NP
VP -> VBD PP PP
VP -> VBD PP S
VP -> VBD PP SBAR
VP -> VBD PRT
VP -> VBD PRT ADVP
VP -> VBD PRT ADVP PP
VP -> VBD PRT ADVP SBAR
VP -> VBD PRT NP
VP -> VBD PRT NP NP
VP -> VBD PRT NP PP
VP -> VBD PRT PP
VP -> VBD PRT PP PP
VP -> VBD PRT PP S
VP -> VBD PRT PP SBAR
VP -> VBD PRT S
VP -> VBD PRT SBAR
VP -> VBD RB
VP -> VBD RB ADJP
VP -> VBD RB ADJP SBAR
VP -> VBD RB ADVP ADJP
VP -> VBD RB ADVP VP
VP -> VBD RB NP
VP -> VBD RB PP
VP -> VBD RB S
VP -> VBD RB VP
VP -> VBD S
VP -> VBD S PP
VP -> VBD S SBAR
VP -> VBD SBAR
VP -> VBD UCP
VP -> VBD VP
VP -> VBG
VP -> VBG ADJP
VP -> VBG ADVP
VP -> VBG ADVP ADVP
VP -> VBG ADVP NP
VP -> VBG ADVP PP
VP -> VBG ADVP S
VP -> VBG CC VBG ADVP
VP -> VBG CC VBG PP
VP -> VBG NP
VP -> VBG NP ADVP
VP -> VBG NP PP
VP -> VBG NP PRT
VP -> VBG NP S
VP -> VBG NP SBAR
VP -> VBG PP
VP -> VBG PP ADVP
VP -> VBG PP ADVP SBAR
VP -> VBG PP PP
VP -> VBG PP S
VP -> VBG PRT
VP -> VBG PRT NP
VP -> VBG PRT NP PP
VP -> VBG PRT PP
VP -> VBG S
VP -> VBG SBAR
VP -> VBG VP
VP -> VBN
VP -> VBN ADJP
VP -> VBN ADJP NP
VP -> VBN ADJP PP
VP -> VBN ADJP SBAR
VP -> VBN ADVP
VP -> VBN ADVP ADVP PP
VP -> VBN ADVP NP
VP -> VBN ADVP PP
VP -> VBN ADVP SBAR
VP -> VBN CC VBN ADVP SBAR
VP -> VBN CC VBN SBAR
VP -> VBN NP
VP -> VBN NP NP
VP -> VBN NP PP
VP -> VBN NP PP PP
VP -> VBN NP PRT PP
VP -> VBN NP S
VP -> VBN NP SBAR
VP -> VBN PP
VP -> VBN PP ADVP
VP -> VBN PP NP
VP -> VBN PP PP
VP -> VBN PP PP NP
VP -> VBN PP PP PP
VP -> VBN PP PP SBAR
VP -> VBN PP SBAR
VP -> VBN PRT
VP -> VBN PRT ADVP
VP -> VBN PRT ADVP PP
VP -> VBN PRT NP
VP -> VBN PRT PP
VP -> VBN PRT PP PP
VP -> VBN PRT SBAR
VP -> VBN S
VP -> VBN SBAR
VP -> VBN VP
VP -> VBP
VP -> VBP ADJP
VP -> VBP ADVP VP
VP -> VBP NP
VP -> VBP NP ADVP
VP -> VBP NP SBAR
VP -> VBP PP
VP -> VBP RB ADVP
VP -> VBP S
VP -> VBP SBAR
VP -> VBP VP
VP -> VBZ
VP -> VBZ ADJP
VP -> VBZ ADJP PP
VP -> VBZ ADJP S
VP -> VBZ ADJP SBAR
VP -> VBZ ADVP NP
VP -> VBZ ADVP PP
VP -> VBZ ADVP VP
VP -> VBZ NP
VP -> VBZ NP ADVP
VP -> VBZ NP PP
VP -> VBZ PP
VP -> VBZ PRT
VP -> VBZ PRT S
VP -> VBZ S
VP -> VBZ SBAR
VP -> VBZ VP
VP -> VP CC RB VP
VP -> VP CC VP
VP -> VP CC VP SBAR
VP -> VP CONJP VP
WDT -> 'that'
WDT -> 'what'
WDT -> 'which'
WHADJP -> WRB JJ
WHADVP -> WDT NN
WHADVP -> WRB
WHADVP -> WRB ADJP
WHNP -> WDT
WHNP -> WDT NN
WHNP -> WHNP NP
WHNP -> WHNP PP
WHNP -> WP
WHNP -> WP NN PP
WHPP -> IN WHNP
WHPP -> TO WHNP
WP -> 'what'
WP -> 'who'
WPS -> 'whose'
WRB -> 'WRB'
WRB -> 'how'
WRB -> 'when'
WRB -> 'where'
WRB -> 'why'
X -> DT JJR"""

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

