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

gram = """
ROOT -> FRAG
ROOT -> INTJ
ROOT -> NP
ROOT -> S
ROOT -> SBARQ
ROOT -> SQ
S -> A2 ADVP A2 NP VP
S -> A2 PP A2 NP VP
S -> ADVP A2 NP VP A3
S -> ADVP A2 S A2 NP VP A3
S -> ADVP NP A2 PP A2 NP ADVP VP A3
S -> ADVP NP VP
S -> ADVP NP VP A3
S -> ADVP VP
S -> CC A2 ADVP A2 NP VP A3
S -> CC ADVP A2 PP A2 NP VP A3
S -> CC ADVP NP VP A3
S -> CC NP NP VP
S -> CC NP VP
S -> CC NP VP A3
S -> CC PP A2 NP VP A3
S -> CC PP NP VP A3
S -> CC S A2 CC IN S A3
S -> CC S A2 CC S A3
S -> CC SBAR A2 NP ADVP VP A3
S -> CC SBAR A2 NP VP A3
S -> CC SBAR NP VP A3
S -> IN A2 SBAR A2 NP VP A3
S -> IN ADVP NP VP A3
S -> IN NP VP
S -> IN NP VP A3
S -> NP A2 ADVP A2 NP VP A3
S -> NP A2 PP A2 VP
S -> NP ADJP
S -> NP ADVP VP
S -> NP ADVP VP A3
S -> NP NP
S -> NP NP VP
S -> NP VP
S -> NP VP A3
S -> PP A2 NP VP A3
S -> PP NP VP
S -> PP NP VP A3
S -> PRN NP VP
S -> RB NP VP A3
S -> RB PP NP VP A3
S -> S A2 CC A2 PP
S -> S A2 CC S
S -> S A2 CC S A3
S -> S A2 NP VP A3
S -> S A4 A1 S A3 A1
S -> S A4 CC S A3
S -> S A4 S A3
S -> S A4 S A4 S A3
S -> S VP A3
S -> SBAR A2 NP ADVP VP A3
S -> SBAR A2 NP VP
S -> SBAR A2 NP VP A3
S -> SBAR A2 PP A2 NP VP A3
S -> SBAR A2 S CC ADVP A2 PP A3
S -> SBAR NP VP
S -> SBAR NP VP A3
S -> SBAR VP A3
S -> VP
S -> VP A3
A1 -> "'"
A2 -> ','
A3 -> '!'
A3 -> '.'
A3 -> '?'
A4 -> ':'
A4 -> ';'
A5 -> "'"
A5 -> '`'
ADJP -> ADJP PP
ADJP -> DT RB JJ
ADJP -> JJ
ADJP -> JJ PP
ADJP -> JJ S
ADJP -> NP JJ
ADJP -> RB
ADJP -> RB JJ
ADJP -> RB JJ PP
ADJP -> RB JJ S
ADJP -> RB JJR
ADJP -> RB RB JJ
ADJP -> RB RB JJ S
ADJP -> RB RB RBR JJ
ADJP -> RBR JJ
ADJP -> RBR JJ PP
ADJP -> RBS JJ
ADJP -> VBN PP
ADVP -> ADVP A2 ADVP
ADVP -> ADVP SBAR
ADVP -> DT IN RB
ADVP -> DT RB
ADVP -> IN DT
ADVP -> IN JJ
ADVP -> IN RB
ADVP -> NP RB
ADVP -> NP RBR
ADVP -> RB
ADVP -> RB DT NN
ADVP -> RB IN NN
ADVP -> RB NP
ADVP -> RB RB
ADVP -> RB RB RB
ADVP -> RB RBR
ADVP -> RBR
ADVP -> RBS
CC -> 'and'
CC -> 'but'
CC -> 'neither'
CC -> 'nor'
CC -> 'or'
CD -> '1909'
CD -> '1920'
CD -> 'four'
CD -> 'one'
CD -> 'six'
CD -> 'thousand'
CD -> 'three'
CD -> 'two'
CONJP -> CC RB
DT -> 'a'
DT -> 'all'
DT -> 'an'
DT -> 'another'
DT -> 'any'
DT -> 'each'
DT -> 'every'
DT -> 'no'
DT -> 'some'
DT -> 'that'
DT -> 'the'
DT -> 'these'
DT -> 'this'
DT -> 'those'
EX -> 'there'
FRAG -> ADJP A2 NP
FRAG -> ADVP A3
FRAG -> INTJ A2 NP A3
FW -> 'i'
IN -> 'about'
IN -> 'after'
IN -> 'against'
IN -> 'among'
IN -> 'around'
IN -> 'as'
IN -> 'at'
IN -> 'because'
IN -> 'behind'
IN -> 'between'
IN -> 'beyond'
IN -> 'by'
IN -> 'except'
IN -> 'for'
IN -> 'from'
IN -> 'if'
IN -> 'in'
IN -> 'inside'
IN -> 'into'
IN -> 'like'
IN -> 'of'
IN -> 'on'
IN -> 'once'
IN -> 'out'
IN -> 'over'
IN -> 'since'
IN -> 'so'
IN -> 'than'
IN -> 'that'
IN -> 'through'
IN -> 'under'fo
IN -> 'until'
IN -> 'upon'
IN -> 'with'
IN -> 'without'
INTJ -> INTJ A2 INTJ
INTJ -> UH
INTJ -> UH A3
JJ -> 'able'
JJ -> 'active'
JJ -> 'angry'
JJ -> 'ashamed'
JJ -> 'bad'
JJ -> 'beautiful'
JJ -> 'big'
JJ -> 'bushes'
JJ -> 'busy'
JJ -> 'certain'
JJ -> 'charming'
JJ -> 'clear'
JJ -> 'close'
JJ -> 'cold'
JJ -> 'complete'
JJ -> 'considerable'
JJ -> 'convenient'
JJ -> 'cool'
JJ -> 'coquettish'
JJ -> 'cumbersome'
JJ -> 'dangerous'
JJ -> 'deep'
JJ -> 'difficult'
JJ -> 'disarranged'
JJ -> 'distinguished'
JJ -> 'enough'
JJ -> 'entire'
JJ -> 'essential'
JJ -> 'extinct'
JJ -> 'extraordinary'
JJ -> 'familiar'
JJ -> 'few'
JJ -> 'first'
JJ -> 'full'
JJ -> 'golden'
JJ -> 'good'
JJ -> 'great'
JJ -> 'green'
JJ -> 'grief'
JJ -> 'happy'
JJ -> 'hard'
JJ -> 'huge'
JJ -> 'human'
JJ -> 'important'
JJ -> 'impressive'
JJ -> 'inconsistent'
JJ -> 'infested'
JJ -> 'inhabited'
JJ -> 'inspiring'
JJ -> 'invisible'
JJ -> 'irritated'
JJ -> 'isolated'
JJ -> 'last'
JJ -> 'lazy'
JJ -> 'little'
JJ -> 'long'
JJ -> 'lovely'
JJ -> 'magnificent'
JJ -> 'many'
JJ -> 'mental'
JJ -> 'miraculous'
JJ -> 'much'
JJ -> 'mysterious'
JJ -> 'naive'
JJ -> 'new'
JJ -> 'odd'
JJ -> 'old'
JJ -> 'only'
JJ -> 'other'
JJ -> 'overpowering'
JJ -> 'perfect'
JJ -> 'poor'
JJ -> 'possible'
JJ -> 'precious'
JJ -> 'prepared'
JJ -> 'present'
JJ -> 'primeval'
JJ -> 'proud'
JJ -> 'quiet'
JJ -> 'ready'
JJ -> 'rid'
JJ -> 'rumpled'
JJ -> 'sad'
JJ -> 'same'
JJ -> 'satisfied'
JJ -> 'second'
JJ -> 'secret'
JJ -> 'serious'
JJ -> 'shipwrecked'
JJ -> 'short'
JJ -> 'sickly'
JJ -> 'simple'
JJ -> 'single'
JJ -> 'small'
JJ -> 'successful'
JJ -> 'such'
JJ -> 'sudden'
JJ -> 'sure'
JJ -> 'surprised'
JJ -> 'tall'
JJ -> 'terrible'
JJ -> 'third'
JJ -> 'thunderstruck'
JJ -> 'tiny'
JJ -> 'true'
JJ -> 'turkish'
JJ -> 'unhappy'
JJ -> 'urgent'
JJ -> 'useful'
JJ -> 'valuable'
JJ -> 'volcanic'
JJ -> 'weak'
JJ -> 'white'
JJ -> 'whole'
JJ -> 'wild'
JJ -> 'worried'
JJ -> 'worth'
JJ -> 'wrong'
JJ -> 'young'
JJR -> 'greater'
JJR -> 'larger'
JJR -> 'more'
JJS -> 'best'
JJS -> 'earliest'
JJS -> 'greatest'
MD -> 'can'
MD -> 'could'
MD -> 'dare'
MD -> 'may'
MD -> 'might'
MD -> 'must'
MD -> 'need'
MD -> 'ought'
MD -> 'shall'
MD -> 'should'
MD -> 'will'
MD -> 'would'
NN -> 'absence'
NN -> 'accident'
NN -> 'acquaintance'
NN -> 'act'
NN -> 'adornment'
NN -> 'advantage'
NN -> 'affection'
NN -> 'age'
NN -> 'air'
NN -> 'airplane'
NN -> 'amazement'
NN -> 'animal'
NN -> 'answer'
NN -> 'anyone'
NN -> 'anything'
NN -> 'apparition'
NN -> 'appearance'
NN -> 'arizona'
NN -> 'assistance'
NN -> 'asteroid'
NN -> 'astonishment'
NN -> 'astronomer'
NN -> 'awaken'
NN -> 'baobab'
NN -> 'beauty'
NN -> 'bit'
NN -> 'boa'
NN -> 'bolt'
NN -> 'book'
NN -> 'box'
NN -> 'breakdown'
NN -> 'breakfast'
NN -> 'breeze'
NN -> 'bridge'
NN -> 'bud'
NN -> 'care'
NN -> 'career'
NN -> 'catastrophe'
NN -> 'chair'
NN -> 'chamber'
NN -> 'chance'
NN -> 'child'
NN -> 'chimney'
NN -> 'china'
NN -> 'cold'
NN -> 'color'
NN -> 'consequence'
NN -> 'constrictor'
NN -> 'contemplation'
NN -> 'contrary'
NN -> 'copy'
NN -> 'costume'
NN -> 'cough'
NN -> 'course'
NN -> 'creature'
NN -> 'crying'
NN -> 'danger'
NN -> 'day'
NN -> 'death'
NN -> 'dejection'
NN -> 'demonstration'
NN -> 'departure'
NN -> 'desert'
NN -> 'desire'
NN -> 'drawing'
NN -> 'drinking'
NN -> 'drop'
NN -> 'earth'
NN -> 'effort'
NN -> 'elegance'
NN -> 'elephant'
NN -> 'end'
NN -> 'engine'
NN -> 'entertainment'
NN -> 'escape'
NN -> 'everybody'
NN -> 'everything'
NN -> 'fact'
NN -> 'failure'
NN -> 'fatigue'
NN -> 'fault'
NN -> 'fear'
NN -> 'field'
NN -> 'flock'
NN -> 'flower'
NN -> 'force'
NN -> 'forest'
NN -> 'form'
NN -> 'fragrance'
NN -> 'france'
NN -> 'friend'
NN -> 'geography'
NN -> 'glance'
NN -> 'glass'
NN -> 'globe'
NN -> 'golf'
NN -> 'good'
NN -> 'grace'
NN -> 'grass'
NN -> 'habitation'
NN -> 'hammer'
NN -> 'hand'
NN -> 'hat'
NN -> 'head'
NN -> 'heart'
NN -> 'height'
NN -> 'herd'
NN -> 'home'
NN -> 'house'
NN -> 'hunger'
NN -> 'hurry'
NN -> 'idea'
NN -> 'importance'
NN -> 'indifference'
NN -> 'information'
NN -> 'inside'
NN -> 'instant'
NN -> 'journey'
NN -> 'jungle'
NN -> 'kind'
NN -> 'knowledge'
NN -> 'land'
NN -> 'laugh'
NN -> 'laughter'
NN -> 'lesson'
NN -> 'level'
NN -> 'life'
NN -> 'love'
NN -> 'man'
NN -> 'matter'
NN -> 'mechanic'
NN -> 'middle'
NN -> 'migration'
NN -> 'minute'
NN -> 'model'
NN -> 'moment'
NN -> 'moralist'
NN -> 'morning'
NN -> 'muzzle'
NN -> 'mystery'
NN -> 'name'
NN -> 'nature'
NN -> 'necessity'
NN -> 'night'
NN -> 'nobody'
NN -> 'noon'
NN -> 'nothing'
NN -> 'number'
NN -> 'ocean'
NN -> 'one'
NN -> 'opinion'
NN -> 'order'
NN -> 'outside'
NN -> 'painter'
NN -> 'patience'
NN -> 'peal'
NN -> 'pencil'
NN -> 'person'
NN -> 'picture'
NN -> 'pilot'
NN -> 'place'
NN -> 'plane'
NN -> 'planet'
NN -> 'plant'
NN -> 'pleasure'
NN -> 'pocket'
NN -> 'portrait'
NN -> 'pride'
NN -> 'prince'
NN -> 'problem'
NN -> 'profession'
NN -> 'purpose'
NN -> 'question'
NN -> 'radiance'
NN -> 'raft'
NN -> 'rage'
NN -> 'railing'
NN -> 'ram'
NN -> 'region'
NN -> 'remorse'
NN -> 'reply'
NN -> 'report'
NN -> 'resemblance'
NN -> 'reserve'
NN -> 'reverie'
NN -> 'right'
NN -> 'ring'
NN -> 'room'
NN -> 'sahara'
NN -> 'sailor'
NN -> 'sand'
NN -> 'seed'
NN -> 'sense'
NN -> 'seriousness'
NN -> 'sheep'
NN -> 'shelter'
NN -> 'shrub'
NN -> 'silence'
NN -> 'soil'
NN -> 'something'
NN -> 'sort'
NN -> 'spite'
NN -> 'sprout'
NN -> 'star'
NN -> 'story'
NN -> 'style'
NN -> 'subject'
NN -> 'success'
NN -> 'suggestion'
NN -> 'sun'
NN -> 'sunrise'
NN -> 'sunset'
NN -> 'sweetness'
NN -> 'talk'
NN -> 'telescope'
NN -> 'thing'
NN -> 'thirst'
NN -> 'time'
NN -> 'tone'
NN -> 'treasure'
NN -> 'trouble'
NN -> 'truth'
NN -> 'twilight'
NN -> 'understanding'
NN -> 'untruth'
NN -> 'use'
NN -> 'verge'
NN -> 'voice'
NN -> 'volcano'
NN -> 'warfare'
NN -> 'water'
NN -> 'way'
NN -> 'week'
NN -> 'will'
NN -> 'work'
NN -> 'world'
NN -> 'youth'
NNP -> "'"
NNP -> 'A3'
NNP -> 'alas'
NNS -> 'adventures'
NNS -> 'airplanes'
NNS -> 'arms'
NNS -> 'baobabs'
NNS -> 'birds'
NNS -> 'boxes'
NNS -> 'brothers'
NNS -> 'bushes'
NNS -> 'castles'
NNS -> 'colors'
NNS -> 'constrictors'
NNS -> 'creatures'
NNS -> 'curls'
NNS -> 'darkness'
NNS -> 'days'
NNS -> 'deeds'
NNS -> 'details'
NNS -> 'doubts'
NNS -> 'elephants'
NNS -> 'encounters'
NNS -> 'errors'
NNS -> 'eruptions'
NNS -> 'eyes'
NNS -> 'feet'
NNS -> 'figures'
NNS -> 'fires'
NNS -> 'flowers'
NNS -> 'forests'
NNS -> 'friends'
NNS -> 'games'
NNS -> 'hands'
NNS -> 'i'
NNS -> 'matters'
NNS -> 'means'
NNS -> 'memories'
NNS -> 'miles'
NNS -> 'millions'
NNS -> 'misfortunes'
NNS -> 'mistakes'
NNS -> 'neckties'
NNS -> 'ones'
NNS -> 'others'
NNS -> 'paints'
NNS -> 'parts'
NNS -> 'passengers'
NNS -> 'pencils'
NNS -> 'people'
NNS -> 'petals'
NNS -> 'pictures'
NNS -> 'pieces'
NNS -> 'plants'
NNS -> 'politics'
NNS -> 'poppies'
NNS -> 'portraits'
NNS -> 'preparations'
NNS -> 'questions'
NNS -> 'repairs'
NNS -> 'reproaches'
NNS -> 'risks'
NNS -> 'roots'
NNS -> 'rosebushes'
NNS -> 'sands'
NNS -> 'secrets'
NNS -> 'seeds'
NNS -> 'shoots'
NNS -> 'shoulders'
NNS -> 'sprouts'
NNS -> 'stars'
NNS -> 'states'
NNS -> 'steps'
NNS -> 'stories'
NNS -> 'strategems'
NNS -> 'tasks'
NNS -> 'tears'
NNS -> 'things'
NNS -> 'thorns'
NNS -> 'thoughts'
NNS -> 'times'
NNS -> 'tools'
NNS -> 'trees'
NNS -> 'volcanoes'
NNS -> 'walls'
NNS -> 'weapons'
NNS -> 'words'
NNS -> 'worlds'
NNS -> 'years'
NP -> ADJP NN
NP -> CC NP CC NP
NP -> CD
NP -> CD JJ NN
NP -> CD JJ NNS
NP -> CD NN
NP -> CD NNS
NP -> DT
NP -> DT ADJP
NP -> DT ADJP JJ NN
NP -> DT ADJP NN
NP -> DT CD NNS
NP -> DT JJ
NP -> DT JJ JJ NN
NP -> DT JJ JJ NNS
NP -> DT JJ NN
NP -> DT JJ NN NN
NP -> DT JJ NN POS
NP -> DT JJ NN S
NP -> DT JJ NNS
NP -> DT JJS NN
NP -> DT NN
NP -> DT NN NN
NP -> DT NN NNS
NP -> DT NN POS
NP -> DT NN S
NP -> DT NNS
NP -> DT RB
NP -> DT VBN NN
NP -> DT VBN NNS
NP -> EX
NP -> FW
NP -> JJ
NP -> JJ ADJP NNS
NP -> JJ DT NN
NP -> JJ JJ NNS
NP -> JJ NN
NP -> JJ NN CC NN
NP -> JJ NN NN
NP -> JJ NNS
NP -> JJ NX
NP -> JJR
NP -> JJS
NP -> NN
NP -> NN CC NN
NP -> NN NNS
NP -> NNP
NP -> NNP NNP
NP -> NNS
NP -> NNS CC NNS
NP -> NP A2 CC NP
NP -> NP A2 CC NP A2
NP -> NP A2 CC NP A2 CC NP A2
NP -> NP A2 CONJP NP
NP -> NP A2 NP
NP -> NP A2 NP A2
NP -> NP A2 NP A2 CC NP A2 CC NP
NP -> NP A2 NP A2 NP
NP -> NP A2 NP A2 NP A2 NP
NP -> NP A2 PP
NP -> NP A2 PP A2
NP -> NP A2 SBAR
NP -> NP A2 SBAR A2
NP -> NP ADJP
NP -> NP CC NP
NP -> NP NN
NP -> NP NNS
NP -> NP NP
NP -> NP NP A3
NP -> NP PP
NP -> NP PP PP
NP -> NP PP S
NP -> NP PP SBAR
NP -> NP SBAR
NP -> NP VP
NP -> NP A5 SBAR
NP -> PDT DT JJ NN
NP -> PDT DT JJ NNS
NP -> PDT DT NN
NP -> PDT DT NNS
NP -> PDT PRPS NN
NP -> PDT PRPS NNS
NP -> PRP
NP -> PRPS CD NNS
NP -> PRPS JJ JJ NN
NP -> PRPS JJ JJ NNS
NP -> PRPS JJ NN
NP -> PRPS JJ NNS
NP -> PRPS JJS NN
NP -> PRPS NN
NP -> PRPS NN NN
NP -> PRPS NNS
NP -> PRPS NNS NN
NP -> PRPS VBG NN
NP -> PRPS VBG NN NN
NP -> QP
NP -> RB
NP -> RB CD NN
NP -> RB DT JJ
NP -> RB DT NN
NP -> RB JJ
NP -> VBG
NX -> NNS
PDT -> 'all'
PDT -> 'such'
POS -> "'s"
PP -> ADVP IN NP
PP -> IN
PP -> IN NP
PP -> IN S
PP -> IN SBAR
PP -> PP CC RB PP
PP -> RB IN NP
PP -> RB PP A2 RB PP
PP -> TO
PP -> TO NP
PRN -> A2 ADVP PP A2
PRN -> A2 S A2
PRP -> 'he'
PRP -> 'her'
PRP -> 'herself'
PRP -> 'him'
PRP -> 'himself'
PRP -> 'it'
PRP -> 'me'
PRP -> 'mine'
PRP -> 'myself'
PRP -> 'one'
PRP -> 'she'
PRP -> 'them'
PRP -> 'themselves'
PRP -> 'they'
PRP -> 'us'
PRP -> 'we'
PRP -> 'you'
PRP -> 'yourself'
PRPS -> 'her'
PRPS -> 'his'
PRPS -> 'its'
PRPS -> 'my'
PRPS -> 'our'
PRPS -> 'their'
PRPS -> 'your'
PRT -> RP
PRT -> RP CC RP
QP -> CD CC CD NNS
RB -> 'again'
RB -> 'ago'
RB -> 'all'
RB -> 'alone'
RB -> 'along'
RB -> 'already'
RB -> 'also'
RB -> 'always'
RB -> 'any'
RB -> 'apart'
RB -> 'as'
RB -> 'awake'
RB -> 'away'
RB -> 'carefully'
RB -> 'carelessly'
RB -> 'certainly'
RB -> 'close'
RB -> 'closely'
RB -> 'completely'
RB -> 'deeply'
RB -> 'even'
RB -> 'ever'
RB -> 'exactly'
RB -> 'extremely'
RB -> 'fairly'
RB -> 'far'
RB -> 'hard'
RB -> 'here'
RB -> 'intimately'
RB -> 'just'
RB -> 'late'
RB -> 'later'
RB -> 'little'
RB -> 'much'
RB -> "n't"
RB -> 'naively'
RB -> 'neither'
RB -> 'never'
RB -> 'not'
RB -> 'now'
RB -> 'obviously'
RB -> 'often'
RB -> 'once'
RB -> 'only'
RB -> 'peacefully'
RB -> 'perhaps'
RB -> 'really'
RB -> 'regularly'
RB -> 'right'
RB -> 'scarcely'
RB -> 'seriously'
RB -> 'simply'
RB -> 'slowly'
RB -> 'so'
RB -> 'somewhere'
RB -> 'soon'
RB -> 'steadily'
RB -> 'still'
RB -> 'straight'
RB -> 'suddenly'
RB -> 'sweetly'
RB -> 'then'
RB -> 'there'
RB -> 'therefore'
RB -> 'thunderstruck'
RB -> 'thus'
RB -> 'too'
RB -> 'uncertainly'
RB -> 'unfortunately'
RB -> 'up'
RB -> 'very'
RB -> 'well'
RB -> 'yet'
RBR -> 'better'
RBR -> 'less'
RBR -> 'more'
RBS -> 'best'
RBS -> 'most'
RP -> 'around'
RP -> 'away'
RP -> 'down'
RP -> 'off'
RP -> 'on'
RP -> 'out'
RP -> 'over'
RP -> 'up'
SBAR -> IN FRAG
SBAR -> IN NN S
SBAR -> IN S
SBAR -> RB IN S
SBAR -> RB WHADVP S
SBAR -> S
SBAR -> SBAR A4 CC SBAR
SBAR -> WHADVP A2 S
SBAR -> WHADVP S
SBAR -> WHNP S
SBAR -> WHPP S
SBARQ -> CC WHNP SQ A3
SBARQ -> WHADJP SQ A3
SBARQ -> WHNP SQ A3
SBARQ -> WHPP SQ A3
SQ -> CC VBZ NP NP A3
SQ -> MD NP VP
SQ -> NP VP
SQ -> VBZ NP ADJP A3
SQ -> VBZ NP VP
SQ -> VP
TO -> 'to'
UCP -> ADJP CC PP
UH -> 'oh'
UH -> 'yes'
VB -> 'answer'
VB -> 'appear'
VB -> 'attempt'
VB -> 'be'
VB -> 'believe'
VB -> 'breathe'
VB -> 'bring'
VB -> 'clean'
VB -> 'come'
VB -> 'complete'
VB -> 'confuse'
VB -> 'describe'
VB -> 'destroy'
VB -> 'disobey'
VB -> 'distinguish'
VB -> 'do'
VB -> 'doubt'
VB -> 'draw'
VB -> 'eat'
VB -> 'emerge'
VB -> 'excuse'
VB -> 'fall'
VB -> 'find'
VB -> 'fly'
VB -> 'forget'
VB -> 'france'
VB -> 'get'
VB -> 'give'
VB -> 'go'
VB -> 'grow'
VB -> 'have'
VB -> 'hear'
VB -> 'hold'
VB -> 'imagine'
VB -> 'know'
VB -> 'learn'
VB -> 'let'
VB -> 'like'
VB -> 'look'
VB -> 'love'
VB -> 'make'
VB -> 'move'
VB -> 'overtake'
VB -> 'place'
VB -> 'produce'
VB -> 'put'
VB -> 'reach'
VB -> 'read'
VB -> 'reassure'
VB -> 'remember'
VB -> 'reply'
VB -> 'return'
VB -> 'say'
VB -> 'see'
VB -> 'shrug'
VB -> 'sleep'
VB -> 'solve'
VB -> 'start'
VB -> 'suffer'
VB -> 'surprise'
VB -> 'swell'
VB -> 'take'
VB -> 'talk'
VB -> 'tell'
VB -> 'thank'
VB -> 'travel'
VB -> 'treat'
VB -> 'try'
VB -> 'understand'
VB -> 'unscrew'
VB -> 'want'
VB -> 'weigh'
VB -> 'wish'
VBD -> 'accepted'
VBD -> 'adjusted'
VBD -> 'answered'
VBD -> 'asked'
VBD -> 'began'
VBD -> 'believed'
VBD -> 'bent'
VBD -> 'blinked'
VBD -> 'broke'
VBD -> 'buried'
VBD -> 'called'
VBD -> 'came'
VBD -> 'cast'
VBD -> 'chose'
VBD -> 'cleaned'
VBD -> 'coughed'
VBD -> 'cried'
VBD -> 'described'
VBD -> 'did'
VBD -> 'dressed'
VBD -> 'drew'
VBD -> 'explained'
VBD -> 'felt'
VBD -> 'forced'
VBD -> 'gave'
VBD -> 'guessed'
VBD -> 'had'
VBD -> 'heard'
VBD -> 'jumped'
VBD -> 'knew'
VBD -> 'lasted'
VBD -> 'laughed'
VBD -> 'lay'
VBD -> 'learned'
VBD -> 'lived'
VBD -> 'looked'
VBD -> 'made'
VBD -> 'neglected'
VBD -> 'passed'
VBD -> 'pointed'
VBD -> 'pondered'
VBD -> 'possessed'
VBD -> 'pulled'
VBD -> 'put'
VBD -> 'realized'
VBD -> 'responded'
VBD -> 'rocked'
VBD -> 'said'
VBD -> 'sank'
VBD -> 'saw'
VBD -> 'seemed'
VBD -> 'set'
VBD -> 'showed'
VBD -> 'smelled'
VBD -> 'split'
VBD -> 'stared'
VBD -> 'stood'
VBD -> 'stopped'
VBD -> 'succeeded'
VBD -> 'tended'
VBD -> 'thought'
VBD -> 'took'
VBD -> 'tossed'
VBD -> 'was'
VBD -> 'watered'
VBD -> 'went'
VBD -> 'were'
VBD -> 'wished'
VBG -> 'awkward'
VBG -> 'becoming'
VBG -> 'blundering'
VBG -> 'breaking'
VBG -> 'digesting'
VBG -> 'drawing'
VBG -> 'eating'
VBG -> 'examining'
VBG -> 'falling'
VBG -> 'growing'
VBG -> 'having'
VBG -> 'heating'
VBG -> 'knowing'
VBG -> 'looking'
VBG -> 'making'
VBG -> 'setting'
VBG -> 'skirting'
VBG -> 'sobbing'
VBG -> 'starting'
VBG -> 'straying'
VBG -> 'swallowing'
VBG -> 'taking'
VBG -> 'trying'
VBN -> 'asked'
VBN -> 'awakened'
VBN -> 'been'
VBN -> 'blown'
VBN -> 'bought'
VBN -> 'broken'
VBN -> 'carried'
VBN -> 'caught'
VBN -> 'choked'
VBN -> 'cleaned'
VBN -> 'colored'
VBN -> 'come'
VBN -> 'comforted'
VBN -> 'concerned'
VBN -> 'cost'
VBN -> 'crashed'
VBN -> 'darkened'
VBN -> 'decided'
VBN -> 'disheartened'
VBN -> 'done'
VBN -> 'drawn'
VBN -> 'dressed'
VBN -> 'dropped'
VBN -> 'embarassed'
VBN -> 'exhausted'
VBN -> 'explained'
VBN -> 'faded'
VBN -> 'fainting'
VBN -> 'fallen'
VBN -> 'flown'
VBN -> 'found'
VBN -> 'given'
VBN -> 'gone'
VBN -> 'got'
VBN -> 'had'
VBN -> 'improved'
VBN -> 'inhabited'
VBN -> 'inseparable'
VBN -> 'judged'
VBN -> 'known'
VBN -> 'learned'
VBN -> 'let'
VBN -> 'looked'
VBN -> 'lost'
VBN -> 'loved'
VBN -> 'made'
VBN -> 'obliged'
VBN -> 'passed'
VBN -> 'revealed'
VBN -> 'run'
VBN -> 'seen'
VBN -> 'seized'
VBN -> 'stuck'
VBN -> 'suffered'
VBN -> 'surprised'
VBN -> 'taken'
VBN -> 'tried'
VBN -> 'understood'
VBN -> 'united'
VBN -> 'upset'
VBN -> 'watched'
VBN -> 'worked'
VBP -> 'add'
VBP -> 'am'
VBP -> 'are'
VBP -> 'ask'
VBP -> 'attend'
VBP -> 'believe'
VBP -> 'bring'
VBP -> 'burn'
VBP -> 'do'
VBP -> 'feel'
VBP -> 'go'
VBP -> 'have'
VBP -> 'let'
VBP -> 'like'
VBP -> 'live'
VBP -> 'make'
VBP -> 'need'
VBP -> 'pass'
VBP -> 'pull'
VBP -> 'resemble'
VBP -> 'see'
VBP -> 'sleep'
VBP -> 'speak'
VBP -> 'tell'
VBP -> 'think'
VBP -> 'try'
VBP -> 'understand'
VBZ -> "'s"
VBZ -> 'beg'
VBZ -> 'bores'
VBZ -> 'discovers'
VBZ -> 'does'
VBZ -> 'eats'
VBZ -> 'flies'
VBZ -> 'gets'
VBZ -> 'goes'
VBZ -> 'has'
VBZ -> 'is'
VBZ -> 'knows'
VBZ -> 'makes'
VBZ -> 'means'
VBZ -> 'perfumed'
VBZ -> 'recognizes'
VBZ -> 'says'
VBZ -> 'seems'
VBZ -> 'spreads'
VBZ -> 'stars'
VP -> ADVP VBD NP
VP -> JJ NP
VP -> JJ S
VP -> MD
VP -> MD ADVP VP
VP -> MD NP S
VP -> MD PRN VP
VP -> MD RB VP
VP -> MD S
VP -> MD VP
VP -> RB TO VP
VP -> TO VP
VP -> VB
VP -> VB ADJP
VP -> VB ADJP ADVP
VP -> VB ADJP SBAR
VP -> VB ADVP
VP -> VB ADVP A2 SBAR
VP -> VB NP
VP -> VB NP A2 ADVP
VP -> VB NP A2 PP
VP -> VB NP A2 PP A2 SBAR
VP -> VB NP A2 SBAR
VP -> VB NP ADVP
VP -> VB NP ADVP SBAR
VP -> VB NP NP
VP -> VB NP PP
VP -> VB NP PRT PP
VP -> VB NP S
VP -> VB NP SBAR
VP -> VB NP VP
VP -> VB PP
VP -> VB PP A2 SBAR
VP -> VB PP A2 A5 ADVP A2 SBAR
VP -> VB PP ADVP
VP -> VB PP PP
VP -> VB PP SBAR
VP -> VB PRT
VP -> VB PRT A2 ADVP A2 SBAR
VP -> VB PRT NP
VP -> VB PRT PP
VP -> VB PRT S
VP -> VB S
VP -> VB S NP A2 PP
VP -> VB SBAR
VP -> VB VP
VP -> VBD
VP -> VBD A2 ADVP
VP -> VBD A2 ADVP A2 SBAR
VP -> VBD A2 PP A2 PP
VP -> VBD ADJP
VP -> VBD ADJP A2 SBAR
VP -> VBD ADJP PP
VP -> VBD ADJP S
VP -> VBD ADJP SBAR
VP -> VBD ADVP A2 ADVP A2 PP
VP -> VBD ADVP ADJP
VP -> VBD ADVP NP
VP -> VBD ADVP PP
VP -> VBD ADVP PP A2 PP
VP -> VBD ADVP PP SBAR
VP -> VBD ADVP SBAR
VP -> VBD ADVP VP
VP -> VBD NP
VP -> VBD NP A2 ADVP A2 S
VP -> VBD NP A2 SBAR
VP -> VBD NP ADVP
VP -> VBD NP ADVP A2 PP A2 SBAR
VP -> VBD NP ADVP A2 S
VP -> VBD NP NP
VP -> VBD NP PP
VP -> VBD NP PP A2 ADVP
VP -> VBD NP PP A2 NP
VP -> VBD NP PRT ADVP
VP -> VBD NP S
VP -> VBD PP
VP -> VBD PP A2 ADJP
VP -> VBD PP A2 ADVP
VP -> VBD PP A2 PP
VP -> VBD PP ADVP
VP -> VBD PP NP
VP -> VBD PP SBAR
VP -> VBD PRT A2 PP
VP -> VBD PRT NP
VP -> VBD PRT NP ADVP
VP -> VBD PRT PP SBAR
VP -> VBD PRT SBAR
VP -> VBD RB ADJP
VP -> VBD RB ADVP VP
VP -> VBD RB NP
VP -> VBD RB PP
VP -> VBD RB VP
VP -> VBD S
VP -> VBD SBAR
VP -> VBD VP
VP -> VBG
VP -> VBG ADJP SBAR
VP -> VBG ADVP PP
VP -> VBG CC VBG
VP -> VBG NP
VP -> VBG NP A2 PP
VP -> VBG NP ADVP
VP -> VBG NP PP
VP -> VBG NP PRT PP
VP -> VBG PP
VP -> VBG PRT NP
VP -> VBG PRT PP
VP -> VBG S
VP -> VBG SBAR
VP -> VBN
VP -> VBN A2 SBAR
VP -> VBN ADJP
VP -> VBN ADJP PP
VP -> VBN ADVP
VP -> VBN ADVP ADVP
VP -> VBN ADVP NP
VP -> VBN ADVP PP
VP -> VBN NP
VP -> VBN NP ADVP A2 ADVP
VP -> VBN NP PP
VP -> VBN NP PP PP
VP -> VBN PP
VP -> VBN PP A2 SBAR
VP -> VBN PP NP PP
VP -> VBN PP PP
VP -> VBN PP SBAR
VP -> VBN PRT
VP -> VBN PRT PP
VP -> VBN S
VP -> VBN SBAR
VP -> VBN VP
VP -> VBP
VP -> VBP A2 A5 SBAR
VP -> VBP ADJP
VP -> VBP ADJP A2 SBAR
VP -> VBP ADVP
VP -> VBP ADVP ADJP
VP -> VBP ADVP NP A2 PP
VP -> VBP ADVP PP
VP -> VBP ADVP VP
VP -> VBP NP
VP -> VBP NP A2 ADVP A2 PP
VP -> VBP NP PP
VP -> VBP NP S
VP -> VBP NP SBAR
VP -> VBP PP
VP -> VBP PP S
VP -> VBP PRT NP
VP -> VBP PRT PP
VP -> VBP RB ADVP ADJP
VP -> VBP RB ADVP VP
VP -> VBP RB VP
VP -> VBP S
VP -> VBP SBAR
VP -> VBP VP
VP -> VBZ
VP -> VBZ ADJP
VP -> VBZ ADJP A2 SBAR
VP -> VBZ ADJP NP
VP -> VBZ ADJP PP
VP -> VBZ ADJP SBAR
VP -> VBZ ADVP
VP -> VBZ ADVP ADJP
VP -> VBZ ADVP PP
VP -> VBZ ADVP VP
VP -> VBZ NP
VP -> VBZ NP A2 PP
VP -> VBZ PP
VP -> VBZ PP A2 ADVP A2 SBAR
VP -> VBZ PP SBAR
VP -> VBZ PRT A2 ADVP PP
VP -> VBZ RB ADVP VP
VP -> VBZ RB VP
VP -> VBZ S
VP -> VBZ SBAR
VP -> VBZ UCP
VP -> VBZ VP
VP -> VP A2 CC VP
VP -> VP A2 VP A2 PP
VP -> VP CC VP
WDT -> 'that'
WDT -> 'what'
WDT -> 'which'
WHADJP -> WRB JJ
WHADVP -> WRB
WHNP -> DT
WHNP -> WDT
WHNP -> WDT NN
WHNP -> WDT NNS
WHNP -> WHADJP NNS
WHNP -> WP
WHPP -> IN WHNP
WP -> 'what'
WP -> 'who'
WRB -> 'how'
WRB -> 'when'
WRB -> 'whenever'
WRB -> 'where'
WRB -> 'why'
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

