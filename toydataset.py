from nltk.parse.generate import generate, demo_grammar
from nltk import CFG
# grammar = CFG.fromstring(
#     """S -> NP VP
#     NP -> Det N
#     PP -> P NP
#     VP -> 'slept'
#     VP -> 'saw' NP
#     VP -> 'wanted' NP
#     VP -> 'walked' PP
#     Det -> 'the'
#     Det -> 'a'
#     N -> 'man'
#     N -> 'woman'
#     N -> 'park'
#     N -> 'dog'
#     P -> 'in'
#     P -> 'with'""")

# grammar = CFG.fromstring(
#     """ S -> NP VP
#         PP -> P NP    
#         NP -> Det N | Det N PP | 'I'
#         VP -> V NP | VP PP
#         Det -> 'an' | 'my'
#         N -> 'elephant' | 'pajamas'
#         V -> 'shot'
#         P -> 'in'
#         """)

# grammar = CFG.fromstring(
#     """ S   -> NP VP
#         NP  -> Det N | Det Adj N
#         VP  -> V NP | V NP PP
#         PP  -> P Det Pla
#         Det -> 'a' | 'the' | 'my' | 'your' | 'his'
#         NP  -> 'john' | 'may' | 'bob' | 'kevin' | 'kyle' 
#         N   -> 'man' | 'dog' | 'cat' | 'telescope'  | 'bear' | 'squirrel' | 'tree' | 'fish'
#         N   -> 'dolphin' | 'eagle' | 'horse' | 'chicken' | 'bird' | 'shark' | 'pig' | 'lion'
#         N   -> 'turkey' | 'wolf' | 'rabbit' | 'duck' | 'monkey'
#         V   -> 'saw' | "killed" | 'ate' | 'chased' | 'said' | 'was' | 'cooked' | 'shot' | 'played'
#         P   -> 'in' | 'on' | 'by' | 'with'
#         Pla ->  'park' | 'school' | 'library'
#         Adj  -> 'angry' | 'frightened' |  'little' | 'tall' | 'playful' | 'great' | "giant"
#         Adj  -> 'native' | 'shy' |  'untrained' | 'wild' | 'loyal' | 'lazy' | "gentle"
#         """)

grammar = CFG.fromstring(
    """ S   -> NP VP
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
        """)

for sentence in generate(grammar):
     print(' '.join(sentence))
