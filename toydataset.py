from nltk.parse.generate import generate, demo_grammar
from nltk import CFG
grammar = CFG.fromstring(
    """S -> NP VP
    NP -> Det N
    PP -> P NP
    VP -> 'slept'
    VP -> 'saw' NP
    VP -> 'wanted' NP
    VP -> 'walked' PP
    Det -> 'the'
    Det -> 'a'
    N -> 'man'
    N -> 'park'
    N -> 'dog'
    P -> 'in'
    P -> 'with'""")

for sentence in generate(grammar, n=100):
     print(' '.join(sentence))
