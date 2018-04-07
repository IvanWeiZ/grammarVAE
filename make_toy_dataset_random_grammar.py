from random import choice
import nltk
import prince_grammar
import sys
# This function is based on _generate_all() in nltk.parse.generate
# It therefore assumes the same import environment otherwise.

def generate_sample(grammar, prod, frags):        
    if prod in grammar._lhs_index: # Derivation
        derivations = grammar._lhs_index[prod]            
        derivation = choice(derivations)            
        for d in derivation._rhs:            
            generate_sample(grammar, d, frags)
    elif prod in grammar._rhs_index:
        # terminal
        frags.append(str(prod))





# # print(generate_sample(toy_grammar.gram))
# print(prince_grammar.GCFG.start())
# frags = []  
# generate_sample(prince_grammar.GCFG, prince_grammar.GCFG.start(), frags)
# print( ' '.join(frags) )

# frags = []  
# generate_sample(prince_grammar.GCFG, prince_grammar.GCFG.start(), frags)
# print( ' '.join(frags) )

# from nltk.parse.generate import generate, demo_grammar
# from nltk import CFG

# for sentence in generate(prince_grammar.GCFG,n=1):
#      print(' '.join(sentence))

parser = nltk.ChartParser(prince_grammar.GCFG)

sent="once when i was six years old i saw a magnificent picture in a book , called true stories from nature , about the primeval forest .".split()
sent="i have tried .".split()
sent="i have seen them intimately , close at hand .".split()
sent="and that is how i made the acquaintance of the little prince .".split()
for tree in parser.parse_one(sent):
    print(tree)