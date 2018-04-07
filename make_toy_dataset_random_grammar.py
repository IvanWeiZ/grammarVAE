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


tokens="once when i was six years old i saw a magnificent picture in a book , called true stories from nature , about the primeval forest .".split()
tokens="i have tried .".split()
tokens="i have seen them intimately , close at hand .".split()
tokens="in a week the NNS would VB out".split()
parser = nltk.ChartParser(prince_grammar.GCFG)
parse_trees = parser.parse_one(tokens)
for tree in parse_trees:
    print(tree)

# MAX_LEN=277
# NCHARS = len(prince_grammar.GCFG.productions())

# parser = nltk.ChartParser(prince_grammar.GCFG)

# parse_trees = [parser.parse(t).next() for t in tokens]
# print(parse_trees)

# productions_seq = [tree.productions() for tree in parse_trees]
# print(productions_seq)

# indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
# print(indices)

# one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
# print(one_hot)
# for tree in parser.parse_one(sent):
#     print(tree)