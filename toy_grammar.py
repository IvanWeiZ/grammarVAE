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

gram = """S -> NP VP
    NP -> Det N
    NP -> Det Adj N
    VP -> V NP
    VP -> V NP PP
    PP -> P Det Pla
    Det -> 'a'
    Det -> 'the'
    Det -> 'my'
    Det -> 'your'
    NP -> 'bob'
    NP -> 'kevin'
    NP -> 'kyle'
    N -> 'man'
    N -> 'dog'
    N -> 'cat'
    N -> 'chicken'
    N -> 'bird'
    N -> 'pig'
    N -> 'lion'
    N -> 'bear'
    N -> 'turkey'
    N -> 'wolf'
    N -> 'rabbit'
    N -> 'duck'
    N -> 'monkey'
    V -> 'saw'
    V -> 'killed'
    V -> 'caught'
    V -> 'chased'
    V -> 'played'
    P -> 'in'
    P -> 'by'
    Pla -> 'park'
    Pla -> 'school'
    Pla -> 'forest'
    Adj -> 'angry'
    Adj -> 'frightened'
    Adj -> 'little'
    Adj -> 'wild'
    Adj -> 'big'"""


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

