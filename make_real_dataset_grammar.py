from __future__ import print_function
import nltk
import pdb
import real_grammar
import numpy as np
import h5py
# import toy_vae
import argparse


path='data/real1_lower.txt'

def get_arguments():
    parser = argparse.ArgumentParser(description='Make toy datset')
    parser.add_argument('--path', type=str, default=path)
    return parser.parse_args()

args=get_arguments()
print(args.path)

f = open(args.path,'r')
L = []

count = -1
for line in f:
    line = line.strip()
    L.append(line)
f.close()

MAX_LEN=100
NCHARS = len(real_grammar.GCFG.productions())

def to_one_hot(strs):
    """ Encode a list of strs strings to one-hot vectors """
    prod_map = {}
    for ix, prod in enumerate(real_grammar.GCFG.productions()):
        prod_map[prod] = ix
    tokens = map(lambda x: x.split(), strs)
    parser = nltk.ChartParser(real_grammar.GCFG)
    parse_trees = [parser.parse(t).next() for t in tokens]
    productions_seq = [tree.productions() for tree in parse_trees]
    print(max(map(lambda x: len(x), strs)))
    indices = [np.array([prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    one_hot = np.zeros((len(indices), MAX_LEN, NCHARS), dtype=np.float32)
    for i in xrange(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot


OH = np.zeros((len(L),MAX_LEN,NCHARS))
for i in range(0, len(L),100):
    # try:
    #     to_one_hot([L[i]])
    # except Exception as e:
    #     print(i+1,"\t\t",L[i])
    #     print(str(e))
    
    print('Processing: i=[' + str(i) + ':' + str(i+100) + ']')
    onehot = to_one_hot(L[i:i+100])
    OH[i:i+100,:,:] = onehot

h5f = h5py.File('real_grammar_dataset.h5','w')
h5f.create_dataset('data', data=OH)
h5f.close()
