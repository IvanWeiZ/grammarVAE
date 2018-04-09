import numpy as np
import pdb
from models.utils import many_one_hot
import h5py
import argparse

path='data/toylanguage100k'

def get_arguments():
    parser = argparse.ArgumentParser(description='Make toy datset')
    parser.add_argument('--path', type=str, default=path)
    return parser.parse_args()

args=get_arguments()
print(args.path)

f = open(args.path,'r')

L = []
chars = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','r','s','t','u','v','w','y',' ']
DIM = len(chars)
for line in f:
    line = line.strip()
    L.append(line)
f.close()

count = 0
MAX_LEN = 80
OH = np.zeros((len(L),MAX_LEN,DIM))
for line in L:
    indices = []
    for c in line:
        indices.append(chars.index(c))
    if len(indices) < MAX_LEN:
        indices.extend((MAX_LEN-len(indices))*[DIM-1])
    OH[count,:,:] = many_one_hot(np.array(indices), DIM)
    count = count + 1
f.close()
h5f = h5py.File('toy_str_dataset.h5','w')
h5f.create_dataset('data', data=OH)
h5f.create_dataset('chr',  data=chars)
h5f.close()
