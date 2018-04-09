import nltk
import numpy as np

import toy_grammar
import models.model_toy
import models.model_toy_str


def pop_or_nothing(S):
    try: return S.pop()
    except: return 'Nothing'

def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    try:
        return ''.join(seq)
    except:
        return ''



class ToyGrammarModel(object):

    def __init__(self, weights_file, latent_rep_size=56):
        """ Load the (trained) zinc encoder/decoder, grammar model. """
        self._grammar = toy_grammar
        self._model = models.model_toy
        self.MAX_LEN = self._model.MAX_LEN
        self._productions = self._grammar.GCFG.productions()
        self._prod_map = {}
        for ix, prod in enumerate(self._productions):
            self._prod_map[prod] = ix
        self._parser = nltk.ChartParser(self._grammar.GCFG)
        self._tokenize = lambda x: x.split()
        self._n_chars = len(self._productions)
        self._lhs_map = {}
        for ix, lhs in enumerate(self._grammar.lhs_list):
            self._lhs_map[lhs] = ix
        self.vae = self._model.MoleculeVAE()
        self.vae.load(self._productions, weights_file, max_length=self.MAX_LEN, latent_rep_size=latent_rep_size)


    def encode(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        assert type(smiles) == list
        tokens = map(self._tokenize, smiles)
        parse_trees = [self._parser.parse(t).next() for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
        one_hot = np.zeros((len(indices), self.MAX_LEN, self._n_chars), dtype=np.float32)
        for i in xrange(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        self.one_hot = one_hot
        return self.vae.encoder.predict(one_hot)

    def encodeMV(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        assert type(smiles) == list
        tokens = map(self._tokenize, smiles)
        parse_trees = [self._parser.parse(t).next() for t in tokens]
        productions_seq = [tree.productions() for tree in parse_trees]
        indices = [np.array([self._prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
        one_hot = np.zeros((len(indices), self.MAX_LEN, self._n_chars), dtype=np.float32)
        for i in xrange(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        self.one_hot = one_hot
        return self.vae.encoderMV.predict(one_hot)[0]

    def _sample_using_masks(self, unmasked):
        """ Samples a one-hot vector, masking at each timestep.
            This is an implementation of Algorithm ? in the paper. """
        eps = 1e-100
        X_hat = np.zeros_like(unmasked)
        print(X_hat)
        # Create a stack for each input in the batch
        S = np.empty((unmasked.shape[0],), dtype=object)
        print(S)
        for ix in xrange(S.shape[0]):
            S[ix] = [str(self._grammar.start_index)]
        print(S)
        print(unmasked.shape[1])
        # Loop over time axis, sampling values and updating masks
        for t in xrange(unmasked.shape[1]):
            print('S_____________________________________')
            print(S)
            next_nonterminal = [self._lhs_map[pop_or_nothing(a)] for a in S]
            print('______next_nonterminal______')
            print(next_nonterminal)
            mask = self._grammar.masks[next_nonterminal]
            print('______mask______')
            print(mask)
            masked_output = np.exp(unmasked[:,t,:])*mask + eps
            print('______masked_output______')
            print(masked_output)
            sampled_output = np.argmax(np.random.gumbel(size=masked_output.shape) + np.log(masked_output), axis=-1)
            print('______sampled_output______')
            print(sampled_output)
            X_hat[np.arange(unmasked.shape[0]),t,sampled_output] = 1.0
            # Identify non-terminals in RHS of selected production, and
            # push them onto the stack in reverse order
            rhs = [filter(lambda a: (type(a) == nltk.grammar.Nonterminal) and (str(a) != 'None'),
                          self._productions[i].rhs()) 
                   for i in sampled_output]
            print('______rhs_______')
            print(rhs)
            for ix in xrange(S.shape[0]):
                S[ix].extend(map(str, rhs[ix])[::-1])
        return X_hat # , ln_p

    def decode(self, z):
        """ Sample from the grammar decoder """
        assert z.ndim == 2
        unmasked = self.vae.decoder.predict(z)
        X_hat = self._sample_using_masks(unmasked)
        # Convert from one-hot to sequence of production rules
        prod_seq = [[self._productions[X_hat[index,t].argmax()] 
                     for t in xrange(X_hat.shape[1])] 
                    for index in xrange(X_hat.shape[0])]
        return [prods_to_eq(prods) for prods in prod_seq]



class ToyCharacterModel(object):

    def __init__(self, weights_file, latent_rep_size=56):
        self._model = models.model_toy_str
        self.MAX_LEN = 80
        self.vae = self._model.MoleculeVAE()
        self.charlist = ['a','b','c','d','e','f','g','h','i','k','l','m','n','o','p','r','s','t','u','v','w','y',' ']
        self._char_index = {}
        for ix, char in enumerate(self.charlist):
            self._char_index[char] = ix
        self.vae.load(self.charlist, weights_file, max_length=self.MAX_LEN, latent_rep_size=latent_rep_size)


    def encode(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in xrange(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        return self.vae.encoder.predict(one_hot)

    def encodeMV(self, smiles):
        """ Encode a list of smiles strings into the latent space """
        indices = [np.array([self._char_index[c] for c in entry], dtype=int) for entry in smiles]
        one_hot = np.zeros((len(indices), self.MAX_LEN, len(self.charlist)), dtype=np.float32)
        for i in xrange(len(indices)):
            num_productions = len(indices[i])
            one_hot[i][np.arange(num_productions),indices[i]] = 1.
            one_hot[i][np.arange(num_productions, self.MAX_LEN),-1] = 1.
        return self.vae.encoderMV.predict(one_hot)[0]

    def decode(self, z):
        """ Sample from the character decoder """
        assert z.ndim == 2
        out = self.vae.decoder.predict(z)
        noise = np.random.gumbel(size=out.shape)
        sampled_chars = np.argmax(np.log(out) + noise, axis=-1)
        char_matrix = np.array(self.charlist)[np.array(sampled_chars, dtype=int)]
        return [''.join(ch).strip() for ch in char_matrix]
