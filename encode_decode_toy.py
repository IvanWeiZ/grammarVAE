import sys
#sys.path.insert(0, '..')
import toy_vae
import numpy as np

# 1. load grammar VAE
grammar_weights = "results/toy_vae_grammar_L56_E100_val.hdf5"
grammar_model = toy_vae.ToyGrammarModel(grammar_weights)

# 2. let's encode and decode some example SMILES strings
smiles = []

# z: encoded latent points
# NOTE: this operation returns the mean of the encoding distribution
# if you would like it to sample from that distribution instead
# replace line 83 in molecule_vae.py with: return self.vae.encoder.predict(one_hot)
z1 = grammar_model.encode(smiles)

# mol: decoded SMILES string
# NOTE: decoding is stochastic so calling this function many
# times for the same latent point will return different answers

for mol,real in zip(grammar_model.decode(z1),smiles):
    print mol + '  ' + real



# 3. the character VAE (https://github.com/maxhodak/keras-molecules)
# works the same way, let's load it
char_weights = "pretrained/toy_vae_str_L56_E100_val.hdf5"
char_model = molecule_vae.ToyCharacterModel(char_weights)

# 4. encode and decode
z2 = char_model.encode(smiles)
for mol in char_model.decode(z2):
    print mol


