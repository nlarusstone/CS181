import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

NUM_DATA = int(1e5)
NUM_FEATURES = 256

df = pd.read_csv('train.csv', quotechar='"', nrows=NUM_DATA)
print "Done reading... now calculating features"
smiles = df.smiles
N = df.shape[0]

mfs = Chem.MolFromSmiles
gmfbv = lambda x: AllChem.GetMorganFingerprintAsBitVect(
    x,
    4,
    nBits=NUM_FEATURES,
    useFeatures=True)
tobitstr = DataStructs.ExplicitBitVect.ToBitString
all_func = lambda x: tobitstr(gmfbv(mfs(x)))
features = [all_func(x) for x in smiles]
phi = np.array([[1] + map(int, feat) for feat in features])
t = df.gap

all_data = np.zeros((N, NUM_FEATURES + 2))
all_data[:, :-1] = phi
all_data[:, -1] = t

print 'Done calculating features... now writing data'

np.savetxt('train_{0}_{1}.csv'.format(NUM_FEATURES, NUM_DATA), all_data,
           delimiter=',')
