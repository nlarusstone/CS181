import pandas as pd
import numpy as np
from itertools import izip
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

NUM_FEATURES = 256

df = pd.read_csv('test.csv', quotechar='"')
print 'Done reading... now calculating features'
ids = df.Id
smiles = df.smiles
N = df.shape[0]

mfs = Chem.MolFromSmiles
gmfbv = lambda x: AllChem.GetMorganFingerprintAsBitVect(
    x,
    4,
    nBits=NUM_FEATURES,
    useFeatures=True)
tobitstr = DataStructs.ExplicitBitVect.ToBitString
all_func = lambda s: tobitstr(gmfbv(mfs(s)))
features = np.array([all_func(sm) for sm in smiles])

phi = np.array([[ID, 1] + map(int, feat) for ID, feat in izip(ids, features)])

print 'Done calculating features... now writing data'

np.savetxt('test_{0}features.csv'.format(NUM_FEATURES), phi, delimiter=',')
