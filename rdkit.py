__author__ = 'nlarusstone'
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

df = pd.read_csv("../train.csv", quotechar='"', nrows=1e3)
smiles = df.iloc[:, 0]
N = len(df)
mols = [0 for i in xrange(N)]
fp2 = [0 for i in xrange(N)]
features = [0 for i in xrange(N)]
for i in xrange(N):
    mols[i] = Chem.MolFromSmiles(smiles[i])
    fp2[i] = AllChem.GetMorganFingerprintAsBitVect(mols[i], 3, nBits=1024, useFeatures=True)
    features[i] = DataStructs.ExplicitBitVect.ToBitString(fp2[i])