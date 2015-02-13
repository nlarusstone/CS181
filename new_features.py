import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
# from sklearn import linear_model

df = pd.read_csv('../train.csv', quotechar='"', nrows=5e4)
print "Done reading"
smiles = df.smiles
N = df.shape[0]

mols = [Chem.MolFromSmiles(smile) for smile in smiles]
fp2 = [
    AllChem.GetMorganFingerprintAsBitVect(
        mol,
        3,
        nBits=1024,
        useFeatures=True) for mol in mols]
features = [DataStructs.ExplicitBitVect.ToBitString(fp) for fp in fp2]

phi = np.ndarray([map(int, feature) + [1] for feature in features])
phi_t = phi.T
t = df.gap

left = np.dot(phi_t, phi) + .21*np.identity(phi.shape[1])
right = np.dot(phi_t, t)
w = np.linalg.solve(left, right)

errs = t - np.dot(phi, w)
sqerr = sum([x ** 2 for x in errs])
rmse = np.sqrt(sqerr/N)
print rmse
