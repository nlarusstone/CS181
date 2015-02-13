import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn import linear_model

df = pd.DataFrame.from_csv('../train.csv')
N = df.shape[0]
print "Done reading"

phi = df.iloc[:, 1:-1]
phi['const'] = 1
phi_t = phi.transpose()
t = df.gap

left = np.dot(phi_t, phi) + .21*np.identity(phi.shape[1])
right = np.dot(phi_t, t)
w = np.linalg.solve(left, right)

errs = t - np.dot(phi, w)
sqerr = sum([x ** 2 for x in errs])
rmse = np.sqrt(sqerr/N)
print rmse
