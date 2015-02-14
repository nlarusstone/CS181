import pandas as pd
import numpy as np

df = pd.read_csv('train_256_100000.csv', header=None)
N = len(df)
phi = np.array(df.iloc[:, :-1])
phi_t = phi.T
gaps = np.array(df.iloc[:, -1])

left = np.dot(phi_t, phi) + .22*np.identity(phi.shape[1])
right = np.dot(phi_t, gaps)
w = np.linalg.solve(left, right)

errs = gaps - np.dot(phi, w)
sqerr = sum([x ** 2 for x in errs])
rmse = np.sqrt(sqerr/N)
print rmse
