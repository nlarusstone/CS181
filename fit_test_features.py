import pandas as pd
import numpy as np
from itertools import izip
import csv

train_df = pd.read_csv('train_512_250000.csv', header=None)
phi = np.array(train_df.iloc[:, :-1])
phi_t = phi.T
t = np.array(train_df.iloc[:, -1])

left = np.dot(phi_t, phi) + .22*np.identity(phi.shape[1])
right = np.dot(phi_t, t)
w = np.linalg.solve(left, right)

test_df = pd.read_csv('test_512features.csv', header=None)
test_phi = np.array(test_df.iloc[:, 1:])
ids = np.array(test_df.iloc[:, 0])
N = len(test_phi)
preds = np.dot(test_phi, w)

with open('pred.csv', 'w') as pred_fh:
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')
    pred_csv.writerow(['Id', 'Prediction'])
    for ID, pred in izip(ids, preds):
        pred_csv.writerow([int(ID), pred])
