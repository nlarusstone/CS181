import csv
import gzip
import numpy as np
import pandas as pd
from rdkit import chem

train_filename = '../train.csv.gz'
test_filename = '../test.csv.gz'
pred_filename = 'pred.csv'

# Load the training file.
train_data = []
with gzip.open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_df = pd.read_csv(train_fh, quotechar='"')

phi = train_df.iloc[:, 1:-1]
phi_t = phi.transpose()
t = train_df.gap
left = np.identity(phi.shape[1]) + np.dot(phi_t, phi)
right = np.dot(phi_t, t)
w = np.linalg.solve(left, right)

# Load the test file.
with gzip.open(test_filename, 'r') as test_fh:

    test_df = pd.read_csv(test_fh, quotechar='"')

test_phi = test_df.iloc[:, 2:]
ids = test_df.iloc[:, 0]
pred_gaps = np.dot(test_phi, w.T)

# Write a prediction file.
with open(pred_filename, 'w') as pred_fh:

    # Produce a CSV file.
    pred_csv = csv.writer(pred_fh, delimiter=',', quotechar='"')

    # Write the header row.
    pred_csv.writerow(['Id', 'Prediction'])

    for ID, pred in zip(ids, pred_gaps):
        pred_csv.writerow([ID, pred])
