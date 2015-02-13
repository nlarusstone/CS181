import csv
import gzip
import numpy as np
import pandas as pd


train_filename = '../7feats_train.csv'
test_filename = '../test.csv.gz'
pred_filename = 'pred.csv'

# Load the training file.
train_data = []
# with gzip.open(train_filename, 'r') as train_fh:
with open(train_filename, 'r') as train_fh:

    # Parse it as a CSV file.
    train_df = pd.read_csv(train_fh, quotechar='"')

print 'done reading data'
phi = train_df.iloc[:, 1:-1]
phi['const'] = 1
phi_t = phi.transpose()
t = train_df.gap
N = phi.shape[0]
S = 3.0
best_lam = None
best_rmse = 100
for lam in np.linspace(0.18, 0.22, 8):
    tot_err = pd.Series()
    for test in xrange(int(S)):
        lt_test_bnd = int(test*N/S)
        rt_test_bnd = int((test+1)*N/S)
        test_phi = phi.iloc[0:lt_test_bnd].append(phi.iloc[rt_test_bnd:])
        test_phi_t = test_phi.transpose()
        test_t = t.iloc[0:lt_test_bnd].append(t.iloc[rt_test_bnd:])
        left = lam*np.identity(test_phi.shape[1]) + np.dot(test_phi_t, test_phi)
        right = np.dot(test_phi_t, test_t)
        test_w = np.linalg.solve(left, right)
        validation_phi = phi.iloc[lt_test_bnd:rt_test_bnd]
        validation_t = t.iloc[lt_test_bnd:rt_test_bnd]
        errors = np.dot(validation_phi, test_w) - validation_t
        tot_err = tot_err.append(errors)
    rmse = np.sqrt(np.dot(tot_err.transpose(), tot_err)/N)
    if rmse < best_rmse:
        best_rmse = rmse
        best_lam = lam
    print 'lambda = {0:.3f} -> rmse = {1:.8f}'.format(lam, rmse)
print best_lam

left = best_lam*np.identity(phi.shape[1]) + np.dot(phi_t, phi)
right = np.dot(phi_t, t)
w = np.linalg.solve(left, right)

# Load the test file.
with gzip.open(test_filename, 'r') as test_fh:

    test_df = pd.read_csv(test_fh, quotechar='"')

test_phi = test_df.iloc[:, 2:]
test_phi['const'] = 1
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
