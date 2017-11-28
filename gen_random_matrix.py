#!/usr/bin/env python
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('--n_rows', type=int, required=True)
parser.add_argument('--n_cols', type=int, required=True)
parser.add_argument('--nan_pct', type=float, required=True)
parser.add_argument('--output', type=str, required=True)

args = parser.parse_args()

n_rows = args.n_rows
n_cols = args.n_cols
nan_pct = args.nan_pct
output = args.output

mu_0 = 0.7
mu_1 = 1.5
var_0 = 2.2
var_1 = 4.5
cov = 2.7
rv = stats.multivariate_normal([mu_0, mu_1], [[var_0, cov], [cov, var_1]])

m = np.hstack(rv.rvs(((n_cols + 1) / 2, n_rows)))[:, range(n_cols)]
l = int(n_rows * n_cols * nan_pct)
x, y = np.meshgrid(range(n_rows), range(n_cols))
selected = np.random.choice(xrange(n_rows * n_cols), l, replace=False)
indexes = zip(np.hstack(x), np.hstack(y))
for sel in selected:
    i, j = indexes[sel]
    m[i, j] = np.nan

df = pd.DataFrame(m)
df.to_csv(output, index=None, header=None, na_rep='nan')

