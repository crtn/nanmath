#!/bin/sh
./gen_random_matrix.py --n_rows 15 --n_cols 10 --nan_pct 0.1 --output mat.15x10

./gen_random_matrix.py --n_rows 252 --n_cols 1000 --nan_pct 0.2 --output mat.252x1k
./gen_random_matrix.py --n_rows 252 --n_cols 5000 --nan_pct 0.2 --output mat.252x5k
./gen_random_matrix.py --n_rows 252 --n_cols 10000 --nan_pct 0.2 --output mat.252x10k
./gen_random_matrix.py --n_rows 252 --n_cols 20000 --nan_pct 0.2 --output mat.252x20k
./gen_random_matrix.py --n_rows 252 --n_cols 30000 --nan_pct 0.2 --output mat.252x30k

