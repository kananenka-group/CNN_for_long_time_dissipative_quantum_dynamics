# Convolutional Neural Networks for Long Time Dissipative Quantum Dynamics

This repository contains code and data accompanying the following paper: [L. E. H. Rodriguez and A. A. Kananenka, *J. Phys. Chem. Lett.* 12, 2476-2483 (2021)](https://pubs.acs.org/doi/10.1021/acs.jpclett.1c00079)

## Data:
Each file X.s in ./data folder contains four elements of the reduced density matrix obtained by running [PHI software](https://www.ks.uiuc.edu/Research/phi/)
for a set of parameters specified in ./data/key.txt. First column in ./data/key.txt file contains a number X that points to a data file X.s. Second through
fourth columns specify the bias (epsilon), reorganization energy (lambda), and temperature (T) correspndingly. For example, data file 0.s contains the reduced density matrix elements for 
epsilon=0, lambda=2.0, and T=50.

