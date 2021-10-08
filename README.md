# Slingshot in Python

This is a Python implementation of the Slingshot pseudotime algorithm (Street et al., 2018). 
The original implementation is written in R: https://github.com/kstreet13/slingshot.

A complete working example is located in `slingshot.ipynb`, using a synthetically generated dataset.

Installation: 

- Requires `numpy`, `scipy`, `sklearn`
- My fork of `pcurvepy` (https://github.com/mossjacob/pcurvepy):
  - `pip install git+https://github.com/mossjacob/pcurvepy`

- Finally install pyslingshot:
  - `pip install pyslingshot`

<img src=readme_example.png height="200">

[1] Street, K., Risso, D., Fletcher, R.B., Das, D., Ngai, J., Yosef, N., Purdom, E. and Dudoit, S., 2018. Slingshot: cell lineage and pseudotime inference for single-cell transcriptomics. BMC genomics, 19(1), pp.1-16.
