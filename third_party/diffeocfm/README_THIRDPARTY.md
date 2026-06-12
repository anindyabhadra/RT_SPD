# DiffeoCFM baseline files

This directory contains selected files copied from the DiffeoCFM repository by Collas et al., used only to reproduce the baseline comparisons. Their original LICENSE is included in this directory.

The file `data.py` is adapted from the original DiffeoCFM repository. In this copy, imports and dataset-loading utilities requiring `moabb` were removed or disabled to avoid dependency conflicts; the remaining functionality used by this repository is unchanged.

Most selected files are MIT licensed under the upstream DiffeoCFM license. The file `spd.py` is adapted from `riemannian-fm` and is licensed under CC BY-NC 4.0. Baselines depending on `spd.py`, including Riemannian SPD-CFM, are therefore non-commercial unless separate permission is obtained.

Link to original repository: https://github.com/antoinecollas/DiffeoCFM
