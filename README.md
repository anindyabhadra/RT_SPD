# RT_SPD: Reverse Telescoping (RT) Generative SPD Split Flow and Langevin Diffusion

This repository is organized around the synthetic experiments reported in the paper.

## Setting up the virtual environment

From the repository root run:

```bash
setup_venv.sh
```

## Layout

```text
basic_rt/
  rt_core.py                 Core RT encoding/decoding/square-root/verification routines.
  verify_rt.py               Small numerical verification script.

flows/
  run_wishart_spd_baselines_distributional.py
                             Shared Wishart/Wishart-mixture simulation and metrics generator.
  split_free_flow_rt_shared.py
                             Conditional triangular RT split-free flow.
  split_hamiltonian_flow_rt_shared.py
                             Conditional triangular RT Hamiltonian/divergence-free flow.

langevin/
  rt_intrinsic_langevin_wishart_mixture_mainfig.py
                             Intrinsic RT Langevin sampler for Wishart-mixture targets.
                             
third_party/diffeocfm/
  Selected DiffeoCFM baseline files from Collas et al. (2025) and their original license notice.
```


## Basic verification

From the repository root:

```bash
python basic_rt/verify_rt.py --p 8
```

This checks reconstruction of `Theta`, direct reconstruction of `Theta^{-1}`, square-root decoders, the log-determinant coordinate, determinant-one normalized shape paths, and the intrinsic `y=(v,d,beta)` decoder.

## Generative flow experiments

From the repository root, to run only the two RT flows:

```bash
python flows/run_wishart_spd_baselines_distributional.py \
  --target mixture \
  --p 20 \
  --df 40 \
  --n-train 5000 \
  --n-test 2000 \
  --epochs 300 \
  --out-prefix wishart_p20_mix \
  --skip-diffeocfm \
  --skip-spd-cfm
```

To include DiffeoCFM and Riemannian SPD-CFM baselines omit the `--skip-diffeocfm` / `--skip-spd-cfm` flags.

## Intrinsic Langevin experiment

```bash
python langevin/rt_intrinsic_langevin_wishart_mixture_mainfig.py \
  --p-list 20 50 \
  --df-extra1 20 \
  --df-extra2 20 \
  --n-chains 56 \
  --n-steps 20000 \
  --burn-in 10000 \
  --thin 50 \
  --dt 1e-4 \
  --n-true 8000 \
  --init mixture_means \
  --out rt_langevin_wishart_mixture.png
```

## Removing cached files and cleanup

To remove cached files, from the repository root run:

```bash cleanup_venv.sh
```

To remove cached files and results, from the repository root run:

```bash cleanup_venv.sh --all
```

## License

The original code in this repository is released under the MIT License; see
`LICENSE`.

This repository also contains selected third-party baseline files under
`third_party/diffeocfm/`. These files are not covered by the root MIT license. They retain their original licenses. In particular, the DiffeoCFM repository states that most of its code is MIT licensed, while `spd.py` is adapted from `riemannian-fm` and is licensed under CC BY-NC 4.0. Baselines depending on `spd.py`, including the Riemannian SPD-CFM baseline, are therefore restricted to non-commercial use unless separate permission is obtained from the relevant rights holders.


## Reference

