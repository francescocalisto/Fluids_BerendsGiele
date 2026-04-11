# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Study of ordered Berends–Giele (BG) recursions for fluid dynamical systems, focusing on the 2D Burgers equation (no pressure, no incompressibility). Research for Francesco Calisto's Caltech PhD (advisor: Cliff). Connection to Poisson associahedra and NNSE recursions.

## Running the code

The Python environment with numpy is in the conda `base` environment:

```bash
conda run -n base python3 burgers_100pt_1.py            # 6 configs at n=100
conda run -n base python3 burgers_100pt_1.py --n-max 50 # stop at n=50
conda run -n base python3 burgers_100pt_1.py --plot     # save bg_growth_rates.png

conda run -n base python3 validate_bg.py                # run all 4 validation tests
conda run -n base python3 validate_bg.py --plot         # also save validate_plots.png
```

## Key files

- `burgers_100pt_1.py` — main BG recursion code; `bg_nonabelian` and `bg_abelian` are the core functions.
- `validate_bg.py` — four numerical validation tests (see below).
- `burgers_100pt_note.pdf` — theory notes with equations, results table, and interpretation.

## Physics and architecture

**Propagator**: `s[i,j] = |K_{i..j}|² − Σ|k_m|²` encodes the Mandelstam-like denominator.

**Two recursions** (eq. 3–4 of the notes):
- Non-abelian (Lie bracket, antisymmetric): `s_P u_P = Σ_{P=QR} [(u_Q·K_R)u_R − (u_R·K_Q)u_Q]`
- Abelian (one-sided advection): `s_P u_P = Σ_{P=QR} (u_Q·K_R)u_R`

**Polarization sectors**:
- Longitudinal `ε=k`: reduces to scalar Burgers → Cole–Hopf integrable → currents stay bounded.
- Transverse `ε=k⊥`: non-integrable → geometric growth ~10^n per step (non-abelian).

**Exact Cole–Hopf formula** (eq. 5, proved): for the full unordered abelian current with longitudinal polarizations:
```
u_{i,P} = K_{P,i} · (n−1)! / 2^{n−1}
```
Key structural insight: the abelian BG current for any single ordering equals `k_last / 2^{n−1}`; summing all n! orderings gives `(n−1)! K_P / 2^{n−1}`.

## Validation tests

Test A verifies the Cole–Hopf formula above by summing the abelian BG over all permutations (n=2..7); relative error is at machine precision (~10⁻¹⁵).

Test B checks the n=2 non-abelian transverse formula analytically: `J[0,1] = (k₁×k₂)(ε₁+ε₂) / (2 k₁·k₂)`.

Test C verifies transversality preservation `K_P·J = 0` for the non-abelian vertex (ratio |u_∥/u_⊥| ~ 10⁻¹⁶ at low n).

Test D shows the longitudinal/transverse contrast: slope ≈ −0.17 (bounded) vs +1.21 (geometric growth).

## Known numerical results (n=100, from the notes)

| Config | log10\|J₁₀₀\|² | slope (dec/step) |
|--------|----------------|-----------------|
| NA/Transverse/Kol shell | 111.5 | +1.21 |
| NA/Transverse/Cascade   | 110.6 | +0.97 |
| NA/Transverse/IR        |  81.8 | +0.95 |
| Abelian/Transverse/Kol  |  44.8 | +0.29 |
| NA/Longitudinal/Kol     |   8.7 | −0.17 |
| NA/Transverse/Lattice   |  62.0 | −0.24 |
