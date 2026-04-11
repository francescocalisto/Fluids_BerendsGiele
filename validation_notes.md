# Validation notes for the 2D Burgers BG recursion

This document explains, in honest detail, how the four numerical tests in
`validate_bg.py` are constructed and what the resulting plots (`validate_plots.png`)
actually show — including their limitations.

---

## Infrastructure

Both recursion functions (`bg_nonabelian`, `bg_abelian` in `burgers_100pt_1.py`)
fill an `n × n` array `J[i,j]` that stores the BG current for the contiguous
sub-word `(i, i+1, …, j)`.  The propagator denominator is

    s[i,j] = |K_{i..j}|² − Σ_{p=i}^{j} |k_p|²  =  2 Σ_{p<q} k_p · k_q

computed once via prefix sums (`precompute`).  Each test exercises this
infrastructure on specific momentum/polarization inputs and cross-checks the
output against an independent prediction.

---

## Test A — Cole–Hopf formula via full symmetrization

**What it checks.**
For longitudinal polarizations `ε_i = k_i`, the abelian BG current for a
fixed ordering `(1,…,n)` equals `k_n / 2^{n−1}` — only the last momentum
survives.  Summing over all `n!` orderings places every `k_i` in the last
slot exactly `(n−1)!` times, giving the Cole–Hopf prediction

    Σ_{σ ∈ S_n} J_abelian(k_{σ₁},…,k_{σₙ})  =  (n−1)! K_P / 2^{n−1}

**Implementation.**
For each `n = 2,…,7`, one set of `n` random momenta is drawn (seed 0) and
`bg_abelian` is called for all `n!` permutations.  The sum is compared to the
formula above via relative error `|BG sum − formula| / |formula|`.

**Honest caveats.**
- The upper limit `n = 7` (5040 permutations) is chosen for speed; `n = 8`
  (40 320 calls) is feasible but has not been run here.
- Only a single random instance is tested per `n`.  The identity is proved
  analytically so this is a consistency / bug-detection check, not a
  statistical one.
- Relative errors land at `3×10⁻¹⁶ – 3×10⁻¹⁴`, consistent with accumulated
  floating-point rounding over `n!` additions of numbers that can differ
  in magnitude.

**What the plot shows (top-left panel).**
Both the BG sum and the Cole–Hopf formula are plotted on a log scale.  They
overlap exactly by construction — the point is to confirm the code produces
identical numbers.  A twin right axis (purple triangles) shows the relative
error per `n`; it sits at machine precision throughout, confirming the identity
holds to full double precision.

---

## Test B — n=2 analytic formula (non-abelian, transverse)

**What it checks.**
For `n = 2` with transverse polarizations `ε_i = k_i^⊥ = (−k_{iy}, k_{ix})`,
the non-abelian vertex reduces to

    J[0,1]  =  (k₁ × k₂)(ε₁ + ε₂) / (2 k₁·k₂)

where `×` is the 2D cross product `a_x b_y − a_y b_x`.

**Derivation.**
`s₁₂ = 2 k₁·k₂`.  The vertex is
`(ε₁·k₂) ε₂ − (ε₂·k₁) ε₁`.  Because `ε_i ⊥ k_i` in 2D, the dot products
become `ε_i · k_j = ±(k_i × k_j)`, and both terms contribute a factor of
`(k₁ × k₂)`, giving `(k₁ × k₂)(ε₁ + ε₂)`.

**Implementation.**
40 random pairs `(k₁, k₂)` are drawn (seed 1); trials where `|k₁·k₂| < 10⁻¹⁰`
(near-null propagator) are skipped.  `bg_nonabelian` is called and the result
is compared to the analytic formula.

**Honest caveats.**
- This is a single-step check: the recursion only needs to perform one
  sum over one split, so it is the weakest possible test of the multi-step
  machinery.
- 40 trials with a fixed seed is not a statistical sweep; it demonstrates
  absence of an obvious coding error rather than robustness across the full
  parameter space.
- Trials with near-zero propagator denominators are silently dropped; their
  behavior is untested.

**What the plot shows (top-right panel).**
Relative errors for all 40 trials on a log scale, with the pass threshold
`10⁻¹²` shown as a dashed red line.  All points cluster around `10⁻¹⁵`,
two to three orders of magnitude below the threshold, indicating the analytic
formula is reproduced at essentially machine precision.

---

## Test C — Transversality preservation (non-abelian, transverse)

**What it checks.**
The non-abelian Lie-bracket vertex satisfies `K_P · J_P = 0` exactly at
every order, analogous to incompressibility in Navier–Stokes.  We measure
the ratio

    |u_∥ / u_⊥|  =  |J · K̂_P| / |J · K̂_P^⊥|

for `n = 2,…,40` using a single fixed random instance (seed 2, 40 Gaussian
momenta scaled by 2).

**Implementation.**
`bg_nonabelian` is called once for all `n ≤ 40`; then for each `n` the
current `J[0, n−1]` is projected onto `K̂_P` and its perpendicular.  The
test passes if the ratio stays below `10⁻⁴`.

**Honest caveats.**
- A single random seed is used.  For adversarial choices of momenta (e.g.
  nearly collinear or with a near-zero propagator at some intermediate step)
  numerical cancellations could in principle push the ratio higher.
- The ratio at low `n` is literally zero or `O(10⁻¹⁶)` — this is expected,
  because the `n = 2` result is exact and rounding errors have not yet had
  chance to accumulate.
- At `n = 40` the ratio is still `~10⁻¹⁶`, showing no evidence of error
  growth for this instance.  Whether this persists to much larger `n` or
  for pathological momenta is not checked here.

**What the plot shows (bottom-left panel).**
`|u_∥ / u_⊥|` on a log scale from `n = 2` to `40`.  The curve is flat at
machine precision throughout, well below the `10⁻⁴` threshold (dashed red).
Note that some points show exactly zero before the log is taken; they appear
at the bottom of the axis.

---

## Test D — Longitudinal plateau vs transverse growth

**What it checks.**
Cole–Hopf integrability predicts:
- **Longitudinal** (`ε = k`): the abelian BG current is exactly
  `k_n / 2^{n−1}`, bounded in magnitude.  For the non-abelian recursion the
  argument is less direct, but the Lie-bracket vertex also preserves the
  integrable structure, so `log₁₀|J|²` should remain `O(1)`.
- **Transverse** (`ε = k⊥`): no integrable structure; currents grow
  geometrically with `n`, giving a linear trend in `log₁₀|J|²`.

**Implementation.**
`n_max = 100` Kolmogorov-shell momenta (`|k| = 1`, random angles, seed 42)
are generated.  `bg_nonabelian` is called twice — once with longitudinal,
once with transverse polarizations.  `log₁₀|J[0, n−1]|²` is computed for
each `n`, and a linear fit over the last 20 points gives the asymptotic slope.

**Honest caveats.**
- Results depend on the specific random seed.  The Kolmogorov-shell
  configuration is physically motivated (single-scale forcing) but not
  representative of all possible inputs.
- "Bounded" for the longitudinal curve means the slope is near zero on
  average, not that the current is literally bounded: the curve oscillates
  (it reaches `log₁₀|J|² ≈ 12` at `n = 90` before returning to `~9` at
  `n = 100`).  This is consistent with a convergent but not monotone series.
- The pass threshold `|slope_long| < 0.5` is generous.  For this seed the
  slope is `−0.17`, comfortably inside the bound.
- The transverse slope `+1.21` means `|J|² ≈ 10^{1.21 n}` at large `n`,
  implying a radius of convergence of roughly `|z| < 10^{−0.6} ≈ 0.25` in
  a generating-function sense.  This is consistent with (but does not prove)
  finite radius of convergence.

**What the plot shows (bottom-right panel).**
`log₁₀|J|²` vs `n` for both polarizations.  The blue (longitudinal) curve
wanders near zero, while the red (transverse) curve climbs linearly, reaching
≈111 at `n = 100`.  The dashed horizontal line marks `log₁₀ = 0` for
reference.  The divergence of the two curves is the clearest visual signature
of the Cole–Hopf integrability contrast.

---

## Summary of what these tests do and do not establish

| Test | What it establishes | What it does not establish |
|------|--------------------|-----------------------------|
| A | Cole–Hopf identity holds to machine precision for `n ≤ 7`, one random instance | Identity at large `n`; behavior for degenerate momenta |
| B | `n=2` analytic formula reproduced at machine precision, 40 trials | Any `n > 2` behavior; near-zero propagator cases |
| C | Transversality at `~10⁻¹⁶` for one instance, `n ≤ 40` | Behavior for adversarial momenta; large `n` accumulation |
| D | Stark longitudinal/transverse contrast for Kol-shell, seed 42, `n ≤ 100` | Universality over seeds; rigorous convergence bounds |

Taken together, the tests provide strong numerical evidence that the
implementation is correct and that the Cole–Hopf/integrability structure
is faithfully reproduced.  They do not constitute a proof of any of the
analytic claims; those are established in the theory notes (`burgers_100pt_note.pdf`).
