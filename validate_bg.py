"""
Numerical validation of the 2D Burgers Berends-Giele recursion.

Four tests
----------
A  Cole–Hopf formula (exact, small n)
     Symmetrize the abelian BG current over all n! orderings and compare to
     the exact Cole–Hopf prediction (eq. 5 of the notes):
         Σ_{σ ∈ S_n} J_abelian(k_{σ₁},…,k_{σₙ})  =  K_P · (n−1)! / 2^{n−1}
     for longitudinal polarizations ε_i = k_i.  Tested for n = 2 … 7.

B  n=2 analytic formula (non-abelian, transverse)
     For two arbitrary momenta and transverse polarizations ε_i = k_i^⊥:
         J[0,1]  =  (k₁ × k₂)(ε₁ + ε₂) / (2 k₁·k₂)
     where × is the 2D cross product.

C  Transversality preservation (non-abelian, transverse)
     The Lie-bracket vertex preserves K_P · J = 0 order by order, mirroring
     incompressibility in Navier–Stokes.  We check |u_∥/u_⊥| ≪ 1.

D  Longitudinal plateau vs transverse growth (Cole–Hopf integrability)
     Longitudinal: log10|J|² stays O(1) bounded → convergent series.
     Transverse:   log10|J|² grows ~linearly with n → finite radius of convergence.

Usage
-----
    python validate_bg.py              # run all tests, print results
    python validate_bg.py --plot       # also save validate_plots.png
    python validate_bg.py --n-max 50   # use n=50 for Test D
"""

import argparse
import numpy as np
from itertools import permutations
from math import factorial

from burgers_100pt_1 import bg_abelian, bg_nonabelian


# ── small helpers ─────────────────────────────────────────────────────────────

def cross2d(a, b):
    """2D cross product: a_x b_y − a_y b_x."""
    return float(a[0]*b[1] - a[1]*b[0])

GREEN = "\033[32m"
RED   = "\033[31m"
RESET = "\033[0m"

def status(ok):
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# Test A — Cole–Hopf formula via full symmetrization of the abelian current
# ══════════════════════════════════════════════════════════════════════════════

def test_A(n_max=7, seed=0):
    """
    Exact identity (proved via Cole–Hopf integrability):
        Σ_{σ ∈ S_n} J_abelian(k_{σ₁},…,k_{σₙ})  =  K_P · (n−1)! / 2^{n−1}

    Why it works: for longitudinal ε_i = k_i the abelian BG current for any
    fixed ordering (1,…,n) equals k_n / 2^{n−1} — only the last momentum
    survives.  Summing over all n! orderings gives each k_i in the last slot
    exactly (n−1)! times, yielding (n−1)! K_P / 2^{n−1}.
    """
    print("=" * 66)
    print("Test A — Cole–Hopf formula  (abelian BG, longitudinal ε=k)")
    print("  Σ_σ J_abelian(σ)  =  K_P · (n−1)! / 2^{n−1}")
    print("=" * 66)

    rng      = np.random.RandomState(seed)
    passed   = True
    ns_A     = []
    vals_bg  = []
    vals_ch  = []
    rel_errs = []

    for n in range(2, n_max + 1):
        momenta = rng.randn(n, 2)          # generic random momenta
        K_P     = momenta.sum(axis=0)

        # sum abelian BG over all n! orderings with ε_i = k_i
        total = np.zeros(2)
        for perm in permutations(range(n)):
            mom_p = momenta[list(perm)]
            J, _, _ = bg_abelian(n, mom_p, mom_p.copy())
            total += J[0, n-1]

        expected = K_P * factorial(n - 1) / 2**(n - 1)
        norm_e   = np.linalg.norm(expected)
        rel_err  = (np.linalg.norm(total - expected)
                    / (norm_e + 1e-300))
        good     = rel_err < 1e-9
        passed   = passed and good

        ns_A.append(n)
        vals_bg.append(float(np.linalg.norm(total)))
        vals_ch.append(float(norm_e))
        rel_errs.append(float(rel_err))

        print(f"  n={n:2d}  |BG sum| = {np.linalg.norm(total):9.4e}"
              f"  |formula| = {norm_e:9.4e}"
              f"  rel_err = {rel_err:.1e}  {status(good)}")

    print()
    return passed, ns_A, vals_bg, vals_ch, rel_errs


# ══════════════════════════════════════════════════════════════════════════════
# Test B — n=2 analytic formula, non-abelian transverse
# ══════════════════════════════════════════════════════════════════════════════

def test_B(n_trials=40, seed=1):
    """
    For n=2 with transverse polarizations ε_i = (−k_{iy}, k_{ix}):

        J[0,1]  =  (k₁ × k₂)(ε₁ + ε₂) / (2 k₁·k₂)

    Derivation:
      s₁₂ = 2 k₁·k₂
      vertex = (ε₁·k₂) ε₂ − (ε₂·k₁) ε₁
             = (k₁×k₂) ε₂ + (k₁×k₂) ε₁     [since ε_i·k_j = ±(k_i×k_j)]
             = (k₁×k₂)(ε₁ + ε₂)
    """
    print("=" * 66)
    print("Test B — n=2 analytic formula  (non-abelian, transverse ε=k⊥)")
    print("  J[0,1] = (k₁×k₂)(ε₁+ε₂) / (2 k₁·k₂)")
    print("=" * 66)

    rng      = np.random.RandomState(seed)
    max_err  = 0.0
    passed   = True
    trial_ids = []
    b_rel_errs = []

    for trial in range(n_trials):
        k1 = rng.randn(2);  k2 = rng.randn(2)
        dot = float(np.dot(k1, k2))
        if abs(dot) < 1e-10:
            continue
        eps1 = np.array([-k1[1], k1[0]])
        eps2 = np.array([-k2[1], k2[0]])

        J, _, _ = bg_nonabelian(2, np.array([k1, k2]), np.array([eps1, eps2]))
        computed = J[0, 1]

        expected = cross2d(k1, k2) * (eps1 + eps2) / (2.0 * dot)
        rel_err  = (np.linalg.norm(computed - expected)
                    / (np.linalg.norm(expected) + 1e-300))
        max_err  = max(max_err, rel_err)
        trial_ids.append(trial)
        b_rel_errs.append(float(rel_err))
        if rel_err > 1e-12:
            passed = False
            print(f"  trial {trial:3d}: FAIL  rel_err = {rel_err:.2e}")

    print(f"  {n_trials} random trials,  max rel_err = {max_err:.2e}  {status(passed)}")
    print()
    return passed, trial_ids, b_rel_errs


# ══════════════════════════════════════════════════════════════════════════════
# Test C — transversality preservation
# ══════════════════════════════════════════════════════════════════════════════

def test_C(n_max=40, seed=2):
    """
    The non-abelian Lie-bracket vertex satisfies K_P · J_P = 0 at every order,
    so the n-point current stays transverse to the total momentum.
    This is the Burgers analogue of incompressibility in Navier–Stokes.

    At low n the ratio |u_∥/u_⊥| is at machine precision (< 10⁻¹²).
    Numerical errors accumulate at high n; the test passes if the ratio
    stays below 10⁻⁴ throughout.
    """
    print("=" * 66)
    print("Test C — Transversality  (non-abelian, transverse ε=k⊥)")
    print("  K_P · J[0,n−1] = 0  at every order n")
    print("=" * 66)

    rng     = np.random.RandomState(seed)
    momenta = rng.randn(n_max, 2) * 2.0
    pols    = np.column_stack([-momenta[:, 1], momenta[:, 0]])

    J, K, _ = bg_nonabelian(n_max, momenta, pols)

    max_ratio = 0.0
    passed    = True
    ns_C      = []
    c_ratios  = []

    print(f"  {'n':>4}   {'|u_∥ / u_⊥|':>13}")
    print(f"  {'-'*22}")

    for n in range(2, n_max + 1):
        u   = J[0, n-1]
        KP  = K[0, n-1]
        KP2 = float(np.dot(KP, KP))
        u2  = float(np.dot(u, u))
        if KP2 < 1e-14 or u2 < 1e-300:
            continue
        KPhat  = KP / np.sqrt(KP2)
        KPperp = np.array([-KPhat[1], KPhat[0]])
        u_par  = abs(float(np.dot(u, KPhat)))
        u_perp = abs(float(np.dot(u, KPperp)))
        ratio  = u_par / (u_perp + 1e-300)
        max_ratio = max(max_ratio, ratio)
        ns_C.append(n)
        c_ratios.append(float(ratio))
        bad    = ratio > 1e-4
        if bad:
            passed = False
        if n <= 10 or n % 5 == 0:
            print(f"  {n:4d}   {ratio:13.2e}   {status(not bad)}")

    print(f"\n  Max |u_∥/u_⊥| over n=2..{n_max}: {max_ratio:.2e}  {status(passed)}")
    print()
    return passed, ns_C, c_ratios


# ══════════════════════════════════════════════════════════════════════════════
# Test D — longitudinal plateau vs transverse growth
# ══════════════════════════════════════════════════════════════════════════════

def test_D(n_max=100, seed=42):
    """
    Cole–Hopf integrability predicts:
      Longitudinal (ε=k):  |J_{1..n}|² stays bounded → slope ≈ 0
      Transverse   (ε=k⊥): |J_{1..n}|² grows ~ 10^n  → slope ≈ +1

    Using the Kolmogorov-shell configuration (all |k_p|=1, random angles).
    Expected values from the notes: slope_long ≈ −0.17, slope_trans ≈ +1.21.
    """
    print("=" * 66)
    print("Test D — Longitudinal plateau vs transverse growth")
    print(f"  Kolmogorov shell, non-abelian,  n up to {n_max}")
    print("=" * 66)

    rng     = np.random.RandomState(seed)
    angles  = rng.uniform(0, 2*np.pi, n_max)
    momenta = np.column_stack([np.cos(angles), np.sin(angles)])  # |k|=1

    eps_long  = momenta.copy()
    eps_trans = np.column_stack([-momenta[:, 1], momenta[:, 0]])

    J_long,  _, _ = bg_nonabelian(n_max, momenta, eps_long)
    J_trans, _, _ = bg_nonabelian(n_max, momenta, eps_trans)

    ns        = np.arange(2, n_max + 1)
    log_long  = np.array([
        np.log10(max(float(np.dot(J_long[0, n-1],  J_long[0, n-1])),  1e-300))
        for n in ns])
    log_trans = np.array([
        np.log10(max(float(np.dot(J_trans[0, n-1], J_trans[0, n-1])), 1e-300))
        for n in ns])

    print(f"\n  {'n':>4}  {'log10|J_long|²':>15}  {'log10|J_trans|²':>16}")
    print(f"  {'-'*42}")
    for i, n in enumerate(ns):
        if n <= 10 or n % 10 == 0:
            print(f"  {n:4d}  {log_long[i]:15.2f}  {log_trans[i]:16.2f}")

    # asymptotic slopes over last 20 points
    mask = ns >= n_max - 20
    slope_long  = float(np.polyfit(ns[mask], log_long[mask],  1)[0])
    slope_trans = float(np.polyfit(ns[mask], log_trans[mask], 1)[0])

    long_ok  = abs(slope_long)  < 0.5   # near-zero → bounded
    trans_ok = slope_trans      > 0.5   # clearly positive → growing

    print(f"\n  Asymptotic slopes (last 20 points):")
    print(f"    Longitudinal : {slope_long:+.3f} dec/step  (expect ≈ 0)   {status(long_ok)}")
    print(f"    Transverse   : {slope_trans:+.3f} dec/step  (expect > 0.5) {status(trans_ok)}")
    print(f"\n  Note: log10|J_long(n={n_max})|² = {log_long[-1]:.1f}  (bounded O(1))")
    print(f"        log10|J_trans(n={n_max})|² = {log_trans[-1]:.1f}  (geometric growth)")
    print()

    passed = long_ok and trans_ok
    return passed, ns, log_long, log_trans


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_plots(ns_A, vals_bg, vals_ch, a_rel_errs,
               trial_ids, b_rel_errs,
               ns_C, c_ratios,
               ns_D, log_long, log_trans,
               fname="validate_plots.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── Test A: BG sum vs Cole–Hopf magnitudes + relative error ──────────
    ax = axes[0, 0]
    ax.semilogy(ns_A, vals_bg, "o-",  lw=2, ms=8, color="steelblue",
                label=r"$\sum_\sigma J_{\rm abelian}(\sigma)$  [BG sum]", zorder=5)
    ax.semilogy(ns_A, vals_ch, "s--", lw=2, ms=7, color="tomato", alpha=0.85,
                label=r"$(n{-}1)!\,|K_P|/2^{n-1}$  [Cole–Hopf]", zorder=4)
    ax.set_xlabel("n (number of legs)", fontsize=11)
    ax.set_ylabel(r"$|\,\Sigma\,J\,|$", fontsize=11)
    ax.set_title("Test A: Cole–Hopf formula\n"
                 r"(abelian BG, longitudinal $\varepsilon=k$)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # inset: relative errors on a second y-axis
    ax2 = ax.twinx()
    ax2.semilogy(ns_A, a_rel_errs, "^:", lw=1.2, ms=7, color="purple",
                 alpha=0.7, label="rel. error")
    ax2.set_ylabel("relative error", fontsize=10, color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")
    ax2.legend(fontsize=9, loc="lower right")

    # ── Test B: per-trial relative errors ────────────────────────────────
    ax = axes[0, 1]
    ax.semilogy(trial_ids, b_rel_errs, "o", ms=5, color="steelblue",
                alpha=0.7, label="rel. error per trial")
    threshold = 1e-12
    ax.axhline(threshold, color="tomato", lw=1.5, ls="--",
               label=f"pass threshold ({threshold:.0e})")
    ax.set_xlabel("trial index", fontsize=11)
    ax.set_ylabel("relative error", fontsize=11)
    ax.set_title("Test B: $n=2$ analytic formula\n"
                 r"(non-abelian, transverse $\varepsilon=k^\perp$)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ── Test C: transversality ratio |u_∥/u_⊥| vs n ──────────────────────
    ax = axes[1, 0]
    ax.semilogy(ns_C, c_ratios, "-", lw=1.8, color="steelblue",
                label=r"$|u_\parallel / u_\perp|$")
    ax.axhline(1e-4, color="tomato", lw=1.5, ls="--", label="threshold $10^{-4}$")
    ax.set_xlabel("n (number of legs)", fontsize=11)
    ax.set_ylabel(r"$|u_\parallel / u_\perp|$", fontsize=11)
    ax.set_title("Test C: Transversality preservation\n"
                 r"(non-abelian, $\varepsilon=k^\perp$)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # ── Test D: longitudinal plateau vs transverse growth ────────────────
    ax = axes[1, 1]
    ax.plot(ns_D, log_long,  "-", lw=2, color="steelblue",
            label="Longitudinal  (Cole–Hopf integrable)")
    ax.plot(ns_D, log_trans, "-", lw=2, color="tomato",
            label="Transverse  (non-integrable)")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_xlabel("n (number of legs)", fontsize=11)
    ax.set_ylabel(r"$\log_{10}|J_{1\ldots n}|^2$", fontsize=11)
    ax.set_title("Test D: Longitudinal plateau\nvs transverse growth", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fname}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the 2D Burgers BG recursion (4 numerical tests)")
    parser.add_argument("--plot", action="store_true",
                        help="save figures to validate_plots.png")
    parser.add_argument("--n-max", type=int, default=100,
                        help="max n for Test D  (default: 100)")
    args = parser.parse_args()

    ok_A, ns_A, vals_bg, vals_ch, a_rel_errs = test_A()
    ok_B, trial_ids, b_rel_errs              = test_B()
    ok_C, ns_C, c_ratios                     = test_C()
    ok_D, ns_D, log_long, log_trans          = test_D(n_max=args.n_max)

    if args.plot:
        print("Saving plots …")
        make_plots(ns_A, vals_bg, vals_ch, a_rel_errs,
                   trial_ids, b_rel_errs,
                   ns_C, c_ratios,
                   ns_D, log_long, log_trans)

    print("=" * 66)
    print("SUMMARY")
    print("=" * 66)
    rows = [
        ("A", "Cole–Hopf formula  (abelian sum, n≤7)",         ok_A),
        ("B", "n=2 analytic       (non-abelian, transverse)",   ok_B),
        ("C", "Transversality     (non-abelian, ε=k⊥, n≤40)",  ok_C),
        ("D", "Longitudinal plateau vs transverse growth",      ok_D),
    ]
    for key, desc, result in rows:
        print(f"  Test {key}  {desc:<46} {status(result)}")
    print()
    overall = all(r for _, _, r in rows)
    print(f"  {'ALL PASS' if overall else 'SOME TESTS FAILED'}")
    print()
