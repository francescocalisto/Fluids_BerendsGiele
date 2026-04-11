"""
2D Burgers: ordered Berends-Giele recursion to n points.

O(n^3) algorithm using contiguous subwords.

Non-abelian vertex (antisymmetric / Lie bracket):
  s_P u_{i,P} = Σ_{P=QR} [(u_Q · K_R) u_{i,R} − (u_R · K_Q) u_{i,Q}]

Abelian vertex (asymmetric, Q advects R):
  s_P u_{i,P} = Σ_{P=QR} (u_Q · K_R) u_{i,R}

Propagator denominator:
  s_P = |K_P|^2 − Σ_{p∈P} |k_p|^2 = 2 Σ_{p<q} k_p · k_q

Usage:
  python burgers_100pt_1.py              # run all 6 configs at n=100
  python burgers_100pt_1.py --n-max 50   # stop at n=50
  python burgers_100pt_1.py --plot       # also save bg_growth_rates.png
"""

import argparse
import numpy as np
from time import time


# ── Core recursions ───────────────────────────────────────────────────────────

def precompute(momenta):
    """Return K[i,j] (partial momentum sums) and s[i,j] (propagator denominators)."""
    n = len(momenta)
    cum    = np.zeros((n + 1, 2))
    cum_k2 = np.zeros(n + 1)
    for i in range(n):
        cum[i+1]    = cum[i] + momenta[i]
        cum_k2[i+1] = cum_k2[i] + np.dot(momenta[i], momenta[i])
    K = np.zeros((n, n, 2))
    s = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = cum[j+1] - cum[i]
            s[i, j] = np.dot(K[i, j], K[i, j]) - (cum_k2[j+1] - cum_k2[i])
    return K, s


def bg_nonabelian(n, momenta, polarizations):
    """
    Non-abelian (Lie-bracket) BG recursion.
    Returns J[i,j], K[i,j], s[i,j].
    J[0, n-1] is the n-point current.
    """
    K, s = precompute(momenta)
    J = np.zeros((n, n, 2))
    for i in range(n):
        J[i, i] = polarizations[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j  = i + length - 1
            sp = s[i, j]
            if abs(sp) < 1e-30:
                continue
            res = np.zeros(2)
            for m in range(i, j):
                uQ = J[i, m];   uR = J[m+1, j]
                kQ = K[i, m];   kR = K[m+1, j]
                c1 = uQ[0]*kR[0] + uQ[1]*kR[1]   # u_Q · K_R
                c2 = uR[0]*kQ[0] + uR[1]*kQ[1]   # u_R · K_Q
                res[0] += c1*uR[0] - c2*uQ[0]
                res[1] += c1*uR[1] - c2*uQ[1]
            J[i, j] = res / sp
    return J, K, s


def bg_abelian(n, momenta, polarizations):
    """
    Abelian (one-sided advection) BG recursion.
    The full unordered current is recovered by summing over all n! permutations.
    """
    K, s = precompute(momenta)
    J = np.zeros((n, n, 2))
    for i in range(n):
        J[i, i] = polarizations[i]
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j  = i + length - 1
            sp = s[i, j]
            if abs(sp) < 1e-30:
                continue
            res = np.zeros(2)
            for m in range(i, j):
                uQ = J[i, m];   uR = J[m+1, j]
                kR = K[m+1, j]
                c1 = uQ[0]*kR[0] + uQ[1]*kR[1]   # u_Q · K_R
                res[0] += c1*uR[0]
                res[1] += c1*uR[1]
            J[i, j] = res / sp
    return J, K, s


# ── Momentum configurations ──────────────────────────────────────────────────

def kolmogorov_shell(n, k_f=1.0, seed=42):
    """All |k_p| = k_f with random angles (forcing at a single scale)."""
    rng    = np.random.RandomState(seed)
    angles = rng.uniform(0, 2*np.pi, n)
    return k_f * np.column_stack([np.cos(angles), np.sin(angles)])

def inertial_range(n, k_min=1.0, k_max=10.0, seed=42):
    """|k_p| log-uniform in [k_min, k_max] with random angles."""
    rng   = np.random.RandomState(seed)
    log_k = rng.uniform(np.log(k_min), np.log(k_max), n)
    angles = rng.uniform(0, 2*np.pi, n)
    k = np.exp(log_k)
    return np.column_stack([k*np.cos(angles), k*np.sin(angles)])

def kolmogorov_cascade(n, k_f=1.0, seed=42):
    """|k_p| on a log grid from k_f to 100 k_f."""
    rng    = np.random.RandomState(seed)
    k_vals = k_f * np.logspace(0, 2, n)
    angles = rng.uniform(0, 2*np.pi, n)
    return np.column_stack([k_vals*np.cos(angles), k_vals*np.sin(angles)])

def lattice_modes(n, k_max=5, seed=42):
    """Integer momenta drawn uniformly from [−k_max, k_max]^2 \ {0}."""
    rng = np.random.RandomState(seed)
    mom = np.zeros((n, 2))
    for i in range(n):
        while True:
            kx, ky = rng.randint(-k_max, k_max + 1, 2)
            if kx != 0 or ky != 0:
                break
        mom[i] = [kx, ky]
    return mom

def transverse_pol(momenta):
    """ε_p = k_p^⊥ = (−k_{py}, k_{px}): vortical, non-integrable sector."""
    return np.column_stack([-momenta[:, 1], momenta[:, 0]])

def longitudinal_pol(momenta):
    """ε_p = k_p: potential flow, Cole–Hopf integrable sector."""
    return momenta.copy()


# ── Driver ───────────────────────────────────────────────────────────────────

def run(n_max, mom_func, pol_func, rec_func, label, **kw):
    """Run the BG recursion and print a table of results."""
    print(f"\n{'='*72}")
    print(f"  {label}  (n_max={n_max})")
    print(f"{'='*72}")

    momenta = mom_func(n_max, **kw)
    pols    = pol_func(momenta)

    t0 = time()
    J, K, s = rec_func(n_max, momenta, pols)
    dt = time() - t0
    print(f"  Computed in {dt:.3f} s")

    # Column header
    print(f"\n{'n':>5} {'log10|J|²':>11} {'|J|':>12} {'u_∥':>12} {'u_⊥':>12} {'s_P':>10}")
    print("-" * 65)

    results = []
    for n in range(2, n_max + 1):
        u   = J[0, n-1]
        u2  = float(np.dot(u, u))
        KP  = K[0, n-1]
        KP2 = float(np.dot(KP, KP))
        if KP2 > 1e-14:
            KPmag = np.sqrt(KP2)
            KPhat  = KP / KPmag
            KPperp = np.array([-KPhat[1], KPhat[0]])
            u_par  = float(np.dot(u, KPhat))    # true longitudinal component
            u_perp = float(np.dot(u, KPperp))   # true transverse component
        else:
            u_par = u_perp = 0.0
        sp      = s[0, n-1]
        log10u2 = np.log10(abs(u2)) if abs(u2) > 1e-300 else -300.0

        results.append(dict(n=n, u2=u2, log10u2=log10u2,
                            u_par=u_par, u_perp=u_perp, sP=sp))

        do_print = (n <= 15) or (n <= 50 and n % 5 == 0) or (n % 10 == 0)
        if do_print:
            print(f"{n:5d} {log10u2:11.3f} {np.sqrt(abs(u2)):12.4e} "
                  f"{u_par:12.4e} {u_perp:12.4e} {sp:10.2f}")

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_growth_rates(all_results, labels, fname="bg_growth_rates.png"):
    """Save a figure showing log10|J_{1..n}|^2 vs n for all configurations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    colors = ['tab:blue', 'tab:orange', 'tab:green',
              'tab:red',  'tab:purple', 'tab:brown']
    styles = ['-', '-', '-', '--', '-', '--']

    fig, ax = plt.subplots(figsize=(9, 5))
    for r, lb, c, ls in zip(all_results, labels, colors, styles):
        ns = [d['n'] for d in r]
        ys = [d['log10u2'] for d in r]
        ax.plot(ns, ys, color=c, ls=ls, lw=1.8, label=lb)

    ax.set_xlabel("n (number of legs)", fontsize=12)
    ax.set_ylabel(r"$\log_{10}|J_{1\ldots n}|^2$", fontsize=12)
    ax.set_title("2D Burgers BG currents — growth with multiplicity", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"\n  Saved plot: {fname}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="2D Burgers BG recursion to n points (O(n³))")
    parser.add_argument("--n-max", type=int, default=100,
                        help="maximum number of legs (default: 100)")
    parser.add_argument("--plot", action="store_true",
                        help="save growth-rate figure to bg_growth_rates.png")
    args = parser.parse_args()
    N = args.n_max

    r1 = run(N, kolmogorov_shell,  transverse_pol,   bg_nonabelian,
             "NON-ABELIAN | TRANSVERSE  | Kolmogorov shell  |k|=1",
             k_f=1.0, seed=42)
    r2 = run(N, inertial_range,    transverse_pol,   bg_nonabelian,
             "NON-ABELIAN | TRANSVERSE  | Inertial range [1,10]",
             k_min=1.0, k_max=10.0, seed=42)
    r3 = run(N, lattice_modes,     transverse_pol,   bg_nonabelian,
             "NON-ABELIAN | TRANSVERSE  | Lattice modes",
             seed=42)
    r4 = run(N, kolmogorov_shell,  transverse_pol,   bg_abelian,
             "ABELIAN     | TRANSVERSE  | Kolmogorov shell",
             k_f=1.0, seed=42)
    r5 = run(N, kolmogorov_shell,  longitudinal_pol, bg_nonabelian,
             "NON-ABELIAN | LONGITUDINAL| Kolmogorov shell  (Cole–Hopf check)",
             k_f=1.0, seed=42)
    r6 = run(N, kolmogorov_cascade, transverse_pol,  bg_nonabelian,
             "NON-ABELIAN | TRANSVERSE  | Kolmogorov cascade",
             k_f=1.0, seed=42)

    all_r  = [r1, r2, r3, r4, r5, r6]
    labels = ["NA/T/Kol", "NA/T/IR", "NA/T/Lat",
              "AB/T/Kol", "NA/L/Kol", "NA/T/Cas"]

    # ── Summary table ──
    print(f"\n{'='*72}")
    print("  SUMMARY: log10|J_{1..n}|²")
    print(f"{'='*72}")
    hdr = f"{'n':>5}" + "".join(f" {lb:>11}" for lb in labels)
    print(hdr)
    print("-" * (5 + 12*len(labels)))
    for idx in range(len(r1)):
        n = r1[idx]["n"]
        do_print = (n <= 20) or (n <= 50 and n % 5 == 0) or (n % 10 == 0)
        if do_print:
            row = f"{n:5d}" + "".join(f" {r[idx]['log10u2']:11.2f}" for r in all_r)
            print(row)

    # ── Asymptotic growth rates ──
    cutoff = max(80, N - 20)
    print(f"\n  Asymptotic slopes (linear fit over n={cutoff}..{N}):")
    for lb, r in zip(labels, all_r):
        xs = [d["n"]      for d in r if d["n"] >= cutoff and abs(d["u2"]) > 1e-300]
        ys = [d["log10u2"] for d in r if d["n"] >= cutoff and abs(d["u2"]) > 1e-300]
        if len(xs) >= 5:
            slope, _ = np.polyfit(xs, ys, 1)
            print(f"    {lb:12s}: {slope:+.4f} dec/step,  "
                  f"log10|J_{N}|² = {r[-1]['log10u2']:.1f}")
        else:
            print(f"    {lb:12s}: insufficient data")

    if args.plot:
        plot_growth_rates(all_r, labels)
