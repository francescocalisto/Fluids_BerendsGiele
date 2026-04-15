"""
1D Burgers velocity field: perturbative sum truncated at n=N_max.

From Cheung's notes, eq (16):

u(t,p) = Σ_{n=1}^{N} exp(-ν Σ p_i² t) δ(p - Σ p_i) 
         × (1/n!) ∫ J_n(p1,...,pn) Π f(p_i) dp_i

with J_n = (-i/2ν)^{n-1} (n-1)! × (Σ p_i) / (Π p_i)

The exact solution via Cole-Hopf:
  u = -2ν ∂_x log θ,   θ(t,x) = ∫ exp[-(x-y)²/(4νt) + φ(y)/(2ν)] dy / √(4πνt)

where u(0,x) = -∂_x φ(x), i.e., φ(x) = -∫₀ˣ u(0,y) dy.

APPROACH: 
- Discretize momenta on a grid p_j = j Δp for j = -M,...,M
- Choose a specific initial condition f(p) (Fourier transform of u(0,x))
- Compute each n-point contribution by Monte Carlo integration over the 
  n momenta {p_1,...,p_n} with the constraint Σp_i = p
- Sum n=1 to N_max=100
- Compare with exact Cole-Hopf solution

SIMPLER APPROACH (avoiding high-dimensional integrals):
The perturbative expansion is really the Taylor expansion of the Cole-Hopf
solution in powers of the initial amplitude ε. Write f(p) = ε g(p) and 
expand u = Σ ε^n u_n. Then:

u_n(t,p) = ∫ dp_1...dp_n δ(p-Σp_i) × J_n(p1,...,pn) × Π g(p_i) × exp(-ν Σ p_i² t) / n!

For a DISCRETE momentum grid with M modes, this integral becomes a 
discrete convolution. For n=100 modes this is still a sum over M^{99} 
terms — intractable directly.

CORRECT APPROACH: Use the convolution structure.
Define h_1(t,p) = g(p) exp(-νp²t).
Then u_n involves the n-fold convolution of h with itself, weighted by J_n.

But J_n = (-i/2ν)^{n-1} (n-1)! × (Σ p_i)/(Π p_i).

The factor (Σ p_i)/(Π p_i) = Σ_j 1/(Π_{i≠j} p_i) makes this NOT a simple 
convolution — the kernel depends on all momenta individually.

ACTUALLY: let's use the Cole-Hopf directly.
θ = 1 + Σ_n ε^n θ_n, where θ is the solution to the heat equation with 
initial condition θ(0,x) = exp(-φ(x)/(2ν)) ≈ 1 - εφ_1/(2ν) + ε²φ_1²/(8ν²) - ...

Then u = -2ν ∂_x log θ, expanded in ε.

SIMPLEST TRACTABLE APPROACH: 
Pick a FINITE number of discrete momenta (modes), and compute using the 
perturbiner formula directly.

u(t,x) = Σ_{n=1}^{N} (1/n!) Σ_{p1,...,pn ∈ modes} J_n(p1,...,pn) × Π_i (ε a_i) 
          × exp(i(Σp_i)x - ν(Σp_i²)t)

For K discrete modes, each n-point contribution has K^n terms (with repetition).
At K=10, n=100: 10^100 — impossible.

THE RIGHT WAY: Compute the Cole-Hopf solution EXACTLY and compare term by term.

Let's use a single-mode initial condition: f(p) = ε δ(p - p0) for simplicity.
Then u_1 = ε exp(ip0 x - νp0² t), and the n-th order term only involves p_i = p0 
for all i. So:

u_n = (ε^n / n!) J_n(p0,...,p0) exp(i n p0 x - ν n p0² t)

J_n(p0,...,p0) = (-i/2ν)^{n-1} (n-1)! × (n p0)/(p0^n) = (-i/2ν)^{n-1} (n-1)! n / p0^{n-1}

So u_n = ε^n / n! × (-i/2ν)^{n-1} (n-1)! × n/p0^{n-1} × exp(inp0 x - νnp0²t)
       = ε^n (-i/2ν)^{n-1} / p0^{n-1} × exp(inp0 x - νnp0²t)
       = exp(ip0x - νp0²t) × [-iε/(2νp0) × exp(ip0x - νp0²t)]^{n-1}

Defining z = exp(ip0 x - νp0²t), this is:
u_n = z × (-iε/(2νp0))^{n-1} × z^{n-1} = z × [-iε z/(2νp0)]^{n-1}

The sum: u = Σ_{n=1}^∞ z × [-iεz/(2νp0)]^{n-1} = z / (1 + iεz/(2νp0))
         = 2νp0 z / (2νp0 + iεz)

This is the exact answer for the single-mode case! And we can check it 
against Cole-Hopf. But it's boring since it's just a geometric series.

BETTER: Use TWO modes and compute the full perturbative sum.
f(p) = ε₁ δ(p-p1) + ε₂ δ(p-p2)

Then the n-th order term involves all partitions of n into (k, n-k) where 
k modes have momentum p1 and n-k have momentum p2. This gives a binomial
sum that we can evaluate efficiently.

EVEN BETTER: Use the position-space Cole-Hopf directly.
"""

import numpy as np
from scipy.special import factorial
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# APPROACH: Position-space computation with Cole-Hopf comparison
#
# Initial condition: u(0,x) = -ε ∂_x φ₀(x) with φ₀(x) chosen.
# We pick u(0,x) = ε A sin(kx) (single Fourier mode, real).
#
# Cole-Hopf exact: u(t,x) = -2ν ∂_x log θ(t,x)
# where θ satisfies the heat equation with θ(0,x) = exp(φ₀(x)/(2ν)),
# and u(0,x) = -∂_x φ₀(x), so φ₀(x) = (εA/k) cos(kx).
#
# θ(0,x) = exp[(εA)/(2νk) cos(kx)]
#
# The heat kernel gives:
# θ(t,x) = Σ_n I_n(εA/(2νk)) exp(-νn²k²t) exp(inkx)
# where I_n is the modified Bessel function.
#
# u(t,x) = -2ν ∂_x log θ = -2ν (∂_x θ)/θ
#
# The perturbative expansion in ε:
# θ(0,x) = exp[α cos(kx)] with α = εA/(2νk)
# = Σ_m α^m cos^m(kx) / m!
# = Σ_m α^m / m! × Σ_j binomial terms in e^{ijkx}
#
# Each power of α gives a higher Fourier harmonic.
# The n-th order perturbative term u_n involves n powers of the 
# initial condition, hence Fourier modes up to harmonic n.
# =====================================================================

def exact_cole_hopf(t, x, eps_A, k, nu):
    """
    Exact Cole-Hopf solution for u(0,x) = eps_A * sin(kx).
    
    φ₀(x) = (eps_A/k) cos(kx), so θ(0,x) = exp[eps_A/(2νk) cos(kx)].
    θ(t,x) = Σ_n I_n(α) exp(-ν n² k² t) exp(i n k x)  (n = -∞ to ∞)
    u = -2ν (∂_x θ)/θ
    
    α = eps_A / (2νk)
    """
    from scipy.special import iv as besseli
    
    alpha = eps_A / (2.0 * nu * k)
    
    # Truncate Bessel sum at N_bessel terms
    N_bessel = 200
    
    # Compute θ and ∂_x θ
    theta = np.zeros_like(x, dtype=complex)
    dtheta = np.zeros_like(x, dtype=complex)
    
    for n in range(-N_bessel, N_bessel + 1):
        In = besseli(n, alpha)  # I_n(α) 
        phase = np.exp(1j * n * k * x)
        decay = np.exp(-nu * n**2 * k**2 * t)
        
        theta += In * decay * phase
        dtheta += In * decay * (1j * n * k) * phase
    
    u = -2.0 * nu * dtheta / theta
    return np.real(u)


def perturbative_sum_single_fourier(t, x, eps_A, k, nu, N_max):
    """
    Perturbative expansion of u(t,x) for u(0,x) = eps_A sin(kx).
    
    In Fourier space: f(p) = eps_A/(2i) [δ(p-k) - δ(p+k)]
    
    The n-th order term involves choosing n momenta from {+k, -k}.
    If we choose j momenta = +k and (n-j) momenta = -k, the total 
    momentum is P = (2j-n)k and Σp_i² = nk².
    
    J_n for this configuration:
    J_n = (-i/(2ν))^{n-1} (n-1)! × (2j-n)k / (k^n × (-1)^{n-j})
        = (-i/(2ν))^{n-1} (n-1)! × (2j-n) (-1)^j / k^{n-1}
    
    The amplitude of each configuration:
    - Choosing j of n momenta to be +k: C(n,j) ways
    - Each +k contributes eps_A/(2i), each -k contributes -eps_A/(2i)
    - Product: (eps_A/(2i))^j × (-eps_A/(2i))^{n-j} = (eps_A/(2i))^n × (-1)^{n-j}
    
    So u_n(t,x) = (1/n!) Σ_{j=0}^{n} C(n,j) × J_n(config j) 
                  × (eps_A/(2i))^n × (-1)^{n-j}
                  × exp(i(2j-n)kx - νnk²t)
    
    But J_n = 0 when total momentum P = (2j-n)k = 0, i.e., j = n/2 (n even).
    Also J_n has a pole when any p_i = 0, but since p_i = ±k ≠ 0, we're fine.
    """
    alpha = eps_A / (2.0 * nu * k)
    
    u = np.zeros_like(x, dtype=complex)
    
    for n in range(1, N_max + 1):
        # Sum over j = number of +k momenta (0 to n)
        for j in range(n + 1):
            P = (2*j - n) * k  # total momentum
            
            if P == 0:
                continue  # J_n = 0 when Σp_i = 0
            
            # J_n for this configuration
            # All p_i = ±k, so Π p_i = k^n × (-1)^{n-j}
            # Σ p_i = (2j-n)k
            # J_n = (-i/(2ν))^{n-1} (n-1)! × (2j-n)k / (k^n (-1)^{n-j})
            #      = (-i/(2ν))^{n-1} (n-1)! × (2j-n) (-1)^j / k^{n-1}
            
            Jn = ((-1j) / (2*nu))**(n-1) * factorial(n-1, exact=True) \
                 * (2*j - n) * (-1)**j / k**(n-1)
            
            # Amplitude from initial data
            # (eps_A/(2i))^n × (-1)^{n-j} × C(n,j)
            from scipy.special import comb
            amp = (eps_A / (2j))**n * (-1)**(n-j) * comb(n, j, exact=True)
            
            # Phase and decay
            phase = np.exp(1j * P * x - nu * n * k**2 * t)
            
            # Add contribution (1/n! factor)
            u += (1.0 / factorial(n, exact=True)) * Jn * amp * phase
    
    return np.real(u)


def perturbative_sum_efficient(t, x, eps_A, k, nu, N_max):
    """
    More efficient: use the closed-form J_n and reorganize by Fourier harmonic.
    
    The m-th Fourier harmonic (momentum P = mk) receives contributions from 
    all n ≥ |m|, n ≡ m (mod 2), with j = (n+m)/2.
    
    Contribution to harmonic m from order n:
    c_{n,m} = (1/n!) C(n, (n+m)/2) × J_n × (eps_A/(2i))^n × (-1)^{(n-m)/2}
            × exp(-ν n k² t)
    
    Using J_n = (-i/(2ν))^{n-1} (n-1)! × m / k^{n-1} × (-1)^{(n+m)/2}:
    
    c_{n,m} = (1/n) × C(n, (n+m)/2) × (-i/(2ν))^{n-1} × m/k^{n-1}
              × (-1)^{(n+m)/2} × (eps_A/(2i))^n × (-1)^{(n-m)/2}
              × exp(-νnk²t)
    
    Let me simplify the signs and factors.
    """
    # α = eps_A/(2νk) is the key small parameter
    alpha = eps_A / (2.0 * nu * k)
    
    u = np.zeros_like(x, dtype=complex)
    
    # For each Fourier harmonic m
    M_max = N_max  # maximum harmonic
    
    for m in range(-M_max, M_max + 1):
        if m == 0:
            continue
        
        coeff_m = 0.0 + 0.0j
        
        # Sum over n from |m| to N_max, step 2
        for n in range(abs(m), N_max + 1, 2):
            j = (n + m) // 2
            
            # Binomial coefficient
            from scipy.special import comb
            binom = comb(n, j, exact=True)
            
            # J_n factor: (-i/(2ν))^{n-1} (n-1)! m (-1)^j / k^{n-1}
            # Amplitude: (eps_A/(2i))^n (-1)^{n-j} / n!
            # Combined: (1/n) × binom × (-i/(2ν))^{n-1} × m × (-1)^j / k^{n-1}
            #           × (eps_A/(2i))^n × (-1)^{n-j}
            
            # Let's compute step by step
            # (-i/(2ν))^{n-1} × (eps_A/(2i))^n = (-i)^{n-1} × (eps_A)^n × (-i)^n / ((2ν)^{n-1} × 2^n)
            # = (-i)^{2n-1} × eps_A^n / (2^{2n-1} ν^{n-1})
            # = i × (-1)^n × eps_A^n / (2^{2n-1} ν^{n-1})
            # Hmm, this is getting messy. Let me just use α directly.
            
            # (-i/(2ν))^{n-1} / k^{n-1} = (-i)^{n-1} / (2νk)^{n-1}
            # (eps_A/(2i))^n = eps_A^n / (2i)^n = eps_A^n (-i)^n / 2^n
            # Product: (-i)^{n-1} (-i)^n / (2νk)^{n-1} × eps_A^n / 2^n
            #        = (-i)^{2n-1} × eps_A^n / ((2νk)^{n-1} × 2^n)
            #        = i(-1)^n × eps_A^n / ((2νk)^{n-1} × 2^n)
            #        = i(-1)^n × (eps_A/(2νk))^{n-1} × eps_A / (2νk × 2)
            #        Hmm... let me just use α = eps_A/(2νk).
            #        = i(-1)^n × α^{n-1} × (2νk) × eps_A / (2νk × 2)  -- nope
            
            # Let me just compute numerically.
            factor_Jn = ((-1j) / (2*nu))**(n-1) / k**(n-1) * float(factorial(n-1, exact=True))
            factor_amp = (eps_A / (2j))**n * (-1)**(n-j) * float(binom) / float(factorial(n, exact=True))
            # m comes from Σp_i / (Πp_i / k^n) ... wait, let me redo.
            
            # The total momentum is P = mk.
            # Σp_i = mk, Πp_i = k^n (-1)^{n-j}
            # J_n = (-i/(2ν))^{n-1} (n-1)! × mk / (k^n (-1)^{n-j})
            #      = (-i/(2ν))^{n-1} (n-1)! × m (-1)^j / k^{n-1}
            
            Jn = ((-1j)/(2*nu))**(n-1) * float(factorial(n-1, exact=True)) \
                 * m * (-1)**j / k**(n-1)
            
            # Amplitude from initial data: Π f(p_i) / n!
            # j copies of f(+k) = eps_A/(2i) and (n-j) copies of f(-k) = -eps_A/(2i)
            # × C(n,j) choices × 1/n!
            amp = float(binom) / float(factorial(n, exact=True)) \
                  * (eps_A/(2j))**j * (-eps_A/(2j))**(n-j)
            
            # Decay
            decay = np.exp(-nu * n * k**2 * t)
            
            coeff_m += Jn * amp * decay
        
        u += coeff_m * np.exp(1j * m * k * x)
    
    return np.real(u)


def perturbative_clean(t, x, alpha, k, nu, N_max):
    """
    CLEAN computation using the Bessel function expansion.
    
    The exact solution is:
    θ(t,x) = Σ_m I_m(α) exp(-ν m² k² t) exp(imkx)
    u = -2ν (∂_x θ)/θ
    
    The perturbative expansion in α = ε/(2νk):
    I_m(α) = Σ_{s=0}^∞ (α/2)^{m+2s} / (s! (m+s)!)
    
    So the n-th order in α (= n-th order in ε) comes from terms with m+2s = n.
    
    Let's just expand θ to order α^N and compute u = -2ν ∂_x log θ term by term.
    """
    # θ_approx = Σ_{m=-N}^{N} [Σ_{s: m+2s≤N} (α/2)^{m+2s}/(s!(m+s)!)] exp(-νm²k²t+imkx)
    # 
    # But it's MUCH simpler to note that:
    # θ(0,x) = exp(α cos(kx)) = Σ_{n=0}^∞ α^n cos^n(kx) / n!
    # 
    # cos^n(kx) can be expanded in harmonics using the binomial theorem.
    # Then evolve each harmonic with exp(-νm²k²t).
    #
    # Let's compute θ(t,x) to order α^N:
    # θ(t,x) = Σ_{n=0}^{N} (α^n/n!) Σ_m c_{n,m} exp(-νm²k²t + imkx)
    # where c_{n,m} is the coefficient of e^{imkx} in cos^n(kx).
    #
    # cos^n(kx) = [(e^{ikx}+e^{-ikx})/2]^n = (1/2^n) Σ_{j=0}^n C(n,j) e^{i(2j-n)kx}
    # So c_{n,m} = C(n, (n+m)/2) / 2^n if n≡m (mod 2) and |m|≤n, else 0.
    
    from scipy.special import comb
    
    # Compute θ and ∂_x θ as arrays
    theta = np.zeros_like(x, dtype=complex)
    dtheta_dx = np.zeros_like(x, dtype=complex)
    
    for n in range(0, N_max + 1):
        for m_half in range(-(n//2), n//2 + 1):
            # m = 2*m_half if n even, m = 2*m_half+1 if n odd... 
            # Actually: m runs over values with same parity as n, |m| ≤ n
            pass
        
        for j in range(n + 1):
            m = 2*j - n  # harmonic
            binom = comb(n, j, exact=True)
            
            coeff = (alpha**n / float(factorial(n, exact=True))) * float(binom) / (2.0**n)
            decay = np.exp(-nu * m**2 * k**2 * t)
            phase = np.exp(1j * m * k * x)
            
            theta += coeff * decay * phase
            dtheta_dx += coeff * decay * (1j * m * k) * phase
    
    u = -2.0 * nu * dtheta_dx / theta
    return np.real(u)


# =====================================================================
# MAIN COMPUTATION
# =====================================================================

if __name__ == "__main__":
    
    # Parameters
    nu = 1.0
    k = 1.0
    
    # Spatial grid
    x = np.linspace(0, 2*np.pi, 500)
    
    # Time points
    times = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    
    # Amplitude values (α = ε/(2νk))
    # Small α: perturbation theory converges quickly
    # Large α: need many terms, tests convergence
    alphas = [0.5, 1.0, 2.0, 5.0, 10.0]
    
    fig, axes = plt.subplots(len(alphas), len(times), figsize=(28, 20))
    fig.suptitle('1D Burgers: Perturbative sum (colors) vs exact Cole-Hopf (black dashed)', 
                 fontsize=14, y=0.98)
    
    N_trunc_values = [1, 2, 5, 10, 20, 50, 100]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(N_trunc_values)))
    
    for ai, alpha in enumerate(alphas):
        eps_A = alpha * 2 * nu * k  # eps_A = α × 2νk
        
        for ti, t in enumerate(times):
            ax = axes[ai, ti]
            
            # Exact solution
            u_exact = exact_cole_hopf(t, x, eps_A, k, nu)
            ax.plot(x, u_exact, 'k--', linewidth=2, label='Exact')
            
            # Perturbative truncations
            for ni, N in enumerate(N_trunc_values):
                u_pert = perturbative_clean(t, x, alpha, k, nu, N)
                ax.plot(x, u_pert, color=colors[ni], linewidth=1, 
                        alpha=0.8, label=f'N={N}')
            
            ax.set_title(f'α={alpha}, t={t}', fontsize=9)
            ax.set_xlim(0, 2*np.pi)
            
            if ai == 0 and ti == 0:
                ax.legend(fontsize=6, loc='upper right')
            
            if ai == len(alphas) - 1:
                ax.set_xlabel('x')
            if ti == 0:
                ax.set_ylabel(f'u(t,x)\nα={alpha}')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('/home/claude/burgers_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved convergence plot")
    
    # =====================================================================
    # Quantitative convergence analysis
    # =====================================================================
    
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS: L² error of truncated sum vs exact")
    print("="*80)
    
    t_test = 0.1
    
    print(f"\nt = {t_test}")
    print(f"\n{'N':>5}", end="")
    for alpha in alphas:
        print(f"  {'α='+str(alpha):>12}", end="")
    print()
    print("-" * (5 + 14*len(alphas)))
    
    for N in [1, 2, 3, 5, 10, 20, 30, 50, 70, 100]:
        row = f"{N:5d}"
        for alpha in alphas:
            eps_A = alpha * 2 * nu * k
            u_exact = exact_cole_hopf(t_test, x, eps_A, k, nu)
            u_pert = perturbative_clean(t_test, x, alpha, k, nu, N)
            
            err = np.sqrt(np.mean((u_pert - u_exact)**2))
            if err > 0:
                row += f"  {np.log10(err):12.4f}"
            else:
                row += f"  {'<1e-15':>12}"
        print(row)
    
    print("\n(Values are log₁₀ of RMS error)")
    
    # =====================================================================
    # Growth of individual terms
    # =====================================================================
    
    print("\n" + "="*80)
    print("GROWTH OF INDIVIDUAL PERTURBATIVE TERMS")
    print("="*80)
    
    print(f"\nMax |u_n(t,x)| over x, at t = {t_test}")
    print(f"\n{'n':>5}", end="")
    for alpha in alphas:
        print(f"  {'α='+str(alpha):>12}", end="")
    print()
    print("-" * (5 + 14*len(alphas)))
    
    for n in list(range(1, 21)) + list(range(25, 101, 5)):
        row = f"{n:5d}"
        for alpha in alphas:
            u_n = perturbative_clean(t_test, x, alpha, k, nu, n) \
                - perturbative_clean(t_test, x, alpha, k, nu, n-1)
            max_un = np.max(np.abs(u_n))
            if max_un > 1e-300:
                row += f"  {np.log10(max_un):12.4f}"
            else:
                row += f"  {'<1e-300':>12}"
        print(row)
    
    print("\n(Values are log₁₀ of max|u_n|)")
    
    # =====================================================================
    # Convergence at large α — how many terms needed?
    # =====================================================================
    
    print("\n" + "="*80)
    print("NUMBER OF TERMS NEEDED FOR 1% ACCURACY")
    print("="*80)
    
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        eps_A = alpha * 2 * nu * k
        u_exact = exact_cole_hopf(t_test, x, eps_A, k, nu)
        u_rms = np.sqrt(np.mean(u_exact**2))
        
        if u_rms < 1e-10:
            print(f"  α = {alpha:5.1f}: signal too small at t={t_test}")
            continue
        
        n_needed = None
        for N in range(1, 201):
            u_pert = perturbative_clean(t_test, x, alpha, k, nu, N)
            err = np.sqrt(np.mean((u_pert - u_exact)**2)) / u_rms
            if err < 0.01:
                n_needed = N
                break
        
        if n_needed:
            print(f"  α = {alpha:5.1f}: N = {n_needed:4d} terms needed (relative error < 1%)")
        else:
            print(f"  α = {alpha:5.1f}: > 200 terms needed")
    
    # =====================================================================
    # Save a cleaner comparison plot
    # =====================================================================
    
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
    fig2.suptitle('1D Burgers: convergence of perturbative sum to exact solution', fontsize=13)
    
    test_configs = [
        (1.0, 0.1, "α=1, t=0.1 (easy)"),
        (5.0, 0.1, "α=5, t=0.1 (moderate)"),
        (10.0, 0.1, "α=10, t=0.1 (hard)"),
        (1.0, 0.01, "α=1, t=0.01 (early)"),
        (5.0, 0.01, "α=5, t=0.01 (early, hard)"),
        (10.0, 1.0, "α=10, t=1 (late)"),
    ]
    
    for idx, (alpha, t, title) in enumerate(test_configs):
        ax = axes2[idx//3, idx%3]
        eps_A = alpha * 2 * nu * k
        u_exact = exact_cole_hopf(t, x, eps_A, k, nu)
        ax.plot(x, u_exact, 'k-', linewidth=2.5, label='Exact', zorder=10)
        
        for N, col, ls in [(5, 'C0', '-'), (20, 'C1', '-'), (50, 'C2', '-'), (100, 'C3', '-')]:
            u_pert = perturbative_clean(t, x, alpha, k, nu, N)
            ax.plot(x, u_pert, color=col, linewidth=1.2, linestyle=ls, label=f'N={N}')
        
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlabel('x')
        ax.set_ylabel('u(t,x)')
    
    plt.tight_layout()
    plt.savefig('/home/claude/burgers_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot")
