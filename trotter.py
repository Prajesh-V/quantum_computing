#!/usr/bin/env python3
"""
Trotterisation of a spin chain: error vs step count
===================================================

Model (transverse-field Ising chain, open boundary, n qubits):

    H = -J * sum_{i=0}^{n-2} Z_i Z_{i+1} - h * sum_{i=0}^{n-1} X_i     with J = 1, h = 1

The Hamiltonian is split into two pieces:

    A = -J * sum_i Z_i Z_{i+1}      (all diagonal ZZ terms)
    B = -h * sum_i X_i              (all off-diagonal X terms)
    H = A + B

Two product formulas are compared against the exact evolution U = exp(-i H t):

    1st order (Lie-Trotter):  [ exp(-i A t/r) exp(-i B t/r) ]^r
    2nd order (Strang):       [ exp(-i A t/2r) exp(-i B t/r) exp(-i A t/2r) ]^r

Since A and B do not commute, the product formulas carry an error that
decreases as a power of r. For a single step of length dt the local error is
O(dt^2) (1st order) and O(dt^3) (2nd order); over r = t/dt steps the global
error therefore scales as O(t^2/r) and O(t^3/r^2) respectively.

Two figures of merit are measured:

    operator norm error : ||U_exact - U_trot||_2
    state infidelity    : 1 - |<psi_exact|psi_trot>|^2,  |psi> = U|00...0>

Only numpy, scipy and matplotlib are used. Run `python trotter.py` and all
outputs land in results/ (plots) plus results.csv (the raw numbers).
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from scipy.linalg import expm

import matplotlib

# Non-interactive backend: the script must run headless (no display needed).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)


# ----------------------------------------------------------------------------
# 1. Small linear-algebra helpers (Kronecker products built by hand)
# ----------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def kron_all(ops):
    """Kronecker product of a list of operators, left to right.

    kron_all([P0, P1, P2]) = P0 (x) P1 (x) P2

    This is the only place where tensor products are formed; no library
    helper is used. The first factor is the most significant qubit, so the
    computational basis index b reads as b = b_0 b_1 ... b_{n-1} in binary.
    """
    out = np.array(ops[0], dtype=complex)
    for op in ops[1:]:
        out = np.kron(out, np.asarray(op, dtype=complex))
    return out


def single_qubit(n: int, i: int, op):
    """Embed a 2x2 operator `op` acting on qubit `i` into the n-qubit space."""
    factors = [I2] * n
    factors[i] = op
    return kron_all(factors)


def two_qubit(n: int, i: int, j: int, op_i, op_j):
    """Embed the two-site operator op_i (x) op_j acting on qubits i and j."""
    factors = [I2] * n
    factors[i] = op_i
    factors[j] = op_j
    return kron_all(factors)


# ----------------------------------------------------------------------------
# 2. Hamiltonians
# ----------------------------------------------------------------------------

def tfim_terms(n: int, J: float = 1.0, h: float = 1.0):
    """Transverse-field Ising chain, open boundaries.

    Returns (A, B, H) with A = -J sum ZZ and B = -h sum X.
    All terms inside A commute with each other, and likewise inside B.
    """
    dim = 2 ** n
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    for i in range(n - 1):                      # open boundary: n-1 bonds
        A += -J * two_qubit(n, i, i + 1, Z, Z)
    for i in range(n):
        B += -h * single_qubit(n, i, X)

    return A, B, A + B


def heisenberg_terms(n: int):
    """Isotropic Heisenberg chain, open boundaries (optional variant).

        H = sum_{i} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

    We choose the split ourselves:
        A = sum_i (X_i X_{i+1} + Y_i Y_{i+1})   (flip-flop / hopping part)
        B = sum_i  Z_i Z_{i+1}                  (diagonal Ising part)
    Both halves are sums of mutually commuting bond terms, so each block can
    still be exponentiated directly with expm.
    """
    dim = 2 ** n
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)

    for i in range(n - 1):
        A += two_qubit(n, i, i + 1, X, X) + two_qubit(n, i, i + 1, Y, Y)
        B += two_qubit(n, i, i + 1, Z, Z)

    return A, B, A + B


# ----------------------------------------------------------------------------
# 3. Evolution: exact and Trotterised
# ----------------------------------------------------------------------------

class Spectral:
    """Cached eigendecomposition of a Hermitian operator M.

        exp(-i M t) = V diag(exp(-i lambda t)) V^dag

    This is exact up to round-off, and much cheaper than a Pade expm once the
    dimension is larger than a few qubits (scipy's expm takes about 10 s at
    dim 256). Only the n-scaling study uses it: the main results of the script
    keep scipy.linalg.expm exactly as specified.
    """

    def __init__(self, M):
        self.w, self.V = np.linalg.eigh(M)

    def exp(self, t: float):
        return (self.V * np.exp(-1j * self.w * t)) @ self.V.conj().T


def trotter_unitary(A, B, t: float, r: int, order: int):
    """Product-formula propagator for total time t split into r steps.

    order = 1 : Lie-Trotter,     step = exp(-i A t/r) exp(-i B t/r)
    order = 2 : Strang,          step = exp(-i A t/2r) exp(-i B t/r) exp(-i A t/2r)

    The r-th power of one step is taken with np.linalg.matrix_power, as asked.
    """
    if r < 1:
        raise ValueError("r must be a positive integer")

    dt = t / r
    if order == 1:
        step = expm(-1j * A * dt) @ expm(-1j * B * dt)
    elif order == 2:
        half = expm(-1j * A * (dt / 2.0))
        step = half @ expm(-1j * B * dt) @ half
    else:
        raise ValueError("order must be 1 or 2")

    return np.linalg.matrix_power(step, r)


# ----------------------------------------------------------------------------
# 4. Error measures
# ----------------------------------------------------------------------------

def operator_error(U_exact, U_trot) -> float:
    """Spectral (operator 2-) norm of the difference."""
    return float(np.linalg.norm(U_exact - U_trot, 2))


def infidelity(psi_exact, psi_trot) -> float:
    """1 - |<psi_exact|psi_trot>|^2 for normalised state vectors.

    The result is clipped at 0: round-off can make the overlap marginally
    larger than 1, and a negative infidelity would turn into a NaN in log plots.
    """
    overlap = np.vdot(psi_exact, psi_trot)          # np.vdot conjugates arg 1
    return float(max(0.0, 1.0 - abs(overlap) ** 2))


def basis_state(n: int, bits) -> np.ndarray:
    """Computational basis state |b_0 b_1 ... b_{n-1}> (qubit 0 = MSB)."""
    psi = np.zeros(2 ** n, dtype=complex)
    index = 0
    for b in bits:
        index = (index << 1) | int(b)
    psi[index] = 1.0
    return psi


def neel_state(n: int) -> np.ndarray:
    """Neel state |0101...>: an antiferromagnetic product state."""
    return basis_state(n, [i % 2 for i in range(n)])


def sweep(A, B, H, t: float, rs, orders=(1, 2), psi0=None):
    """Run every r for every order and return the two error measures.

    psi0 is the initial state for the infidelity (default |00...0>).
    Returns dict: results[order] = {"op": ndarray, "inf": ndarray}
    """
    dim = H.shape[0]
    U_exact = expm(-1j * H * t)
    if psi0 is None:
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0                            # |00...0>
    psi_exact = U_exact @ psi0

    results = {}
    for order in orders:
        op_err, inf_err = [], []
        for r in rs:
            U_trot = trotter_unitary(A, B, t, int(r), order)
            op_err.append(operator_error(U_exact, U_trot))
            inf_err.append(infidelity(psi_exact, U_trot @ psi0))
        results[order] = {"op": np.array(op_err), "inf": np.array(inf_err)}
    return results


# ----------------------------------------------------------------------------
# 5. Fitting the scaling exponent
# ----------------------------------------------------------------------------

def fit_slope(rs, errs, floor: float = 1e-13):
    """Least-squares slope of log(error) vs log(r) on the "clean" part.

    Points are discarded when
      * the error has hit the floating-point floor (<= floor): there the curve
        is dominated by round-off and its slope is meaningless, and
      * r is the smallest value (r = 1): the pre-asymptotic regime where
        higher-order terms still contribute.
    Returns (slope, mask).
    """
    rs = np.asarray(rs, dtype=float)
    errs = np.asarray(errs, dtype=float)

    mask = (errs > floor) & (rs > rs.min())
    if mask.sum() < 3:
        return float("nan"), mask

    slope = np.polyfit(np.log(rs[mask]), np.log(errs[mask]), 1)[0]
    return float(slope), mask


def trotter_number(rs, errs, eps: float, n_extrap: int = 4):
    """Smallest r whose operator error is at most eps.

    Inside the measured range this is a log-log interpolation of the curve, so
    it makes no assumption about the exponent. If eps is smaller than every
    measured error (which is the normal case for the first-order formula, where
    reaching 1e-3 needs r of order 1e3) the curve is continued with the power
    law fitted to the last `n_extrap` points, and the caller is told so.

    Returns (r, extrapolated).
    """
    rs = np.asarray(rs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    keep = errs > 0
    rs, errs = rs[keep], errs[keep]
    x, y = np.log(rs), np.log(errs)          # y decreases with x

    if np.log(eps) >= y.min():
        # inside the data: x as a function of y (y is decreasing, so reverse)
        return float(np.exp(np.interp(np.log(eps), y[::-1], x[::-1]))), False

    k = min(n_extrap, len(x))
    slope, intercept = np.polyfit(x[-k:], y[-k:], 1)   # log err = slope*log r + c
    if slope >= 0:
        return float("nan"), True
    return float(np.exp((np.log(eps) - intercept) / slope)), True


# ----------------------------------------------------------------------------
# 6. Plotting helpers
# ----------------------------------------------------------------------------

STYLE = {
    1: dict(label="1st order (Lie-Trotter)", color="#d1495b", marker="o", ls="-"),
    2: dict(label="2nd order (Strang)", color="#00798c", marker="s", ls="-"),
}


def _guide(ax, r_ref, slope, text_xy, text):
    """Dashed reference line with a given power law, for the eye."""
    y0 = 0.5
    ax.plot(r_ref, y0 * (r_ref / r_ref[0]) ** slope, ls=":", color="0.45",
            lw=1.2, label=f"slope {slope:+d} guide")
    ax.annotate(text, xy=text_xy, fontsize=9, color="0.35")


def plot_errors(rs, results, quantity, ylabel, title, path, fit_note=True):
    """Generic log-log plot of one error measure for both orders."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    r_ref = np.array([rs[0], rs[-1]], dtype=float)

    for order in (1, 2):
        errs = results[order][quantity]
        st = STYLE[order]
        label = st["label"]
        if fit_note:
            slope, used = fit_slope(rs, errs)
            label += f"  (fit slope {slope:+.2f})"
            ax.plot(np.asarray(rs)[used], errs[used], ls="none",
                    marker="o" if order == 1 else "s",
                    mfc="none", mec=st["color"], ms=9, mew=1.0,
                    label="  fit window")
        ax.loglog(rs, errs, marker=st["marker"], color=st["color"],
                  ls=st["ls"], ms=4.5, lw=1.5, label=label)

    _guide(ax, r_ref, -1, (r_ref[-1] * 0.55, 3e-2), "~ r^-1")
    _guide(ax, r_ref, -2, (r_ref[-1] * 0.55, 3e-4), "~ r^-2")

    ax.set_xlabel("number of Trotter steps  r")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def floor_study(n, t, rmax, npoints, path, csv_path, floor: float = 1e-13):
    """Push r far past the range used for the slope fits, until round-off wins.

    A product formula applied step by step accumulates round-off roughly like
    eps_machine * r, while its truncation error falls like 1/r (or 1/r^2).  The
    two cross: past that point the measured error grows again, and a slope
    fitted over the whole range is meaningless.  This function locates the
    minimum and compares the slope on the descending branch with the slope
    obtained over every point, to show which one the 1e-13 rule keeps.
    """
    A, B, H = tfim_terms(n)
    rs = np.asarray(np.unique(
        np.round(np.geomspace(1, rmax, npoints)).astype(int)), dtype=float)
    res = sweep(A, B, H, t, rs)
    eps_mach = np.finfo(float).eps

    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    rows, stats = [], {}

    for order in (1, 2):
        errs = res[order]["op"]
        st = STYLE[order]
        i_min = int(np.argmin(errs))
        descend = (rs <= rs[i_min]) & (rs > rs[0]) & (errs > 0)
        slope_desc = float(np.polyfit(np.log(rs[descend]),
                                      np.log(errs[descend]), 1)[0])
        every = errs > 0
        slope_all = float(np.polyfit(np.log(rs[every]), np.log(errs[every]), 1)[0])
        slope_cut, cut_mask = fit_slope(rs, errs, floor=floor)

        stats[order] = dict(e_min=float(errs[i_min]), r_min=float(rs[i_min]),
                            slope_desc=slope_desc, slope_all=slope_all,
                            slope_cut=slope_cut,
                            n_dropped=int((~cut_mask).sum()))

        ax.loglog(rs, errs, marker=st["marker"], ms=3.6, lw=1.4,
                  color=st["color"],
                  label=f"{st['label']}:  min {errs[i_min]:.1e} at r={rs[i_min]:.0f}"
                        + ("" if i_min == len(rs) - 1 else f",  slope {slope_desc:+.2f}"))
        ax.loglog(rs[descend], errs[descend], ls="none",
                  marker="o" if order == 1 else "s", mfc="none",
                  mec=st["color"], ms=8, mew=1.0)

        for r, e_op, e_inf in zip(rs, res[order]["op"], res[order]["inf"]):
            rows.append([order, int(r), f"{e_op:.12e}", f"{e_inf:.12e}"])

    ax.axhline(floor, color="0.35", ls="--", lw=1.2)
    ax.annotate(f"{floor:.0e}: floor assumed by the slope fits (never reached)",
                xy=(rs[0] * 1.4, floor * 1.5), fontsize=8, color="0.35")
    ax.loglog(rs, eps_mach * rs, color="0.6", ls=":", lw=1.5,
              label=r"round-off guide  $\epsilon_{\rm mach}\,r$")
    ax.set_xlabel("number of Trotter steps  r")
    ax.set_ylabel(r"$\Vert U_\mathrm{exact} - U_\mathrm{trot}\Vert_2$")
    ax.set_title(f"How far can r be pushed?  Ising chain n={n}, t={t}\n"
                 "hollow markers: branch used for the slope")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["order", "r", "operator_norm_error", "state_infidelity"])
        writer.writerows(rows)
    return stats


def scaling_study(ns, t, rmax, npoints, eps, path, csv_path):
    """Trotter number r(eps) versus chain length, and the fitted exponent p(n).

    The paper plots the same quantity (their Fig. 3): rather than error against
    r, ask how many steps a target accuracy costs and how that grows with n.
    """
    rs = np.asarray(np.unique(
        np.round(np.geomspace(1, rmax, npoints)).astype(int)), dtype=float)
    r_eps = {1: [], 2: []}
    expo = {1: [], 2: []}
    extrap = {1: [], 2: []}
    rows = []

    for n in ns:
        A, B, H = tfim_terms(n)
        sA, sB, sH = Spectral(A), Spectral(B), Spectral(H)
        U_exact = sH.exp(t)

        for order in (1, 2):
            errs = []
            for r in rs:
                dt = t / int(r)
                if order == 1:
                    step = sA.exp(dt) @ sB.exp(dt)
                else:
                    half = sA.exp(dt / 2.0)
                    step = half @ sB.exp(dt) @ half
                errs.append(operator_error(
                    U_exact, np.linalg.matrix_power(step, int(r))))
            errs = np.asarray(errs)

            p, _ = fit_slope(rs, errs)
            r_val, was_extrap = trotter_number(rs, errs, eps)
            r_eps[order].append(r_val)
            expo[order].append(abs(p))
            extrap[order].append(was_extrap)
            rows.append([n, order, f"{p:.4f}", f"{r_val:.6e}", int(was_extrap)])
            print(f"    n={n}  order {order}:  exponent {p:+.2f}   "
                  f"r({eps:g}) = {r_val:.4g}  "
                  f"({'extrapolated' if was_extrap else 'interpolated'})")

    ns = np.asarray(ns, dtype=float)
    n_exp = {}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    for order in (1, 2):
        st = STYLE[order]
        y = np.asarray(r_eps[order], dtype=float)
        flag = np.asarray(extrap[order], dtype=bool)
        g, _ = np.polyfit(np.log(ns), np.log(y), 1)
        n_exp[order] = float(g)
        ax.loglog(ns, y, marker=st["marker"], ms=5, lw=1.5, color=st["color"],
                  label=f"{st['label']}:  r ~ n^{g:.2f}")
        if flag.any():
            ax.loglog(ns[flag], y[flag], ls="none", marker=st["marker"], ms=10,
                      mfc="none", mec=st["color"], mew=1.3)
        ax2.semilogx(ns, expo[order], marker=st["marker"], ms=5, lw=1.5,
                     color=st["color"], label=st["label"])

    ax.set_xlabel("chain length  n")
    ax.set_ylabel(f"Trotter number  r  for error = {eps:g}")
    ax.set_title(f"Steps needed to reach error {eps:g}\n"
                 f"(hollow = extrapolated past r={rs[-1]:.0f})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    ax2.set_xlabel("chain length  n")
    ax2.set_ylabel(r"fitted exponent  $|d\ln \mathrm{error} / d\ln r|$")
    ax2.set_title("Error exponent is independent of chain length")
    ax2.axhline(1.0, ls=":", lw=1.1, color="0.5")
    ax2.axhline(2.0, ls=":", lw=1.1, color="0.5")
    ax2.set_ylim(0.0, 2.6)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n", "order", "error_exponent",
                         "trotter_number", "extrapolated"])
        writer.writerows(rows)

    return dict(rows=rows, n_exponent=n_exp, eps=eps)


# ----------------------------------------------------------------------------
# 7. Main experiment
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=float, default=1.0, help="evolution time")
    parser.add_argument("--rmax", type=int, default=200, help="largest r")
    parser.add_argument("--npoints", type=int, default=26,
                        help="number of log-spaced r values")
    parser.add_argument("--outdir", default="results", help="plot directory")
    parser.add_argument("--csv", default="results.csv",
                        help="path of the results table")
    parser.add_argument("--no-heisenberg", action="store_true",
                        help="skip the optional Heisenberg variant")
    parser.add_argument("--floor-rmax", type=int, default=10 ** 8,
                        help="largest r in the round-off study")
    parser.add_argument("--no-floor-study", action="store_true",
                        help="skip the round-off study (plot F)")
    parser.add_argument("--eps", type=float, default=1e-3,
                        help="target accuracy of the Trotter-number study")
    parser.add_argument("--scaling-nmax", type=int, default=7,
                        help="largest chain length in the Trotter-number study "
                             "(n=8 adds about 45 s: dim 256)")
    parser.add_argument("--scaling-rmax", type=int, default=200,
                        help="largest r in the Trotter-number study")
    parser.add_argument("--scaling-npoints", type=int, default=14,
                        help="number of log-spaced r values in that study")
    parser.add_argument("--no-scaling", action="store_true",
                        help="skip the Trotter-number study (plot G)")
    args = parser.parse_args()

    t = args.t
    os.makedirs(args.outdir, exist_ok=True)

    # Log-spaced integer step counts, unique and sorted, e.g.
    # 1, 2, 3, 4, 5, 6, 8, 10, ... 200
    rs = np.unique(np.round(
        np.geomspace(1, args.rmax, args.npoints)).astype(int))

    rows = []          # rows for results.csv
    slopes = {}        # slopes[(model, n, order, quantity)] -> slope

    # -- sanity check: the exact propagator must be unitary -------------------
    print("=" * 78)
    print("Sanity checks")
    print("=" * 78)
    A4, B4, H4 = tfim_terms(4)
    U4 = expm(-1j * H4 * t)
    unit_err = np.linalg.norm(U4.conj().T @ U4 - np.eye(2 ** 4), 2)
    print(f"  ||U^dag U - 1||_2 for n=4, t={t}: {unit_err:.3e}")
    if unit_err > 1e-12:
        raise SystemExit("exact propagator is not unitary - aborting")
    herm_err = np.linalg.norm(H4 - H4.conj().T, 2)
    print(f"  ||H - H^dag||_2                  : {herm_err:.3e}")
    if herm_err > 1e-12:
        raise SystemExit("Hamiltonian is not Hermitian - aborting")

    # -- transverse-field Ising chain, n = 4, 5, 6 ----------------------------
    print()
    print("=" * 78)
    print("Transverse-field Ising chain, H = -sum ZZ - sum X, open boundary")
    print("=" * 78)

    datasets = {}
    for n in (4, 5, 6):
        A, B, H = tfim_terms(n)
        res = sweep(A, B, H, t, rs)
        datasets[n] = res

        for order in (1, 2):
            s_op, _ = fit_slope(rs, res[order]["op"])
            s_inf, _ = fit_slope(rs, res[order]["inf"])
            slopes[("TFIM", n, order, "op")] = s_op
            slopes[("TFIM", n, order, "inf")] = s_inf
            print(f"  n={n}  order {order}:  op-norm slope {s_op:+.2f}   "
                  f"infidelity slope {s_inf:+.2f}   "
                  f"(err at r={rs[0]}: {res[order]['op'][0]:.2e}, "
                  f"at r={rs[-1]}: {res[order]['op'][-1]:.2e})")

        for order in (1, 2):
            for r, e_op, e_inf in zip(rs, res[order]["op"], res[order]["inf"]):
                rows.append(["TFIM", n, order, int(r), "0000",
                             f"{e_op:.12e}", f"{e_inf:.12e}"])

    # -- plots A and B: n = 4, both orders ------------------------------------
    n = 4
    plot_errors(rs, datasets[n], "op",
                r"$\Vert U_\mathrm{exact} - U_\mathrm{trot}\Vert_2$",
                f"Operator-norm error, Ising chain n={n}, t={t}",
                os.path.join(args.outdir, "plot_A_operator_error.png"))
    plot_errors(rs, datasets[n], "inf",
                r"$1 - |\langle\psi_\mathrm{exact}|\psi_\mathrm{trot}\rangle|^2$",
                f"State infidelity from |00...0>, Ising chain n={n}, t={t}",
                os.path.join(args.outdir, "plot_B_state_infidelity.png"))

    # -- plot C: chain length dependence -------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    colors = {4: "#1b4965", 5: "#5fa8d3", 6: "#c1666b"}
    for nn in (4, 5, 6):
        for order in (1, 2):
            ax.loglog(rs, datasets[nn][order]["op"], marker="o" if order == 1 else "s",
                      ms=3.8, lw=1.4, color=colors[nn],
                      ls="-" if order == 1 else "--",
                      label=f"n={nn}, {STYLE[order]['label'].split(' (')[0]}")
    ax.set_xlabel("number of Trotter steps  r")
    ax.set_ylabel(r"$\Vert U_\mathrm{exact} - U_\mathrm{trot}\Vert_2$")
    ax.set_title(f"Operator-norm error vs chain length, t={t}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "plot_C_chain_length.png"), dpi=160)
    plt.close(fig)

    # -- plot D: commutator bound for first order ----------------------------
    # For Lie-Trotter:  ||U_trot - U_exact||  <=  (t^2 / 2r) * ||[A, B]||
    norm_comm = float(np.linalg.norm(A4 @ B4 - B4 @ A4, 2))
    bound = (t ** 2 / (2.0 * rs)) * norm_comm
    measured = datasets[4][1]["op"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5))
    ax.loglog(rs, measured, marker="o", ms=4, lw=1.5, color=STYLE[1]["color"],
              label="measured 1st-order error")
    ax.loglog(rs, bound, ls="--", lw=1.7, color="#2a9d8f",
              label=r"bound  $(t^2/2r)\,\Vert[A,B]\Vert$")
    ax.set_xlabel("number of Trotter steps  r")
    ax.set_ylabel(r"$\Vert U_\mathrm{exact} - U_\mathrm{trot}\Vert_2$")
    ax.set_title(f"Commutator bound, Ising chain n=4, t={t}\n"
                 rf"$\Vert[A,B]\Vert_2$ = {norm_comm:.3f}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower left")

    ratio = bound / measured
    ax2.semilogx(rs, ratio, marker="o", ms=4, lw=1.5, color="#6d597a")
    ax2.axhline(1.0, ls="--", lw=1.2, color="0.4", label="bound = error")
    ax2.set_xlabel("number of Trotter steps  r")
    ax2.set_ylabel("bound / measured error")
    ax2.set_title("Bound tightness (values > 1 mean the bound holds)")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "plot_D_commutator_bound.png"), dpi=160)
    plt.close(fig)

    print()
    print(f"  ||[A,B]||_2 (n=4) = {norm_comm:.4f}")
    print(f"  bound/measured ratio ranges from {ratio.min():.2f} to {ratio.max():.2f}")
    print(f"  bound respected for every r: {bool(np.all(ratio >= 1.0))}")

    # -- optional Heisenberg variant -----------------------------------------
    if not args.no_heisenberg:
        print()
        print("=" * 78)
        print("Optional variant: Heisenberg chain, H = sum (XX + YY + ZZ)")
        print("=" * 78)
        nn = 4
        Ah, Bh, Hh = heisenberg_terms(nn)
        # The all-up state is annihilated by A = sum (XX + YY), so exp(-iAt)
        # acts on it as the identity and the infidelity from |00...0> is
        # exactly zero for every r.  That makes the |00...0> curve useless
        # here, so the Heisenberg infidelity is measured from the Neel state
        # |0101...> instead, where A does act non-trivially.
        res_heis_0 = sweep(Ah, Bh, Hh, t, rs)                       # |00...0>
        res_heis = sweep(Ah, Bh, Hh, t, rs, psi0=neel_state(nn))    # |0101...>
        datasets["heis"] = res_heis

        print(f"  infidelity from |{'0' * nn}> for every r: "
              f"max = {res_heis_0[1]['inf'].max():.1e}  "
              "(A annihilates |0...0>, so exp(-iAt)|0...0> = |0...0>)")

        for order in (1, 2):
            s_op, _ = fit_slope(rs, res_heis[order]["op"])
            s_inf, _ = fit_slope(rs, res_heis[order]["inf"])
            slopes[("Heisenberg", nn, order, "op")] = s_op
            slopes[("Heisenberg", nn, order, "inf")] = s_inf
            print(f"  n={nn}  order {order}:  op-norm slope {s_op:+.2f}   "
                  f"infidelity slope {s_inf:+.2f}  (Neel start)")
            for r, e_op, e_inf0 in zip(rs, res_heis_0[order]["op"],
                                       res_heis_0[order]["inf"]):
                rows.append(["Heisenberg", nn, order, int(r), "0000",
                             f"{e_op:.12e}", f"{e_inf0:.12e}"])
            for r, e_op, e_inf in zip(rs, res_heis[order]["op"],
                                      res_heis[order]["inf"]):
                rows.append(["Heisenberg", nn, order, int(r), "0101",
                             f"{e_op:.12e}", f"{e_inf:.12e}"])

        # Two panels: operator error, and infidelity from the Neel state.
        fig, (axe, axf) = plt.subplots(1, 2, figsize=(11.0, 4.5))
        for order in (1, 2):
            st = STYLE[order]
            s_op, _ = fit_slope(rs, res_heis[order]["op"])
            s_inf, _ = fit_slope(rs, res_heis[order]["inf"])
            axe.loglog(rs, res_heis[order]["op"], marker=st["marker"], ms=4,
                       lw=1.5, color=st["color"],
                       label=f"{st['label']}  (slope {s_op:+.2f})")
            axf.loglog(rs, res_heis[order]["inf"], marker=st["marker"], ms=4,
                       lw=1.5, color=st["color"],
                       label=f"{st['label']}  (slope {s_inf:+.2f})")
        axe.set_ylabel(r"$\Vert U_\mathrm{exact} - U_\mathrm{trot}\Vert_2$")
        axe.set_title("Operator-norm error")
        axf.set_ylabel(r"$1 - |\langle\psi_\mathrm{exact}|\psi_\mathrm{trot}\rangle|^2$")
        axf.set_title(r"Infidelity, Neel start $|0101\ldots\rangle$")
        for ax in (axe, axf):
            ax.set_xlabel("number of Trotter steps  r")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"Optional variant: Heisenberg chain n={nn}, t={t}",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "plot_E_heisenberg.png"), dpi=160)
        plt.close(fig)

    # -- plot F: how far can r be pushed before round-off wins? --------------
    if not args.no_floor_study:
        print()
        print("=" * 78)
        print("Round-off floor: operator error when r is pushed far past 200")
        print("=" * 78)
        stats = floor_study(
            4, t, args.floor_rmax, 55,
            os.path.join(args.outdir, "plot_F_roundoff_floor.png"),
            os.path.join(args.outdir, "floor_study.csv"))
        for order in (1, 2):
            s = stats[order]
            print(f"  order {order}: minimum {s['e_min']:.2e} at r={s['r_min']:.0f} "
                  + ("- round-off has taken over"
                     if s["r_min"] < args.floor_rmax else
                     "- still descending at the largest r"))
            print(f"    slope on the descending branch   : {s['slope_desc']:+.2f}")
            print(f"    slope over every point           : {s['slope_all']:+.2f}")
            print(f"    slope with the 1e-13 cut applied : {s['slope_cut']:+.2f}"
                  f"   (cut dropped {s['n_dropped']} points)")

    # -- plot G: Trotter number for a target accuracy, versus chain length ---
    if not args.no_scaling:
        print()
        print("=" * 78)
        print(f"Trotter number r(eps) vs chain length, eps = {args.eps:g}")
        print("=" * 78)
        sc = scaling_study(
            list(range(2, args.scaling_nmax + 1)), t,
            args.scaling_rmax, args.scaling_npoints, args.eps,
            os.path.join(args.outdir, "plot_G_trotter_number.png"),
            os.path.join(args.outdir, "trotter_number.csv"))
        print(f"  {'n':>3} {'order':>6} {'exponent':>9} {'r(eps)':>12}  source")
        for n_, order, p, r_val, was_extrap in sc["rows"]:
            src = "extrapolated" if was_extrap else "interpolated"
            print(f"  {n_:>3} {order:>6} {float(p):>+9.2f} {float(r_val):>12.4g}"
                  f"  {src}")
        for order in (1, 2):
            print(f"  order {order}: r(eps) ~ n^{sc['n_exponent'][order]:+.2f}")

    # -- results table --------------------------------------------------------
    csv_path = args.csv
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["model", "n", "order", "r", "initial_state",
                         "operator_norm_error", "state_infidelity"])
        writer.writerows(rows)
    print()
    print(f"  wrote {csv_path} ({len(rows)} rows)")

    # -- final verification: error must go to zero as r grows -----------------
    print()
    print("=" * 78)
    print("Verification: convergence with r")
    print("=" * 78)
    for n in (4, 5, 6):
        for order in (1, 2):
            e = datasets[n][order]["op"]
            decay = e[0] / e[-1]
            monotone = np.all(np.diff(e) < 0)
            print(f"  n={n} order {order}: error {e[0]:.3e} -> {e[-1]:.3e} "
                  f"(factor {decay:.1e}), strictly decreasing: {monotone}")
            if not (e[-1] < e[0] and decay > 10):
                raise SystemExit("Trotter error did not decrease - check code")

    print()
    print("Fitted slopes (log-log least squares, floating-point floor dropped):")
    print(f"  {'model':>11} {'n':>3} {'order':>6} {'op-norm':>9} {'infidel.':>9}")
    for model in ("TFIM", "Heisenberg"):
        ns = (4, 5, 6) if model == "TFIM" else (4,)
        if (model, ns[0], 1, "op") not in slopes:
            continue                      # variant not run (--no-heisenberg)
        for nn in ns:
            for order in (1, 2):
                s_op = slopes.get((model, nn, order, "op"), float("nan"))
                s_inf = slopes.get((model, nn, order, "inf"), float("nan"))
                print(f"  {model:>11} {nn:>3} {order:>6} "
                      f"{s_op:>+9.2f} {s_inf:>+9.2f}")
    print()


if __name__ == "__main__":
    main()
