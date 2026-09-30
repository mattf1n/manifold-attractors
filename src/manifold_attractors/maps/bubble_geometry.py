"""Recovered cc2a8ac v3 bubble frame (NOT the later incompatible producer)."""
import numpy as np

def fib_lattice(n):
    """Unit Fibonacci (golden-spiral) lattice on S^2 — same construction as sphere_viz.fib_lattice."""
    k = np.arange(n)
    z = 1 - 2 * (k + 0.5) / n
    r = np.sqrt(np.clip(1 - z * z, 0, None))
    ph = np.pi * (3 - np.sqrt(5)) * k
    return np.stack([r * np.cos(ph), r * np.sin(ph), z], 1).astype(np.float32)

def avg_image_frame(FP, FD, ep):
    """Average-of-images frame: axis order F(p), J.u1, J.u2 (a1 exact, then Gram-Schmidt).
    FP (nf,D) = images of the frame points; FD (nf,4,D) = +-u1, +-u2 stencils at each frame point."""
    un = lambda v: v / (np.linalg.norm(v) + 1e-12)
    o = lambda v, b: un(v - (v @ b) * b)
    a1 = un(np.mean([un(FP[k]) for k in range(len(FP))], 0))                              # mean unit F(P_k)
    a2raw = np.mean([un((FD[k, 0] - FD[k, 1]) / (2 * ep)) for k in range(len(FP))], 0)    # mean unit J.u1
    a2 = o(a2raw, a1)
    a3raw = np.mean([un((FD[k, 2] - FD[k, 3]) / (2 * ep)) for k in range(len(FP))], 0)    # mean unit J.u2
    a3 = o(o(a3raw, a1), a2)
    return a1, a2, a3
