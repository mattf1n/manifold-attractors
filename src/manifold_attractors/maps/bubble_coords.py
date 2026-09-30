"""Fresh recovered-v3 bubbles, including the current figure's sixth natural pair."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from .bubble_geometry import fib_lattice, avg_image_frame
from .inputs import load, save

MID, INJECT, READ = 'meta-llama/Llama-3.1-8B', 15, 25
NFIB, M, N_SYNTH, SEED = 4000, 5, 100, 20240706
FIB = fib_lattice(NFIB)
U = lambda x: x / (x.norm() + 1e-9)
unit_np = lambda a: a / (np.linalg.norm(a) + 1e-9)
rows = []
_M = {}

class grid_geom:
    @staticmethod
    def tangent_axes(center,a,b):
        ch=U(center)
        e1=U(a-(a@ch)*ch)
        e2=U(b-(b@ch)*ch-(b@e1)*e1)
        return e1,e2

def model_bits():
    return _M

def forward_images(center_np, ea_np, eb_np, extra_np, frame_pts_np, frame_dirs_np):
    """Cached full-D quantities: lattice images FS, extra-point images FE, the images FP of the frame points,
    and the finite-difference stencils FD at each frame point (for J.ea and J.eb) plus the step ep.
    The FRAME is NOT baked in — stage 3 derives it, so a frame-convention change costs no GPU."""
    B = model_bits()
    dev = B["dev"]
    center = torch.tensor(center_np, device=dev)
    ea, eb = torch.tensor(ea_np, device=dev), torch.tensor(eb_np, device=dev)
    R = float(center.norm())
    X = R * (torch.tensor(FIB, device=dev) @ torch.stack([ea, eb, U(center)]))
    ep = 1e-3 * R + 1e-6
    P = torch.tensor(np.stack(frame_pts_np), device=dev)                     # (nf, D) frame points (on the sphere)
    Dirs = torch.tensor(np.stack(frame_dirs_np), device=dev)                 # (nf, 2, D) per-point tangents
    sten = torch.cat([torch.stack([P[k] + ep * Dirs[k, 0], P[k] - ep * Dirs[k, 0],
                                   P[k] + ep * Dirs[k, 1], P[k] - ep * Dirs[k, 1]])
                      for k in range(len(P))])
    FD = B["fb"](sten).reshape(len(P), 4, -1)                                # (nf,4,D) +-u1_k, +-u2_k per point
    E = torch.tensor(np.stack([unit_np(v) for v in extra_np]), device=dev)
    out = tuple(t.cpu().numpy() for t in (B["fb"](X), B["fb"](R * E), B["fb"](P), FD))
    return out + (ep,)

def bubble(kind, s_idx, center_np, ea_np, eb_np, token, token2, angle_deg, extra_np,
           frame_pts_np, frame_dirs_np):
    FS, FE, FP, FD, ep = forward_images(center_np, ea_np, eb_np, extra_np, frame_pts_np, frame_dirs_np)
    a1, a2, a3 = avg_image_frame(FP, FD, ep)
    A = np.stack([a1, a2, a3], 1)                                  # (D,3) output frame
    R = float(np.linalg.norm(center_np))
    proj = FS @ A                                                  # measured FROM THE ORIGIN
    chat = unit_np(center_np)
    for i in range(NFIB):
        rows.append((kind, s_idx, token, token2, angle_deg, R, i,
                     float(FIB[i, 0]), float(FIB[i, 1]), float(FIB[i, 2]),
                     float(proj[i, 0]), float(proj[i, 1]), float(proj[i, 2])))
    for j, dvec in enumerate(extra_np):                            # the embeddings themselves (rows with i<0)
        dh = unit_np(dvec)
        fibe = np.array([dh @ ea_np, dh @ eb_np, dh @ chat])       # its direction in the frame
        pe = FE[j] @ A                                             # its exact image
        rows.append((kind, s_idx, token, token2, angle_deg, R, -(j + 1),
                     float(fibe[0]), float(fibe[1]), float(fibe[2]),
                     float(pe[0]), float(pe[1]), float(pe[2])))
    rad = np.linalg.norm(proj, axis=1)
    print(f"  [{kind}] {s_idx+1}/{M}: {token!r}" + (f" -> {token2!r} ({angle_deg:.1f}deg)" if token2 else "")
          + f"  R={R:.1f}  ||F|| range=[{rad.min():.1f},{rad.max():.1f}]", flush=True)

def tangent_axes_np(center_np, u_np, v_np):
    """grid_geom.tangent_axes on numpy inputs (keeps stage 3 CPU-only)."""
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    e1, e2 = grid_geom.tangent_axes(t(center_np), t(u_np), t(v_np))
    return e1.numpy(), e2.numpy()

def pair_bubble(kind, s_idx, p, q, tok_p, tok_q, axis_seed):
    """Sphere at the mean magnitude, SYMMETRIC in p and q: the pole is their bisector and the first tangent
    axis is unit(p_hat - q_hat) (the between-embedding direction, tangent at the bisector). p and q therefore
    sit at -+angle_deg/2 from the pole, and swapping p/q only flips the sign of that axis."""
    R = float((np.linalg.norm(p) + np.linalg.norm(q)) / 2)         # mean of the two magnitudes
    phat, qhat = unit_np(p), unit_np(q)
    ctr = R * unit_np(phat + qhat)                                 # pole = the BISECTOR (order-invariant)
    w = unit_np(phat - qhat)                                       # between-embedding dir, tangent at the pole
    ang = float(np.degrees(np.arccos(np.clip(phat @ qhat, -1, 1))))
    torch.manual_seed(axis_seed)
    e1, e2 = tangent_axes_np(ctr, w, torch.randn(D).numpy())
    # per-point tangents of the p->q great circle: toward q at p, continuing onward at q (consistent travel
    # direction, so swapping p/q just reverses both). e2 is orthogonal to span(p,q) -> tangent at both.
    t_p = unit_np(qhat - (qhat @ phat) * phat)                     # at p: toward q
    t_q = unit_np(qhat * (phat @ qhat) - phat)                     # at q: away from p (same travel direction)
    e2 = unit_np(e2 - (e2 @ phat) * phat - (e2 @ qhat) * qhat)     # make e2 _|_ span(p,q)
    # frame points: BOTH embeddings, taken on the sphere (at the grid radius r_mean)
    bubble(kind, s_idx, ctr, e1, e2, tok_p, tok_q, ang, [phat, qhat], [R * phat, R * qhat],
           [[t_p, e2], [t_q, e2]])
