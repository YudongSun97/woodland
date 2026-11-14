"""Non-uniform slip on flat triangles: the first-order normal stress of the facet offset.

For a slip magnitude s(xi) the source part of the first-order kernel is multiplied by s, and the integration by parts
that cancels the geometry part for uniform slip (eq. dsigma2 of the paper) leaves
    Delta sigma_delta = (3/2) C HFP int delta X (X s_xi + Y s_eta) / R^5,      delta = h_I - h,
with h_I the piecewise-linear interpolant of h at the vertices (the flat triangles). With
FT[x_i x_j / |x|^5] = -(2 pi / 3) (k_i k_j / |k| + delta_ij |k|) its symbol is
    -pi C [(k_x^2 / |k| + |k|) FT(delta s_x) + (k_x k_y / |k|) FT(delta s_y)],
evaluated here by FFT on a grid through the centroids, and compared with the error of Woodland EXQ with the exact
slip at every quadrature point (EXP_ox of run_var.py), whose slip direction is exact (epsilon = 0).

    python3 check_geom_term.py [LEVELS] [REFINE]     e.g. 0,1 2  (grid spacing h / (3 * 2 * REFINE))
(after run_nn.py LEVEL EXP and run_var.py LEVEL EXP:x for the levels)
"""
import os
import sys
import numpy as np
import matplotlib.tri as mtri

import run_nn as R
import smallslope_var as SV

S = R.S
RES = os.path.join(R.HERE, "results")


def predicted(level, refine=2, pad=1.0):
    V, T = S.fault_grid(level, R.A, R.MRG)
    h = 1.0 / (8 * 2**level)
    dx = h / (6 * refine)  # centroids (3i + 1) h / 3 and (3i + 2) h / 3 are grid points
    n = int(round((1 + 2 * pad) / dx))
    x = -pad + dx * np.arange(n)
    Xg, Yg = np.meshgrid(x, x, indexing="ij")
    inD = (Xg >= 0) & (Xg <= 1) & (Yg >= 0) & (Yg <= 1)
    interp = mtri.LinearTriInterpolator(mtri.Triangulation(V[:, 0], V[:, 1], T), V[:, 2])
    hI = np.zeros_like(Xg)
    hI[inD] = interp(Xg[inD], Yg[inD]).filled(0.0)
    delta = np.where(inD, hI - S.height(Xg, Yg, R.A), 0.0)
    sx = -np.pi * np.sin(2 * np.pi * Xg) * np.cos(2 * np.pi * Yg)  # s = 1 + 0.5 cos 2 pi x cos 2 pi y
    sy = -np.pi * np.cos(2 * np.pi * Xg) * np.sin(2 * np.pi * Yg)
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K = np.hypot(KX, KY)
    K[0, 0] = 1.0
    F, G = np.fft.fft2(delta * sx), np.fft.fft2(delta * sy)
    sym = -np.pi * S.CC * ((KX**2 / K + K) * F + (KX * KY / K) * G)
    sym[0, 0] = 0.0
    ds = np.real(np.fft.ifft2(sym)) * 1.0  # dx^2 of the convolution sum and 1/dx^2 of ifft cancel
    c = V[T].mean(1)
    i = np.rint((c[:, 0] + pad) / dx).astype(int)
    j = np.rint((c[:, 1] + pad) / dx).astype(int)
    assert np.allclose(x[i], c[:, 0]) and np.allclose(x[j], c[:, 1])
    return c, ds[i, j]


def main(levels=(0, 1), refine=2):
    for lv in levels:
        z = np.load(os.path.join(RES, "woodland_nn_var", "l%d_A0.0001_c3_m0.25.npz" % lv))
        zu = np.load(os.path.join(RES, "woodland_nn", "l%d_A0.0001_c3_m0.25.npz" % lv))
        area, ref = z["area"], z["ref"]
        nrm = lambda e: np.sqrt((area * e**2).sum())
        err = z["EXP_ox"] - ref
        err_u = z["slip"] * (zu["EXP"] - zu["ref"])  # the uniform-slip error of EXQ, weighted by s
        for r in sorted({refine, 2 * refine}):
            c, ds = predicted(lv, r)
            assert np.allclose(c, z["xc"])
            print("level %d, grid h/%d: |err| %.3e, |pred| %.3e, |err - pred| %.3e, |err - pred - s err_u| %.3e "
                  "(relative to |ref|: err %.3e, err - pred %.3e); corr %.4f"
                  % (lv, 6 * r, nrm(err), nrm(ds), nrm(err - ds), nrm(err - ds - err_u), nrm(err) / nrm(ref),
                     nrm(err - ds) / nrm(ref), np.corrcoef(err, ds)[0, 1]), flush=True)


if __name__ == "__main__":
    lv = tuple(int(v) for v in sys.argv[1].split(",")) if len(sys.argv) > 1 else (0, 1)
    main(lv, int(sys.argv[2]) if len(sys.argv) > 2 else 2)
