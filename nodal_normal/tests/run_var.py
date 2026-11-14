"""Woodland on flat triangles with a non-uniform slip s(x, y) (smallslope_var.slip) along the local x
direction of the slip frame: the magnitudes at the centroids are reconstructed by Woodland with the order 0, 1 or 2
(dislocorder), the slip frame as in run_nn.py.

    python3 run_var.py LEVEL [SCHEME:ORDER,...]   -> results/woodland_nn_var/l<LEVEL>_A0.0001_c3_m0.25.npz
    default: FA, NN, QF, EX, EX6, EXP with order 2, EX6 with orders 0 and 1, EXP with orders 0, 1, 3 and x
    (order x: the exact slip magnitude at every quadrature point, an oracle)
"""
import os
import sys
import time
import numpy as np

import run_nn as R
import smallslope_var as SV

S = R.S
OUT = os.path.join(R.HERE, "results", "woodland_nn_var")
RUNS = ["%s:2" % s for s in R.SCHEMES] + ["EX6:0", "EX6:1", "EXP:0", "EXP:1", "EXP:3", "EXP:x"]


def run(level, runs=RUNS):
    t0 = time.time()
    V, T = S.fault_grid(level, R.A, R.MRG)
    X = V[T]
    area = 0.5 * np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)
    c, ref, _ = SV.centroid_reference(level, R.A, R.MRG)
    s_c = SV.slip(c[:, 0], c[:, 1])
    nrm2 = lambda e: np.sqrt((area * e**2).sum())
    vn = R.vertex_normals(V, T)
    fn = os.path.join(OUT, "l%d_A%g%s_m%g.npz" % (level, R.A, S.hill_tag(), R.MRG))
    res = dict(np.load(fn)) if os.path.exists(fn) else {}
    res.update(level=level, A=R.A, margin=R.MRG, nF=len(T), xc=c, area=area, ref=ref, slip=s_c)
    for run_ in runs:
        s, order = run_.split(":")  # order "x": the exact slip magnitude at every quadrature point (oracle)
        t1 = time.time()
        mode = {"FA": 0, "EXP": 2, "EX6": 3}.get(s, 1)
        cw, sig = R.woodland(V, T, mode, vn.get(s), dorder=0 if order == "x" else int(order), slip=s_c,
                             slip_oracle=order == "x")
        assert np.allclose(cw, c)
        k = "%s_o%s" % (s, order)
        res[k] = sig[:, 5]
        res[k + "_full"] = nrm2(sig[:, 5] - ref) / nrm2(ref)
        print("[woodland-var] level %d (%d elements) %s, slip order %s: sigma_nn L2 rel. error full %.3e; %.1f s"
              % (level, len(T), s, order, res[k + "_full"], time.time() - t1), flush=True)
        os.makedirs(OUT, exist_ok=True)
        np.savez(fn, **res)
    print("[woodland-var] -> %s (%.0f s)" % (fn, time.time() - t0), flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2].split(",") if len(sys.argv) > 2 else RUNS)
