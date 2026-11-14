"""Nodal-normal test with Woodland on flat triangles: the smoothed hill (c3) with the margin of the boundary-element
test (HBI), the same fault triangulation, a uniform unit slip along the local x direction of the slip frame.

The slip frame of the flat elements (src/woodland/squirrel/ctzx_surface.cpp, FlatElementSurface) is built from
  FA   the facet normal (the slip in the element plane, Woodland as it is; uniform slip = constant slip per element)
  NN   the barycentric blend of the Max (1999) vertex normals
  QF   the same with the quadric-fit vertex normals
  EX   the same with the exact vertex normals
  EX6  the quadratic blend of the exact normals at the vertices and the edge midpoints (the analogue of EXN for p = 2)
  EXP  the exact normal at every point
so for NN, QF, EX, EX6 and EXP the slip vector is continuous across the element edges. The kernel integrates over the flat
triangles with their facet normal; the traction at the centroid is resolved along the normal of the slip frame there.

    python3 run_nn.py LEVEL [SCHEMES]   -> results/woodland_nn/l<LEVEL>_A0.0001_c3_m0.25.npz
    env WOODLAND_BIN  the woodland_squirrel_unittest of the patched build
                      (default ../build/woodland/squirrel/woodland_squirrel_unittest)
        OMP_NUM_THREADS (default 4), GH_A (default 1e-4)
The reference is the first-order small-slope solution at the centroids (smallslope.py, cached in results/ref).
"""
import os
import subprocess
import sys
import tempfile
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("GH_HILL", "c3")
sys.path.insert(0, HERE)
import smallslope as S  # noqa: E402
import smallslope_var as SV  # noqa: E402

BIN = os.environ.get("WOODLAND_BIN", os.path.join(HERE, "..", "build", "woodland", "squirrel",
                                                  "woodland_squirrel_unittest"))
OUT = os.path.join(HERE, "results", "woodland_nn")
A = float(os.environ.get("GH_A", "1e-4"))  # the reference is linear in A: that of A = 1e-4 scaled
MRG = S.MARGIN
SCHEMES = ("FA", "NN", "QF", "EX", "EX6", "EXP")
SIGN = 1.0  # Woodland dislocation = SIGN * (side of the normal minus the other side); checked against HBI FA


def vertex_normals(V, T):
    """Per-vertex Max, quadric-fit and exact normals (nv, 3) of the triangulation."""
    X = V[T]
    nf = np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0])
    nf /= np.linalg.norm(nf, axis=1)[:, None]
    nmax = S.max_normals(X, nf, T)
    nqf = S.quadric_normals(T, V, nmax)
    out = {}
    for k, c in (("NN", nmax), ("QF", nqf)):
        v = np.zeros_like(V)
        v[T.ravel()] = c.reshape(-1, 3)
        out[k] = v
    out["EX"] = S.exact_normal(V, A)
    mid = 0.5 * (X + np.roll(X, -1, 1))  # midpoints of the edges 01, 12, 20
    out["EX6"] = np.concatenate([S.exact_normal(X, A), S.exact_normal(mid, A)], 1)  # (nt, 6, 3)
    return out


def woodland(V, T, mode, N=None, dorder=2, threads=None, slip=None, slip_oracle=False):
    """Run nn_flat; returns the centroids (nt, 3) and the stress in the slip frame (nt, 6: xx xy xz yy yz zz).
    slip: the slip magnitude at the centroids (nt,), reconstructed by Woodland with the order dorder; None: 1."""
    hill = {"md": 0, "c3": 1}[S.HILL]
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
        with open(fin, "wb") as f:
            np.array([len(V), len(T), mode, dorder, hill, 2 if slip_oracle else int(slip is not None)],
                     np.int32).tofile(f)
            np.array([A, S.MU, S.MU], np.float64).tofile(f)
            np.ascontiguousarray(V, np.float64).tofile(f)
            np.ascontiguousarray(T, np.int32).tofile(f)
            if mode in (1, 3):
                np.ascontiguousarray(N, np.float64).tofile(f)
            if slip is not None:
                np.ascontiguousarray(slip, np.float64).tofile(f)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads or os.environ.get("OMP_NUM_THREADS", "4")))
        subprocess.run([BIN, "nn_flat", "in=%s,out=%s" % (fin, fout)], check=True, env=env)
        r = np.fromfile(fout, np.float64).reshape(-1, 9)
    return r[:, :3], SIGN * r[:, 3:]


def run(level, schemes=SCHEMES):
    t0 = time.time()
    V, T = S.fault_grid(level, A, MRG)
    X = V[T]
    c = X.mean(1)
    area = 0.5 * np.linalg.norm(np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), axis=1)
    cr, _, ref = SV.centroid_reference(level, 1e-4, MRG)
    assert np.allclose(cr[:, :2], c[:, :2])
    ref = ref * (A / 1e-4)  # the first-order solution is linear in A
    nrm2 = lambda e: np.sqrt((area * e**2).sum())
    vn = vertex_normals(V, T)
    fn = os.path.join(OUT, "l%d_A%g%s_m%g.npz" % (level, A, S.hill_tag(), MRG))
    res = dict(np.load(fn)) if os.path.exists(fn) else {}
    res.update(level=level, A=A, margin=MRG, nF=len(T), xc=c, area=area, ref=ref)
    for s in schemes:
        t1 = time.time()
        mode = {"FA": 0, "EXP": 2, "EX6": 3}.get(s, 1)
        cw, sig = woodland(V, T, mode, vn.get(s))
        assert np.allclose(cw, c)
        snn = sig[:, 5]
        e = snn - ref
        res[s] = snn
        res[s + "_t"] = sig[:, [2, 4]]
        res[s + "_full"] = nrm2(e) / nrm2(ref)
        print("[woodland-nn] level %d (%d elements) %s: sigma_nn L2 rel. error full %.3e; %.1f s"
              % (level, len(T), s, res[s + "_full"], time.time() - t1), flush=True)
        os.makedirs(OUT, exist_ok=True)
        np.savez(fn, **res)
    print("[woodland-nn] -> %s (%.0f s)" % (fn, time.time() - t0), flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]), tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else SCHEMES)
