"""First-order normal stress on the hill for a non-uniform slip of magnitude s(x, y) along the local x direction.

The first-order kernel K of the paper (appendix, eq. kernelnu) is linear in the Burgers vector, so
    A sigma1_s(x) = C HFP int_S s(xi) K(x, xi) / R^5 = s(x) A sigma1(x) + C int_S [s(xi) - s(x)] K(x, xi) / R^5,
with sigma1 the uniform-slip solution of smallslope.py. K = O(R^3) at the receiver (the quadratic part vanishes: a
tilted plane has no normal stress), so the second integrand is O(1/R) and is an ordinary integral, evaluated over the
triangles from the target to the sides of S as in smallslope.sigma1.

    s(x, y) = 1 + 0.5 cos(2 pi x) cos(2 pi y)   (s = 1 on the sides of S for m = 0.25: the edges are those of the
                                                  uniform case)

    python3 smallslope_var.py check     the kernel against smallslope.sigma1 (s = 1) and the decomposition against a
                                        direct principal value (non-uniform s), by rays paired through the target
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smallslope as S  # noqa: E402


def slip(x, y):
    """The slip magnitude s(x, y)."""
    return 1.0 + 0.5 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)


def kernel(x, y, xi, eta):
    """K / A of the first-order normal stress (eq. kernelnu) for a source at (xi, eta) and a receiver at (x, y)."""
    X, Y = x - xi, y - eta
    R2 = X * X + Y * Y
    nu = S.NU
    hr, hs = S.height(x, y, 1.0), S.height(xi, eta, 1.0)
    hsx, hsy = S.height(xi, eta, 1.0, (1, 0)), S.height(xi, eta, 1.0, (0, 1))
    hrx, hry = S.height(x, y, 1.0, (1, 0)), S.height(x, y, 1.0, (0, 1))
    return (3 * X * (hr - hs) + (1 - 2 * nu) * ((2 * Y * Y - X * X) * hsx - 3 * X * Y * hsy)
            - 2 * ((1 - 2 * nu) * R2 + 3 * nu * X * X) * hrx - 6 * nu * X * Y * hry)


def correction(x, y, m, sfun=slip, nr=20, nu=80, chunk=1000):
    """C int_S [s(xi) - s(x)] K / R^5 (per unit A) at targets (x, y) inside S = [-m, 1 + m]^2."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    gr, wr = np.polynomial.legendre.leggauss(nr)
    gr, wr = 0.5 * (gr + 1), 0.5 * wr
    gu, wu = np.polynomial.legendre.leggauss(nu)
    a, b = -m, 1.0 + m
    corners = np.array([[a, a], [b, a], [b, b], [a, b]])
    out = np.zeros(len(x))
    for c0 in range(0, len(x), chunk):
        xs, ys = x[c0:c0 + chunk], y[c0:c0 + chunk]
        r = np.stack([xs, ys], -1)
        sr = sfun(xs, ys)
        acc = np.zeros(len(xs))
        for k in range(4):
            Pa, Pb = corners[k], corners[(k + 1) % 4]
            et = (Pb - Pa) / np.linalg.norm(Pb - Pa)
            ne = np.array([et[1], -et[0]])
            d = np.maximum((Pa - r) @ ne, 1e-300)  # targets inside S
            ta, tb = (Pa - r) @ et, (Pb - r) @ et
            ua, ub = np.arcsinh(ta / d), np.arcsinh(tb / d)
            u = 0.5 * (ua + ub)[:, None] + 0.5 * (ub - ua)[:, None] * gu[None, :]
            w_u = 0.5 * (ub - ua)[:, None] * wu[None, :]
            t = d[:, None] * np.sinh(u)
            L = np.sqrt(d[:, None] ** 2 + t * t)
            vx = d[:, None] * ne[0] + t * et[0]
            vy = d[:, None] * ne[1] + t * et[1]
            sx = xs[:, None, None] + gr[None, None, :] * vx[..., None]
            sy = ys[:, None, None] + gr[None, None, :] * vy[..., None]
            X, Y = xs[:, None, None] - sx, ys[:, None, None] - sy
            R5 = (X * X + Y * Y) ** 2.5
            F = (sfun(sx, sy) - sr[:, None, None]) * kernel(xs[:, None, None], ys[:, None, None], sx, sy) / R5
            G = F * gr[None, None, :] * (d[:, None] * L)[..., None]
            acc += S.ein_sum(G, wr, w_u)
        out[c0:c0 + chunk] = acc
    return S.CC * out


def sigma1_var(x, y, m=0.0, sfun=slip, nr=20, nu=80):
    """First-order normal stress per unit A for the slip magnitude sfun on S = [-m, 1 + m]^2."""
    u, _ = S.sigma1(x, y, m)
    return sfun(np.asarray(x, float).ravel(), np.asarray(y, float).ravel()) * u + correction(x, y, m, sfun, nr, nu)


def centroid_reference(level, A=1e-4, m=S.MARGIN):
    """The first-order normal stress (times A) at the centroids of the fault triangulation of a level for the
    non-uniform slip (ref) and for uniform slip (ref_uniform), cached in results/ref."""
    here = os.path.dirname(os.path.abspath(__file__))
    tag = "l%d_A%g%s_m%g.npz" % (level, A, S.hill_tag(), m)
    fn = os.path.join(here, "results", "ref", tag)
    V, T = S.fault_grid(level, A, m)
    c = V[T].mean(1)
    if os.path.exists(fn):
        z = np.load(fn)
        assert np.allclose(z["xc"], c)
        return c, z["ref"], z["ref_uniform"]
    t0 = time.time()
    u, _ = S.sigma1(c[:, 0], c[:, 1], m)
    ref_u = A * u
    ref = slip(c[:, 0], c[:, 1]) * ref_u + A * correction(c[:, 0], c[:, 1], m, nr=32, nu=160)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez(fn, xc=c, ref=ref, ref_uniform=ref_u, slip=slip(c[:, 0], c[:, 1]))
    print("[ref] level %d: %d centroids, %.0f s -> %s" % (level, len(c), time.time() - t0, fn), flush=True)
    return c, ref, ref_u


def pv_direct(x, y, m, sfun, nth=400, nt=200):
    """C PV int_S s K / R^5 at one target by rays through it: for every direction theta in [0, pi) the two opposite
    rays are paired up to the shorter one (their leading O(1/R^2) parts cancel), the rest of the longer ray is
    regular. Gauss in theta between the directions of the corners of S, Gauss in the distance."""
    a, b = -m, 1.0 + m
    ang = np.sort(np.mod(np.arctan2(np.array([a, a, b, b]) - y, np.array([a, b, b, a]) - x), np.pi))
    cuts = np.r_[0.0, ang, np.pi]
    gth, wth = np.polynomial.legendre.leggauss(nth)
    gt, wt = np.polynomial.legendre.leggauss(nt)
    gt, wt = 0.5 * (gt + 1), 0.5 * wt

    def ray_len(c, s_):
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(c > 0, (b - x) / c, np.where(c < 0, (a - x) / c, np.inf))
            ty = np.where(s_ > 0, (b - y) / s_, np.where(s_ < 0, (a - y) / s_, np.inf))
        return np.minimum(tx, ty)

    F = lambda px, py: sfun(px, py) * kernel(x, y, px, py) / ((x - px) ** 2 + (y - py) ** 2) ** 2.5
    tot = 0.0
    for t0, t1 in zip(cuts[:-1], cuts[1:]):
        if t1 - t0 < 1e-14:
            continue
        th = 0.5 * (t0 + t1) + 0.5 * (t1 - t0) * gth
        w = 0.5 * (t1 - t0) * wth
        c, s_ = np.cos(th), np.sin(th)
        R1, R2 = ray_len(c, s_), ray_len(-c, -s_)
        Rm = np.minimum(R1, R2)
        tp = Rm[:, None] * gt[None, :]
        pair = (F(x + tp * c[:, None], y + tp * s_[:, None]) + F(x - tp * c[:, None], y - tp * s_[:, None])) * tp
        I = (pair * wt[None, :]).sum(1) * Rm
        longer = np.where(R1 > R2, 1.0, -1.0)
        Rl = np.maximum(R1, R2)
        tl = Rm[:, None] + (Rl - Rm)[:, None] * gt[None, :]
        rest = F(x + longer[:, None] * tl * c[:, None], y + longer[:, None] * tl * s_[:, None]) * tl
        I += (rest * wt[None, :]).sum(1) * (Rl - Rm)
        tot += (I * w).sum()
    return S.CC * tot


def check(m=S.MARGIN):
    pts = np.array([[0.5, 0.5], [0.3, 0.7], [0.12, 0.43], [0.81, 0.27], [0.95, 0.6], [-0.1, 0.5], [0.5, 1.1]])
    one = lambda px, py: np.ones_like(np.asarray(px, float))
    u, _ = S.sigma1(pts[:, 0], pts[:, 1], m, nr=48, nu=320)
    print("hill %s, margin %g" % (S.HILL, m))
    print("s = 1: smallslope.sigma1 against the direct principal value of the kernel")
    for k, (px, py) in enumerate(pts):
        dv = pv_direct(px, py, m, one)
        print("  (%.2f, %.2f): sigma1 %+.6e  direct %+.6e  diff %.1e" % (px, py, u[k], dv, abs(dv - u[k])))
    print("non-uniform s: decomposition against the direct principal value")
    for nr, nu in ((12, 40), (20, 80), (32, 160)):
        t0 = time.time()
        v = sigma1_var(pts[:, 0], pts[:, 1], m, nr=nr, nu=nu)
        print("  nr %d nu %d (%.2f ms/point):" % (nr, nu, 1e3 * (time.time() - t0) / len(pts))
              + " ".join("%+.6e" % z for z in v))
    for k, (px, py) in enumerate(pts):
        dv = pv_direct(px, py, m, slip)
        print("  (%.2f, %.2f): direct %+.6e  decomposition %+.6e  diff %.1e" % (px, py, dv, v[k], abs(dv - v[k])))


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
