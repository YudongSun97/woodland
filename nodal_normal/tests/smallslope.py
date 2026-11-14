"""First-order small-slope reference solution of the Gaussian-hill benchmark
(nodalnormal/analytical_numerical_model_comparison.md, testcase 70 of the Woodland boundary-element code), and the
geometry of the fault shared by the SIPG (../sipg) and the boundary-element (../bem) tests.

Fault: the graph z = h(x, y) over D = [0, 1]^2 with h = A [cos(2 pi x) - cos(4 pi x)] [2 - 2 cos(2 pi y)] (h and grad h
vanish on the boundary of D), mu = lambda = 1 (nu = 1/4), full space. The slip is uniform, of length 1, along the
local x direction of every point (e_x projected onto the tangent plane and normalized), as the displacement of the
upper side (the side n = (-h_x, -h_y, 1) points into) minus that of the lower side. The reference is the first-order
small-slope solution of the normal stress sigma_nn = n . sigma n (tension positive): sigma_nn = A sigma1(x, y) + O(A^3),

  sigma1 = C PV int_D [-X h_xx(s) - Y h_xy(s)] / R^3 ds - 2 h_x tau_x - 2 h_y tau_y + C h [E(x - 1) - E(x)],

per unit A, with C = mu / (4 pi (1 - nu)), X = x - xi, Y = y - eta, R^2 = X^2 + Y^2, tau_x and tau_y the shear
tractions sigma_xz, sigma_yz of the flat square with the same slip e_x, and E(a) = int_0^1 d eta / (a^2 + (y - eta)^2)^1.5
(the boundary term of the integration by parts). The shear tractions are closed-form contour integrals:
tau_alpha = C HFP int_D [(1 - nu) delta_ab / R^3 + nu d_a d_b (1 / R)] b_b ds. The principal value is the integral
of K (q(s) - q(x)) over four triangles from the target to the sides of D (Gauss in the radial fraction, Gauss after
t = d sinh(u) along the side, d the distance to the side), plus q(x) times the closed-form PV integral of K.

    python3 smallslope.py check       quadrature check of the reference (and of the flat shear tractions against a
                                      polar finite-part evaluation) at a few points
"""
import sys
import time
import numpy as np

NU = 0.25
MU = 1.0
CC = MU / (4 * np.pi * (1 - NU))
INNER = (0.125, 0.875)  # the interior region of the error norm, on mesh lines of every level
MARGIN = 0.25  # the slip (and the fault) covers [-m, 1 + m]^2, the hill D = [0, 1]^2 with the flat margin m around it


# ---------------------------------------------------------------- geometry
import os

# md: the hill of the md file, C^1 at the sides of D; c3: the smoothed hill, C^3 at the sides (env GH_HILL, set_hill)
HILL = os.environ.get("GH_HILL", "c3")


def hill_tag():
    """The tag of the hill in the names of the data and result files (none for md)."""
    return "" if HILL == "md" else "_" + HILL


def set_hill(name):
    """md: f = cos(2 pi x) - cos(4 pi x), g = 2 - 2 cos(2 pi y) (h and grad h vanish on the boundary of D, the
    curvature jumps there); c3: f sin^2(pi x) and (1 - cos(2 pi y))^2, which meet zero with three vanishing derivatives."""
    global HILL
    assert name in ("md", "c3")
    HILL = name


def _f0(x, k):
    a, b = 2 * np.pi, 4 * np.pi
    if k == 0:
        return np.cos(a * x) - np.cos(b * x)
    if k == 1:
        return -a * np.sin(a * x) + b * np.sin(b * x)
    return -a * a * np.cos(a * x) + b * b * np.cos(b * x)


def fx(x, k=0):
    """f(x) of the hill and its derivatives (k <= 2)."""
    if HILL == "md":
        return _f0(x, k)
    s, c, pi = np.sin(np.pi * x), np.cos(np.pi * x), np.pi
    w = (s * s, 2 * pi * s * c, 2 * pi * pi * (c * c - s * s))  # sin^2(pi x) and its derivatives
    if k == 0:
        return _f0(x, 0) * w[0]
    if k == 1:
        return _f0(x, 1) * w[0] + _f0(x, 0) * w[1]
    return _f0(x, 2) * w[0] + 2 * _f0(x, 1) * w[1] + _f0(x, 0) * w[2]


def gy(y, k=0):
    """g(y) of the hill and its derivatives (k <= 2)."""
    a = 2 * np.pi
    g0 = (2 - 2 * np.cos(a * y), 2 * a * np.sin(a * y), 2 * a * a * np.cos(a * y))
    if HILL == "md":
        return g0[k]
    v = (0.5 * g0[0], 0.5 * g0[1], 0.5 * g0[2])  # 1 - cos(2 pi y): g2 = v^2
    if k == 0:
        return v[0] ** 2
    if k == 1:
        return 2 * v[0] * v[1]
    return 2 * (v[1] ** 2 + v[0] * v[2])


def height(x, y, A, k=(0, 0)):
    """h or its derivative d^(k0 + k1) h / dx^k0 dy^k1 on D, zero outside."""
    inD = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    return np.where(inD, A * fx(x, k[0]) * gy(y, k[1]), 0.0)


def exact_normal(V, A):
    """Unit normal (-h_x, -h_y, 1) / |.| of the surface z = h(x, y) at the points V (..., 3) (their x, y)."""
    x, y = V[..., 0], V[..., 1]
    n = np.stack([-height(x, y, A, (1, 0)), -height(x, y, A, (0, 1)), np.ones_like(x)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def local_x(n):
    """Local x direction: e_x projected onto the plane normal to n, normalized."""
    e = np.zeros_like(n)
    e[..., 0] = 1.0
    t = e - (n[..., :1]) * n
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


# ---------------------------------------------------------------- reference solution
def flat_shear(x, y, m=0.0):
    """Shear tractions (tau_x, tau_y) = (sigma_xz, sigma_yz) on z = 0 of uniform slip e_x on the square
    S = [-m, 1 + m]^2 (upper minus lower side), at points inside S: C [-(1 - nu) S_n - nu S_xx], -C nu S_yx with the
    contour integrals S_n of rho . nu / R^3, S_xx of nu_x rho_x / R^3, S_yx of nu_y rho_x / R^3 (rho = s - x, nu the
    outward normal of S)."""
    def side(d, t0, t1):  # int a / (a^2 + t^2)^1.5 dt from t0 to t1, a = d > 0
        return (t1 / np.sqrt(d * d + t1 * t1) - t0 / np.sqrt(d * d + t0 * t0)) / d

    def odd(d, t0, t1):  # int t / (t^2 + d^2)^1.5 dt
        return -1 / np.sqrt(t1 * t1 + d * d) + 1 / np.sqrt(t0 * t0 + d * d)

    a, b = -m, 1 + m
    r_, l_ = side(b - x, a - y, b - y), side(x - a, a - y, b - y)
    t_, b_ = side(b - y, a - x, b - x), side(y - a, a - x, b - x)
    Sn = r_ + l_ + t_ + b_
    Sxx = r_ + l_
    Syx = odd(b - y, a - x, b - x) - odd(y - a, a - x, b - x)
    return CC * (-(1 - NU) * Sn - NU * Sxx), -CC * NU * Syx


def _pv_const(x, y):
    """PV int_D X / R^3 and PV int_D Y / R^3 over the hill D = [0, 1]^2 (closed form; an ordinary integral for
    points outside D)."""
    ash = lambda t, a: np.arcsinh(t / np.abs(a))
    PX = (ash(1 - y, 1 - x) - ash(-y, 1 - x)) - (ash(1 - y, x) - ash(-y, x))
    PY = (ash(1 - x, 1 - y) - ash(-x, 1 - y)) - (ash(1 - x, y) - ash(-x, y))
    return PX, PY


def sigma1(x, y, m=0.0, nr=20, nu=80, chunk=2000):
    """First-order normal stress sigma1 (per unit A) at targets (x, y) of the fault S = [-m, 1 + m]^2 with uniform slip
    on S (m = 0: the hill alone, the case of the md file); also returns the parts (curvature integral over D,
    receiver rotation, boundary term on the sides of S) and the flat shear tractions of S. The curvature integral is
    over the triangles from the target to the sides of D, signed when the target is outside D (the parts outside D
    cancel), with the curvature q of the trigonometric expression of h continued smoothly outside D for the
    subtraction of q at the target (h_xx = 0 outside D only enters through the domain of integration)."""
    x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    gr, wr = np.polynomial.legendre.leggauss(nr)
    gr, wr = 0.5 * (gr + 1), 0.5 * wr
    gu, wu = np.polynomial.legendre.leggauss(nu)
    q1 = lambda a, b: fx(a, 2) * gy(b, 0)  # h_xx / A
    q2 = lambda a, b: fx(a, 1) * gy(b, 1)  # h_xy / A
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rem = np.zeros(len(x))
    for c0 in range(0, len(x), chunk):
        xs, ys = x[c0:c0 + chunk], y[c0:c0 + chunk]
        r = np.stack([xs, ys], -1)
        q1r, q2r = q1(xs, ys), q2(xs, ys)
        acc = np.zeros(len(xs))
        for k in range(4):
            Pa, Pb = corners[k], corners[(k + 1) % 4]
            et = (Pb - Pa) / np.linalg.norm(Pb - Pa)
            ne = np.array([et[1], -et[0]])  # outward (corners counter-clockwise)
            d = (Pa - r) @ ne  # signed: negative where the target is on the outer side of the side line
            ad = np.maximum(np.abs(d), 1e-300)
            ta, tb = (Pa - r) @ et, (Pb - r) @ et
            ua, ub = np.arcsinh(ta / ad), np.arcsinh(tb / ad)
            u = 0.5 * (ua + ub)[:, None] + 0.5 * (ub - ua)[:, None] * gu[None, :]  # (n, nu)
            w_u = 0.5 * (ub - ua)[:, None] * wu[None, :]
            t = ad[:, None] * np.sinh(u)
            L = np.sqrt(d[:, None] ** 2 + t * t)
            vx = d[:, None] * ne[0] + t * et[0]  # (n, nu): the vector to the side point
            vy = d[:, None] * ne[1] + t * et[1]
            sx = xs[:, None, None] + gr[None, None, :] * vx[..., None]  # (n, nu, nr)
            sy = ys[:, None, None] + gr[None, None, :] * vy[..., None]
            X, Y = xs[:, None, None] - sx, ys[:, None, None] - sy
            R3 = (X * X + Y * Y) ** 1.5
            F = (-X * (q1(sx, sy) - q1r[:, None, None]) - Y * (q2(sx, sy) - q2r[:, None, None])) / R3
            # dA = rho d drho dt, dt = L du
            G = F * gr[None, None, :] * (d[:, None] * L)[..., None]
            acc += ein_sum(G, wr, w_u)
        rem[c0:c0 + chunk] = acc
    PX, PY = _pv_const(x, y)
    curv = CC * (rem - q1(x, y) * PX - q2(x, y) * PY)
    tx, ty = flat_shear(x, y, m)
    hx, hy = height(x, y, 1.0, (1, 0)), height(x, y, 1.0, (0, 1))
    rot = -2 * hx * tx - 2 * hy * ty
    a, b = -m, 1 + m
    E = lambda c: ((b - y) / np.sqrt(c * c + (b - y) ** 2) + (y - a) / np.sqrt(c * c + (y - a) ** 2)) / (c * c)
    bnd = CC * height(x, y, 1.0) * (E(x - b) - E(x - a))
    return curv + rot + bnd, dict(curv=curv, rot=rot, bnd=bnd, tx=tx, ty=ty)


def ein_sum(G, wr, w_u):
    return np.einsum("nur,r,nu->n", G, wr, w_u, optimize=True)


def check_reference(m=0.0):
    rng = np.random.default_rng(1)
    pts = np.r_[[[0.5, 0.5], [0.25, 0.5], [0.3, 0.7], [0.02, 0.5], [0.5, 0.003], [0.004, 0.006], [0.9, 0.95]],
                rng.uniform(0, 1, (5, 2))]
    if m > 0:  # targets in the flat margin, next to D and next to the edges of the fault
        pts = np.r_[pts, [[-0.003, 0.5], [0.5, 1.004], [-0.1, -0.1], [1.2, 0.4], [-0.245, 0.6], [0.3, 1.249],
                          [-0.249, -0.248]]]
    print("margin m = %g" % m)
    ref, part = sigma1(pts[:, 0], pts[:, 1], m, nr=48, nu=320)
    for nr, nu in ((12, 40), (16, 60), (20, 80), (24, 120), (32, 160)):
        t0 = time.time()
        s, _ = sigma1(pts[:, 0], pts[:, 1], m, nr=nr, nu=nu)
        print("nr %2d nu %3d: max |diff| %.2e (max |sigma1| %.3e), %.2f ms/point"
              % (nr, nu, np.abs(s - ref).max(), np.abs(ref).max(), 1e3 * (time.time() - t0) / len(pts)))
    for k, (xp, yp) in enumerate(pts):
        print("(%.3f, %.3f): sigma1 %+.6e  curv %+.4e rot %+.4e bnd %+.4e  tau_x %+.4e tau_y %+.4e"
              % (xp, yp, ref[k], part["curv"][k], part["rot"][k], part["bnd"][k], part["tx"][k], part["ty"][k]))


def check_flat_shear():
    """The closed-form flat shear tractions against the finite part in polar coordinates around the target:
    HFP int_D f(theta) / R^3 dA = -int_0^2pi f(theta) / R_D(theta) dtheta, R_D the distance to the boundary of D."""
    th = np.linspace(0, 2 * np.pi, 400001)[:-1] + 1e-7
    c, s = np.cos(th), np.sin(th)
    for x, y in [(0.5, 0.5), (0.18, 0.195), (0.152, 0.512), (0.305, 0.82), (0.82, 0.305)]:
        with np.errstate(divide="ignore"):
            tx_ = np.where(c > 0, (1 - x) / c, np.where(c < 0, -x / c, np.inf))
            ty_ = np.where(s > 0, (1 - y) / s, np.where(s < 0, -y / s, np.inf))
        R = np.minimum(tx_, ty_)
        tx = -CC * np.mean(((1 - 2 * NU) + 3 * NU * c * c) / R) * 2 * np.pi
        ty = -CC * np.mean((3 * NU * c * s) / R) * 2 * np.pi
        a, b = flat_shear(np.array(x), np.array(y))
        print("(%.3f, %.3f): tau_x %.6f (polar %.6f), tau_y %.6f (polar %.6f)" % (x, y, a, tx, b, ty))


# ---------------------------------------------------------------- vertex normals of a triangulated surface
def max_normals(X, n, key):
    """Max (1999) vertex normals of triangles with corners X (nF, 3, 3) and facet normals n (nF, 3): at every corner
    the normalised sum of the weighted normals of the triangles whose corner has the same key (nF, 3) (sipg.max_normals)."""
    ea, eb = np.roll(X, -1, 1) - X, np.roll(X, -2, 1) - X
    w = np.linalg.norm(np.cross(ea, eb), axis=-1) / ((ea * ea).sum(-1) * (eb * eb).sum(-1))
    _, end = np.unique(key.ravel(), return_inverse=True)
    acc = np.zeros((end.max() + 1, 3))
    np.add.at(acc, end, (w[..., None] * n[:, None, :]).reshape(-1, 3))
    acc /= np.linalg.norm(acc, axis=1)[:, None]
    return acc[end].reshape(-1, 3, 3)


def quadric_normals(v, X, nv):
    """Vertex normals of triangles with vertex indices v (nF, 3) into the points X from a least-squares quadric over
    the two-ring of every vertex (the QF normals of sipg.Problem.quadric_normals): w = a u^2 + b uv + c v^2 + d u + e v
    in the tangent frame of the Max normal nv (nF, 3, 3), normal (-d, -e, 1) in that frame, fitted once more in the
    frame of the new normal; one sided at the edges of the surface. Returns (nF, 3, 3)."""
    e = np.unique(np.sort(np.concatenate([v[:, [0, 1]], v[:, [1, 2]], v[:, [2, 0]]]), axis=1), axis=0)
    nb = {}
    for a, b in e:
        nb.setdefault(int(a), set()).add(int(b))
        nb.setdefault(int(b), set()).add(int(a))
    first = {}
    for f in range(len(v)):
        for k in range(3):
            first.setdefault(int(v[f, k]), nv[f, k])
    out = {}
    for q, n0 in first.items():
        ring = set(nb[q])
        for r in list(ring):
            ring |= nb[r]
        ring.discard(q)
        d = X[sorted(ring)] - X[q]
        n = n0
        for _ in range(2):
            t1 = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            u, w_, z = d @ t1, d @ t2, d @ n
            c = np.linalg.lstsq(np.stack([u * u, u * w_, w_ * w_, u, w_], 1), z, rcond=None)[0]
            n = n - c[3] * t1 - c[4] * t2
            n /= np.linalg.norm(n)
        out[q] = n * np.sign(n @ n0)
    return np.array([[out[int(q)] for q in row] for row in v])


def fault_grid(level, A, m=0.0):
    """The structured triangulation of the fault [-m, 1 + m]^2 of a mesh level (the fault surface of both tests):
    squares of edge 1 / (8 2^level), each cut along the diagonal (0, 0)-(1, 1), vertices lifted onto z = h(x, y). Returns
    the vertices V and the triangles T (counter-clockwise seen from +z, so that the facet normals point up)."""
    n = int(round((1 + 2 * m) * 8 * 2**level))
    g = np.linspace(-m, 1.0 + m, n + 1)
    X, Y = np.meshgrid(g, g, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), height(X.ravel(), Y.ravel(), A)], 1)
    idx = lambda i, j: i * (n + 1) + j
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    i, j = i.ravel(), j.ravel()
    T = np.concatenate([np.stack([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)], 1),
                        np.stack([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)], 1)])
    return V, T


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check_reference()
        check_reference(MARGIN)
        check_flat_shear()
