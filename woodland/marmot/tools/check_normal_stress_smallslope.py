#!/usr/bin/env python3
"""Check a marmot 'stress' result against the first-order small-slope model.

The model is that of extra_code/compute_zxy_direct_normal_stress.py: for a
nearly planar fault z' = h(x', y') with uniform slip u along x' (whole space),

    t_n = mu/[2 pi (lambda + 2 mu)] HFP int_D u N / R^5 dx' dy',
    N = 3(lam+mu) x (h_r - h_s) - mu (x^2 - 2 y^2) h_sx - 3 mu x y h_sy
        - ((3 lam + 2 mu) x^2 + 2 mu y^2) h_rx - 3 lam x y h_ry,

with x = x_r - x_s, y = y_r - y_s. The same quadrature is used: midpoint rule
for far cells, tensor Gauss for the 8 neighbors, and a centered four-triangle
Duffy rule with the cubic Taylor term subtracted for the self cell.

The geometry is a rough_fault_geom.py --frame yxz file: y = h(x, w), with w the
depth below the top edge. With the local basis (xhat, -zhat, yhat) this is
exactly the model's (x', y', z') (right handed, normal (-h_x, -h_w, 1)), and
marmot's lcs=x frame is the same basis, so the model's t_n is marmot's
receiver-LCS sigma_zz. h is band limited (rough_fault_geom.py with
lambda_min), so its FFT gives an exact trigonometric polynomial: h and its
derivatives are evaluated exactly at any point.

Usage:
  check_normal_stress_smallslope.py GEOM MARMOT_PREFIX [--gauss-order 12]
MARMOT_PREFIX.py is the output of 'woodland_marmot_unittest stress' with
slip=1:0:0 (lcs=x, halfspace=0); nx, ny, lam, mu are taken from it.
"""

import argparse
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss


def read_geom(filename):
    toks = []
    with open(filename) as f:
        for line in f:
            toks.extend(line.split('#', 1)[0].split())
    ni, nj = int(toks[0]), int(toks[1])
    P = np.array(toks[2:], dtype=float).reshape(nj, ni, 3)
    return P


class FourierSurface:
    """Exact trigonometric interpolant of h sampled on a uniform grid."""

    def __init__(self, h, dx, dw, tol=1e-13):
        nw, nx = h.shape
        H = np.fft.fft2(h)/(nx*nw)
        mx = np.rint(np.fft.fftfreq(nx)*nx).astype(int)
        mw = np.rint(np.fft.fftfreq(nw)*nw).astype(int)
        big = np.abs(H) > tol*np.abs(H).max()
        Mx = np.abs(mx[big.any(axis=0)]).max()
        Mw = np.abs(mw[big.any(axis=1)]).max()
        if 2*Mx + 1 > nx or 2*Mw + 1 > nw or (nx % 2 == 0 and Mx == nx//2) \
           or (nw % 2 == 0 and Mw == nw//2):
            raise ValueError('h is not band limited below the Nyquist '
                             'wavenumber; the Fourier interpolant is not exact')
        self.ix = np.arange(-Mx, Mx + 1)
        self.iw = np.arange(-Mw, Mw + 1)
        self.kx = 2*np.pi*self.ix/(nx*dx)
        self.kw = 2*np.pi*self.iw/(nw*dw)
        self.C = H[np.ix_(self.iw % nw, self.ix % nx)]  # (2Mw+1) x (2Mx+1)
        lost = np.sqrt(max(0.0, (np.abs(H)**2).sum() - (np.abs(self.C)**2).sum()))
        self.lost = lost/np.sqrt((np.abs(H)**2).sum())

    def eval(self, x, w, order=1, chunk=200000):
        """h, h_x, h_w (order 1) and also h_xx, h_xw, h_ww (order 2)."""
        x = np.asarray(x, dtype=float).ravel()
        w = np.asarray(w, dtype=float).ravel()
        nout = 3 if order == 1 else 6
        out = np.empty((nout, x.size))
        kx, kw = self.kx, self.kw
        for b in range(0, x.size, chunk):
            e = slice(b, min(b + chunk, x.size))
            Ex = np.exp(1j*np.outer(x[e], kx))          # n x Mx
            Ew = np.exp(1j*np.outer(w[e], kw))          # n x Mw
            A = Ew @ self.C                             # n x Mx
            out[0, e] = np.real(np.sum(A*Ex, axis=1))
            out[1, e] = np.real(np.sum(A*Ex*(1j*kx), axis=1))
            out[2, e] = np.real(np.sum((Ew*(1j*kw)) @ self.C*Ex, axis=1))
            if order == 2:
                out[3, e] = np.real(np.sum(A*Ex*(-kx*kx), axis=1))
                out[4, e] = np.real(np.sum((Ew*(1j*kw)) @ self.C*Ex*(1j*kx),
                                           axis=1))
                out[5, e] = np.real(np.sum((Ew*(-kw*kw)) @ self.C*Ex, axis=1))
        return out


def direct_numerator(x, y, hr, hs, hsx, hsy, hrx, hry, lam, mu):
    return (3*(lam + mu)*x*(hr - hs)
            - mu*(x*x - 2*y*y)*hsx
            - 3*mu*x*y*hsy
            - ((3*lam + 2*mu)*x*x + 2*mu*y*y)*hrx
            - 3*lam*x*y*hry)


def local_cubic_numerator(x, y, hxx, hxy, hyy, lam, mu):
    return (-0.5*x*((3*lam + mu)*x*x + 4*mu*y*y)*hxx
            + y*((mu - 3*lam)*x*x - 2*mu*y*y)*hxy
            + 1.5*(mu - lam)*x*y*y*hyy)


def make_duffy_rule(dx, dy, gp, gw):
    uv = 0.5*(gp + 1)
    uvw = 0.5*gw
    uu, vv = np.meshgrid(uv, uv, indexing='xy')
    uuw, vvw = np.meshgrid(uvw, uvw, indexing='xy')
    corners = np.array([[-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2],
                        [-dx/2, dy/2]])
    xq, yq, wq = [], [], []
    for t in range(4):
        a, b = corners[t], corners[(t + 1) % 4]
        pts = uu[..., None]*((1 - vv[..., None])*a + vv[..., None]*b)
        jac = uu*abs(a[0]*b[1] - a[1]*b[0])
        xq.append(pts[..., 0].ravel())
        yq.append(pts[..., 1].ravel())
        wq.append((uuw*vvw*jac).ravel())
    return np.concatenate(xq), np.concatenate(yq), np.concatenate(wq)


def model_normal_stress(surf, Lx, Lw, nx, nw, lam, mu, slip=1.0,
                        gauss_order=12, block=64):
    """The reference script's scheme on [0,Lx] x [0,Lw] with nx x nw cells."""
    dx, dw = Lx/nx, Lw/nw
    xa = (np.arange(nx) + 0.5)*dx
    wa = (np.arange(nw) + 0.5)*dw
    X, W = np.meshgrid(xa, wa, indexing='xy')
    xs, ws = X.ravel(), W.ravel()
    hs, hxs, hws = surf.eval(xs, ws)
    pref = mu/(2*np.pi*(lam + 2*mu))
    out = np.zeros(xs.size)
    area = dx*dw
    for b in range(0, xs.size, block):
        e = slice(b, min(b + block, xs.size))
        x = xs[e, None] - xs[None, :]
        y = ws[e, None] - ws[None, :]
        near = (np.abs(x) <= 1.01*dx) & (np.abs(y) <= 1.01*dw)
        r2 = x*x + y*y
        N = direct_numerator(x, y, hs[e, None], hs[None, :], hxs[None, :],
                             hws[None, :], hxs[e, None], hws[e, None], lam, mu)
        with np.errstate(divide='ignore', invalid='ignore'):
            I = N/(r2*r2*np.sqrt(r2))
        I[near] = 0
        out[e] = pref*slip*area*I.sum(axis=1)

    gp, gw = leggauss(gauss_order)
    qx, qy = np.meshgrid(gp, gp, indexing='xy')
    qwt = np.outer(gw, gw).ravel()*dx*dw/4
    qx = qx.ravel()*dx/2
    qy = qy.ravel()*dw/2
    sx, sy, swt = make_duffy_rule(dx, dw, gp, gw)
    hess = surf.eval(xs, ws, order=2)
    for j in range(nw):
        # All near/self quadrature points of one receiver row at once.
        pts_x, pts_w, owner, wts, is_self = [], [], [], [], []
        for i in range(nx):
            ir = j*nx + i
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    jj, ii = j + dj, i + di
                    if not (0 <= jj < nw and 0 <= ii < nx):
                        continue
                    if di == 0 and dj == 0:
                        px, pw, pwt = xs[ir] + sx, ws[ir] + sy, swt
                    else:
                        isrc = jj*nx + ii
                        px, pw, pwt = xs[isrc] + qx, ws[isrc] + qy, qwt
                    pts_x.append(px)
                    pts_w.append(pw)
                    wts.append(pwt)
                    owner.append(np.full(px.size, ir))
                    is_self.append(np.full(px.size, di == 0 and dj == 0))
        px = np.concatenate(pts_x)
        pw = np.concatenate(pts_w)
        pwt = np.concatenate(wts)
        own = np.concatenate(owner)
        slf = np.concatenate(is_self)
        hq, hxq, hwq = surf.eval(px, pw)
        x = xs[own] - px
        y = ws[own] - pw
        r2 = x*x + y*y
        N = direct_numerator(x, y, hs[own], hq, hxq, hwq, hxs[own], hws[own],
                             lam, mu)
        N[slf] -= local_cubic_numerator(x[slf], y[slf], hess[3, own[slf]],
                                        hess[4, own[slf]], hess[5, own[slf]],
                                        lam, mu)
        I = pwt*N/(r2*r2*np.sqrt(r2))
        out += pref*slip*np.bincount(own, weights=I, minlength=xs.size)
    return X, W, out.reshape(nw, nx)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('geom')
    p.add_argument('marmot_prefix')
    p.add_argument('--gauss-order', type=int, default=12)
    p.add_argument('--png', default=None, help='comparison figure '
                   '[PREFIX_vs_smallslope.png]')
    args = p.parse_args(argv)

    P = read_geom(args.geom)
    nj, ni, _ = P.shape
    x0 = P[0, :, 0]
    w0 = -P[:, 0, 2] - (-P[0, 0, 2])
    dx, dw = x0[1] - x0[0], w0[1] - w0[0]
    if not (np.allclose(P[..., 0], x0[None, :]) and
            np.allclose(-P[..., 2], -P[:, :1, 2])):
        sys.exit('geometry is not a rough_fault_geom.py --frame yxz grid')
    h = P[..., 1] - P[..., 1].mean()
    Lx, Lw = x0[-1] - x0[0], w0[-1]
    surf = FourierSurface(h, dx, dw)
    chk = surf.eval(*np.meshgrid(x0 - x0[0], w0, indexing='xy'))[0]
    print('Fourier surface: %d x %d modes, lost energy %.1e, max interp err '
          'at samples %.2e (max|h| %.3g)'
          % (surf.kx.size, surf.kw.size, surf.lost,
             np.abs(chk.reshape(nj, ni) - h).max(), np.abs(h).max()))

    sys.path.insert(0, '.')
    d = __import__(args.marmot_prefix).d
    nx, ny, lam, mu = d['nx'], d['ny'], d['lam'], d['mu']
    if d['lcs_primary'] != 'x' or d['halfspace'] or list(d['slip']) != [1, 0, 0]:
        print('warning: the model assumes lcs=x, halfspace=0, slip=1:0:0')
    Sm = d['sigma'][:, 5].reshape(ny, nx)

    X, W, Sa = model_normal_stress(surf, Lx, Lw, nx, ny, lam, mu,
                                   gauss_order=args.gauss_order)
    # Receiver positions: marmot's are at uniform chord-length parameters.
    cx = d['ctr'][:, 0].reshape(ny, nx) - x0[0]
    cw = -d['ctr'][:, 2].reshape(ny, nx) + P[0, 0, 2]
    print('receiver offset marmot vs model: max %.3g m (cell %.3g m)'
          % (max(np.abs(cx - X).max(), np.abs(cw - W).max()), Lx/nx))

    def stats(A, B, sl):
        a, b = A[sl].ravel(), B[sl].ravel()
        rel = np.linalg.norm(a - b)/np.linalg.norm(b)
        s = np.dot(a, b)/np.dot(b, b)
        return rel, np.corrcoef(a, b)[0, 1], s

    for name, sl in [('all cells', np.s_[:, :]),
                     ('interior (2 cells from edges)', np.s_[2:-2, 2:-2])]:
        rel, r, s = stats(Sm, Sa, sl)
        print('%-30s rel L2 diff %.4f  corr %.5f  best scale marmot = %.4f '
              'x model' % (name, rel, r, s))
    print('rms: marmot %.4g  model %.4g' % (np.sqrt(np.mean(Sm**2)),
                                           np.sqrt(np.mean(Sa**2))))

    np.savez(args.marmot_prefix + '_smallslope.npz', X=X, W=W, model=Sa,
             marmot=Sm, lam=lam, mu=mu)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not found; skipped the figure')
        return 0
    lim = max(np.abs(Sm).max(), np.abs(Sa).max())
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, (A, t) in zip(axs, [(Sm, 'marmot (curved Bezier, dislocorder=2)'),
                                (Sa, 'first-order small-slope model'),
                                (Sm - Sa, 'marmot - model')]):
        pc = ax.pcolormesh(X, W, A, cmap='RdBu_r', vmin=-lim, vmax=lim,
                           shading='auto')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('depth [m]')
        ax.set_title(t, fontsize=11)
    fig.colorbar(pc, ax=axs, label='sigma_n [Pa]', fraction=0.02, pad=0.01)
    rel, r, s = stats(Sm, Sa, np.s_[:, :])
    fig.suptitle('%s vs small-slope model (%d x %d cells): rel L2 diff %.2f%%,'
                 ' corr %.4f' % (args.marmot_prefix, nx, ny, 100*rel, r))
    out = args.png or (args.marmot_prefix + '_vs_smallslope.png')
    fig.savefig(out, dpi=130)
    print('wrote', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
