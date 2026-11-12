#!/usr/bin/env python3
"""Plot a marmot fault geometry file (structured ni x nj point grid).

The points are fit by a plane (least squares). The figure shows
  (a) a map of the out-of-plane deviation over the in-plane coordinates,
  (b) a 3D view with the deviation exaggerated by a stated factor,
  (c) profiles of the deviation through the middle of the grid.
In-plane axes: u along the mean grid i direction, w = n x u, with n the
plane normal oriented as marmot's normal (X_s x X_t). For a file from
rough_fault_geom.py --frame yxz, u = x, w = depth below the top edge, and the
deviation is h.

Usage: plot_geom.py geom.txt [-o geom.png] [--exaggeration E] [--units m]
"""

import argparse
import sys

import numpy as np


def read_geom(filename):
    toks = []
    with open(filename) as f:
        for line in f:
            line = line.split('#', 1)[0]
            toks.extend(line.split())
    ni, nj = int(toks[0]), int(toks[1])
    P = np.array(toks[2:], dtype=float)
    if P.size != 3*ni*nj:
        raise ValueError('expected %d coordinates, got %d' % (3*ni*nj, P.size))
    return P.reshape(nj, ni, 3)


def plane_frame(P):
    c = P.reshape(-1, 3).mean(axis=0)
    _, _, Vt = np.linalg.svd(P.reshape(-1, 3) - c, full_matrices=False)
    n = Vt[2]
    # Orient n as marmot's normal, X_s x X_t.
    Xs = P[:, -1] - P[:, 0]
    Xt = P[-1, :] - P[0, :]
    if np.dot(np.cross(Xs.mean(axis=0), Xt.mean(axis=0)), n) < 0:
        n = -n
    u = Xs.mean(axis=0)
    u = u - np.dot(u, n)*n
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    return c, u, w, n


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('geom')
    p.add_argument('-o', '--out', default=None, help='output image '
                   '[GEOM basename + .png]')
    p.add_argument('--exaggeration', type=float, default=None,
                   help='vertical exaggeration of the 3D view [auto]')
    p.add_argument('--units', default='', help='length unit label, e.g. m')
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args(argv)

    try:
        import matplotlib
    except ImportError:
        sys.exit('plot_geom.py: matplotlib not found. On Sherlock run\n'
                 '  ml viz py-matplotlib/3.10.3_py312\n'
                 '  python3.12 plot_geom.py ...')
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    P = read_geom(args.geom)
    nj, ni, _ = P.shape
    c, u, w, n = plane_frame(P)
    Q = P - c
    U, W, H = Q @ u, Q @ w, Q @ n
    U -= U.min()
    W -= W.min()
    unit = (' [%s]' % args.units) if args.units else ''

    hmax = np.abs(H).max()
    extent = max(np.ptp(U), np.ptp(W))
    E = args.exaggeration
    if E is None:
        E = 0.05*extent/hmax if hmax > 0 else 1
    rms = H.std()
    print('%s: %d x %d points, plane normal (%.3f, %.3f, %.3f)'
          % (args.geom, ni, nj, n[0], n[1], n[2]))
    print('in-plane size %.4g x %.4g%s, deviation rms %.4g max|.| %.4g%s, '
          'rms/size %.3g' % (np.ptp(U), np.ptp(W), unit, rms, hmax, unit,
                             rms/extent))

    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], left=0.07, right=0.97,
                          top=0.92, bottom=0.07, hspace=0.28, wspace=0.12)
    title = args.geom.rsplit('/', 1)[-1]
    fig.suptitle('%s  (%d x %d points)' % (title, ni, nj), fontsize=13)

    # (a) map of the deviation.
    ax = fig.add_subplot(gs[0, 0])
    lim = hmax if hmax > 0 else 1
    pc = ax.pcolormesh(U, W, H, cmap='RdBu_r', vmin=-lim, vmax=lim,
                       shading='gouraud')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('u, along grid i' + unit)
    ax.set_ylabel('w, along grid j' + unit)
    ax.set_title('(a) out-of-plane deviation')
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('deviation along normal' + unit)

    # (b) 3D view, exaggerated.
    ax3 = fig.add_subplot(gs[0, 1], projection='3d')
    Pe = P + ((E - 1)*H)[..., None]*n  # original coordinates, exaggerated
    norm = plt.Normalize(-lim, lim)
    stride = max(1, max(ni, nj)//80)
    ax3.plot_surface(Pe[..., 0], Pe[..., 1], Pe[..., 2],
                     facecolors=plt.cm.RdBu_r(norm(H)), rstride=stride,
                     cstride=stride, linewidth=0, antialiased=False,
                     shade=True)
    ax3.set_xlabel('x' + unit)
    ax3.set_ylabel('y' + unit)
    ax3.set_zlabel('z' + unit)
    ax3.set_title('(b) 3D view, deviation exaggerated x%.3g' % E)
    span = np.ptp(Pe.reshape(-1, 3), axis=0)
    try:
        ax3.set_box_aspect(np.maximum(span, 0.02*span.max()))
    except AttributeError:
        pass
    # Axes that are thin even after exaggeration get no tick labels.
    for d, axis in enumerate((ax3.xaxis, ax3.yaxis, ax3.zaxis)):
        if span[d] < 0.1*span.max():
            axis.set_ticklabels([])
    ax3.view_init(elev=20, azim=-60)

    # (c) profiles through the middle.
    axp = fig.add_subplot(gs[1, :])
    jm, im = nj//2, ni//2
    axp.plot(U[jm, :], H[jm, :], lw=2, color='#2a6fbb',
             label='along u, at w = %.4g' % W[jm, 0])
    axp.plot(W[:, im], H[:, im], lw=2, color='#d1762b',
             label='along w, at u = %.4g' % U[0, im])
    axp.axhline(0, color='#999999', lw=0.8)
    axp.set_xlabel('distance' + unit)
    axp.set_ylabel('deviation' + unit)
    axp.set_title('(c) profiles through the middle of the grid')
    axp.legend(frameon=False)
    axp.grid(True, color='#e5e5e5', lw=0.6)
    for s in ('top', 'right'):
        axp.spines[s].set_visible(False)

    out = args.out or (args.geom.rsplit('.', 1)[0] + '.png')
    fig.savefig(out, dpi=args.dpi)
    print('wrote', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
