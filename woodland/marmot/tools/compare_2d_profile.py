#!/usr/bin/env python3
"""Compare 3D normal stress on a rough fault with 2D models of its x profiles.

3D: sigma_n from a marmot 'stress' output (uniform slip along x, lcs=x,
whole space) on a rough_fault_geom.py --frame yxz geometry y = h(x, w).

2D: each profile h(x, w0) is extruded infinitely along depth (plane strain)
with the same uniform slip u along x. Integrating the 3D first-order
small-slope kernel (check_normal_stress_smallslope.py) over the extrusion
direction gives
    sigma_n(x) = mu u / [pi (1 - nu)]
                 PV int_0^L [h(x) - h(s) - (x - s) h'(x)] / (x - s)^3 ds,
a Cauchy principal value; for one Fourier mode it is
    sigma_n = -mu u / [2 (1 - nu)] i k |k| h_k,
i.e. -mu u / [2 (1 - nu)] times the Hilbert transform of h''. The finite
segment [0, L] matches the 3D fault's extent in x; the periodic (spectral)
version is printed as a check of the interior.

Usage: compare_2d_profile.py GEOM MARMOT_PREFIX
"""

import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from check_normal_stress_smallslope import FourierSurface, read_geom  # noqa


def sigma_n_2d_segment(hfun, L, xr, mu, nu, u=1.0, panel=0.5, ng=8):
    """Finite-segment 2D sigma_n at receivers xr (each a multiple of panel)."""
    npan = int(round(L/panel))
    gp, gw = leggauss(ng)
    edges = np.linspace(0, L, npan + 1)
    xs = (0.5*(edges[:-1] + edges[1:])[:, None] + 0.5*panel*gp[None, :]).ravel()
    ws = np.tile(0.5*panel*gw, npan)
    hs, _, _ = hfun(xs)
    hr, hxr, hxxr = hfun(xr)
    out = np.empty(xr.size)
    for k in range(xr.size):
        t = xr[k] - xs
        num = hr[k] - hs - t*hxr[k]
        # Subtract the 1/t part, -h''(x)/(2 t); its PV over [0, L] is analytic.
        reg = (num + 0.5*hxxr[k]*t*t)/t**3
        pv = np.log(xr[k]/(L - xr[k]))
        out[k] = np.dot(ws, reg) - 0.5*hxxr[k]*pv
    return mu*u/(np.pi*(1 - nu))*out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit(__doc__)
    geom, prefix = argv
    P = read_geom(geom)
    x0 = P[0, :, 0]
    w0 = -P[:, 0, 2] + P[0, 0, 2]
    dx, dw = x0[1] - x0[0], w0[1] - w0[0]
    h = P[..., 1] - P[..., 1].mean()
    L, Lw = x0[-1] - x0[0], w0[-1]
    surf = FourierSurface(h, dx, dw)

    sys.path.insert(0, '.')
    d = __import__(prefix).d
    nx, ny, lam, mu = d['nx'], d['ny'], d['lam'], d['mu']
    nu = lam/(2*(lam + mu))
    S3 = d['sigma'][:, 5].reshape(ny, nx)
    xc = (np.arange(nx) + 0.5)*L/nx
    wc = (np.arange(ny) + 0.5)*Lw/ny
    cell = (L/nx)*(Lw/ny)

    def rms(a):
        return np.sqrt(np.mean(np.square(a)))

    inner = np.s_[2:-2, 2:-2]
    print('3D (marmot, %d x %d cells, %g x %g m, mu %g, nu %g, slip 1 m):'
          % (nx, ny, L, Lw, mu, nu))
    print('  sigma_n RMS %.4g Pa (interior %.4g), L2 norm sqrt(int sigma^2 dA)'
          ' %.4g Pa m' % (rms(S3), rms(S3[inner]),
                         np.sqrt(np.sum(S3**2)*cell)))

    # 2D for every receiver row.
    S2 = np.empty((ny, nx))
    for j in range(ny):
        def hfun(x, w=wc[j]):
            ww = np.full_like(x, w)
            v = surf.eval(x, ww, order=2)
            return v[0], v[1], v[3]
        S2[j] = sigma_n_2d_segment(hfun, L, xc, mu, nu)
    jm = ny//2
    # Periodic spectral check of one profile, in the interior.
    nf = 4096
    per = 2*np.pi/surf.kx[surf.ix == 1][0]  # period of the FFT field
    xf = np.arange(nf)*per/nf
    hf = surf.eval(xf, np.full(nf, wc[jm]))[0]
    hk = np.fft.fft(hf)
    k = 2*np.pi*np.fft.fftfreq(nf, per/nf)
    sp = np.real(np.fft.ifft(-mu/(2*(1 - nu))*1j*k*np.abs(k)*hk))
    spc = np.interp(xc, xf, sp)
    mid = slice(nx//4, 3*nx//4)
    print('2D check, profile at w = %.4g m, middle half in x: finite segment vs'
          ' periodic spectral rel diff %.3f' % (wc[jm], np.linalg.norm(
              (S2[jm] - spc)[mid])/np.linalg.norm(spc[mid])))

    print('2D (plane strain, each x profile extruded along depth, segment '
          '[0, %g] m):' % L)
    print('  profile at w = %.4g m: RMS %.4g Pa (interior %.4g), L2 norm '
          'sqrt(int sigma^2 dx) %.4g Pa m^0.5'
          % (wc[jm], rms(S2[jm]), rms(S2[jm, 2:-2]),
             np.sqrt(np.sum(S2[jm]**2)*L/nx)))
    print('  all %d profiles: RMS %.4g Pa (interior %.4g); per-profile RMS '
          'min %.4g median %.4g max %.4g'
          % (ny, rms(S2), rms(S2[inner]), np.sqrt(np.mean(S2**2, axis=1)).min(),
             np.median(np.sqrt(np.mean(S2**2, axis=1))),
             np.sqrt(np.mean(S2**2, axis=1)).max()))
    print('  3D on the same profile row: RMS %.4g Pa' % rms(S3[jm]))
    print('ratio RMS 2D / 3D: all rows %.3f, interior %.3f, middle profile %.3f'
          % (rms(S2)/rms(S3), rms(S2[inner])/rms(S3[inner]),
             rms(S2[jm])/rms(S3[jm])))
    r = np.corrcoef(S2[inner].ravel(), S3[inner].ravel())[0, 1]
    print('corr(2D, 3D) over interior cells %.3f' % r)
    np.savez(prefix + '_2d_profiles.npz', x=xc, w=wc, sigma_2d=S2,
             sigma_3d=S3)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return 0
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.3,
                          wspace=0.15, left=0.06, right=0.95, top=0.92,
                          bottom=0.07)
    lim = max(np.abs(S3).max(), np.abs(S2).max())
    X, W = np.meshgrid(xc, wc)
    for c, (A, t) in enumerate([(S3, '3D rough fault (marmot)'),
                                (S2, '2D: each x profile, plane strain')]):
        ax = fig.add_subplot(gs[0, c])
        pc = ax.pcolormesh(X, W, A, cmap='RdBu_r', vmin=-lim, vmax=lim,
                           shading='auto')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('depth [m]')
        ax.set_title('%s: RMS %.3g MPa' % (t, rms(A)/1e6), fontsize=11)
    fig.colorbar(pc, ax=fig.axes, label='sigma_n [Pa]', fraction=0.02,
                 pad=0.01)
    ax = fig.add_subplot(gs[1, :])
    ax.plot(xc, S3[jm]/1e6, lw=2, color='#2a6fbb', label='3D, row at w = %.4g m'
            % wc[jm])
    ax.plot(xc, S2[jm]/1e6, lw=2, color='#d1762b', label='2D, same profile')
    ax.axhline(0, color='#999999', lw=0.8)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('sigma_n [MPa]')
    ax.legend(frameon=False)
    ax.grid(True, color='#e5e5e5', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.suptitle('normal stress, uniform 1 m slip along x: 3D vs 2D profiles')
    out = prefix + '_2d_vs_3d.png'
    fig.savefig(out, dpi=130)
    print('wrote', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
