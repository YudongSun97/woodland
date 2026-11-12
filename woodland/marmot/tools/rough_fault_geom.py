#!/usr/bin/env python3
"""Generate a rough fault geometry file (geom.txt) for woodland marmot.

The roughness h(z,x) is generated as in extra_code/rough3D_fft_minlambda_twoH.m:
an anisotropic self-affine random field with Hurst exponents Hx, Hz built by
filtering white noise in the Fourier domain,

    lx = 1/Hx, lz = 1/Hz
    rho = sqrt(|qx|^(2/lx) + |aniso*qz|^(2/lz))
    amplitude spectrum ~ rho^(-(1 + (lx+lz)/2)),

with qx, qz the wavenumbers normalized by their fundamental modes, optional
shortest wavelengths lambda_min_x, lambda_min_z, zero mean, and RMS alpha.
Alternatively, read h from a file (e.g. written by MATLAB with writematrix).

x in [0,Lx] is the along-strike coordinate and z in [0,Lz] the down-dip
distance from the top edge, both in the fault's reference plane; h is the
offset normal to that plane. The rough surface is written as a structured
Nx x Nz point grid (see marmot's GridGeometry::read) in one of two frames:

  --frame yxz (default): vertical fault y = y0 + h(x,z), as squirrel's
      surface=yxz. Point (x, y0 + h, -(ztop + z)). Grid i is along +x and j
      down dip, so marmot's normal is +y. With marmot lcs=x the local frame is
      squirrel's yxz frame: (along x, down depth, normal +y).

  --frame dip: plane with strike azimuth phi (degrees clockwise from north)
      and dip delta, in east-north-up coordinates (x east, y north, z up),
      top edge through (x0, y0, -ztop):
          strike  s = (sin phi, cos phi, 0)
          downdip d = (cos delta cos phi, -cos delta sin phi, -sin delta)
          normal  n = (sin delta cos phi, -sin delta sin phi, cos delta)
      (s, d, n follow the right-hand rule: the fault dips to the right of s,
      and n points up into the hanging wall). Point O + x s + z d + h n.
      Grid i is along s and j runs UP dip (the first row is the bottom
      edge), so marmot's normal is n, and marmot lcs=strike gives
      (strike, up dip, hanging-wall normal), the Aki-Richards frame.

Examples:
  rough_fault_geom.py --Hx 0.8 --Hz 0.8 --Nx 129 --Nz 65 --Lx 2 --Lz 1 \\
      --alpha 0.01 --lambda-min-x 0.1 --lambda-min-z 0.1 --seed 1 -o geom.txt
  rough_fault_geom.py --h-file h.csv --Lx 2 --Lz 1 --frame dip --strike 30 \\
      --dip 60 --ztop 0.1 -o geom.txt
"""

import argparse
import math
import sys

import numpy as np


def rough3d_fft_minlambda_twoh(Hx, Hz, Nx, Nz, Lx, Lz, alpha,
                               lambda_min_x=None, lambda_min_z=None,
                               aniso_ratio=1.0, seed=None):
    """Port of rough3D_fft_minlambda_twoH.m. Returns h (Nz x Nx), x, z, info.

    The spectral model matches the MATLAB function exactly; the random numbers
    do not (numpy's generator differs from MATLAB's rng), so the same seed
    gives a different, statistically equivalent field. To reproduce a MATLAB
    field, write it with writematrix(h,'h.csv') and use --h-file.
    """
    if not (0 < Hx < 1 and 0 < Hz < 1):
        raise ValueError('need 0 < Hx, Hz < 1')
    if Nx < 3 or Nz < 3:
        raise ValueError('need Nx, Nz >= 3')
    if not (Lx > 0 and Lz > 0):
        raise ValueError('need Lx, Lz > 0')
    if alpha < 0:
        raise ValueError('need alpha >= 0')
    if not aniso_ratio > 0:
        raise ValueError('need aniso_ratio > 0')

    rng = np.random.RandomState(seed)

    dx = Lx/(Nx - 1)
    dz = Lz/(Nz - 1)
    x = np.linspace(0, Lx, Nx)
    z = np.linspace(0, Lz, Nz)

    # Unshifted FFT wavenumbers in rad/unit length.
    kx = 2*np.pi*np.fft.ifftshift(
        np.arange(-(Nx//2), int(math.ceil(Nx/2)))) / (Nx*dx)
    kz = 2*np.pi*np.fft.ifftshift(
        np.arange(-(Nz//2), int(math.ceil(Nz/2)))) / (Nz*dz)
    KX, KZ = np.meshgrid(kx, kz)  # Nz x Nx, as MATLAB's meshgrid

    # Normalize by the fundamental modes before applying different powers.
    QX = np.abs(KX)/(2*np.pi/(Nx*dx))
    QZ = np.abs(KZ)/(2*np.pi/(Nz*dz))
    QZ_eff = aniso_ratio*QZ

    lx = 1/Hx
    lz = 1/Hz
    rho = np.sqrt(QX**(2/lx) + QZ_eff**(2/lz))
    with np.errstate(divide='ignore'):
        spectral_filter = rho**(-(1 + (lx + lz)/2))
    spectral_filter[0, 0] = 0  # remove the singular DC component

    # Rectangular high-wavenumber cutoff.
    kx_max = np.inf if lambda_min_x is None else 2*np.pi/lambda_min_x
    kz_max = np.inf if lambda_min_z is None else 2*np.pi/lambda_min_z
    mask = (np.abs(KX) <= kx_max) & (np.abs(KZ) <= kz_max)
    spectral_filter[~mask] = 0

    # Filtering real white noise gives a Hermitian spectrum.
    white_noise = rng.randn(Nz, Nx)
    h = np.real(np.fft.ifft2(np.fft.fft2(white_noise)*spectral_filter))
    h = h - h.mean()

    rms_before = h.std(ddof=1)  # MATLAB std(h(:),0)
    if rms_before < 1e-15:
        print('warning: generated field is nearly zero; check the wavelength '
              'cutoffs', file=sys.stderr)
        h[:] = 0
    else:
        h = alpha*h/rms_before

    info = dict(Hx=Hx, Hz=Hz, lx=lx, lz=lz, aniso_ratio=aniso_ratio, kx=kx,
                kz=kz, rho=rho, mask=mask, spectral_filter=spectral_filter,
                kx_max=kx_max, kz_max=kz_max, dx=dx, dz=dz)
    return h, x, z, info


def read_h(filename):
    """Read h (Nz x Nx) from .npy or a delimited text file."""
    if filename.endswith('.npy'):
        h = np.load(filename)
    else:
        with open(filename) as f:
            first = f.readline()
        delim = ',' if ',' in first else None
        h = np.loadtxt(filename, delimiter=delim, ndmin=2)
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or min(h.shape) < 2:
        raise ValueError('h must be a 2D array with at least 2 rows and '
                         'columns; got shape %s' % (h.shape,))
    if not np.all(np.isfinite(h)):
        raise ValueError('h contains nonfinite values')
    return h


def embed(h, x, z, args):
    """Return the grid points, shape (nj, ni, 3), rows written in order j."""
    Nz, Nx = h.shape
    X, Z = np.meshgrid(x, z)  # Nz x Nx
    if args.frame == 'yxz':
        P = np.stack([X, args.y0 + h, -(args.ztop + Z)], axis=-1)
        return P  # j = 0 is the top edge; j increases down dip
    phi = math.radians(args.strike)
    delta = math.radians(args.dip)
    s = np.array([math.sin(phi), math.cos(phi), 0])
    d = np.array([math.cos(delta)*math.cos(phi),
                  -math.cos(delta)*math.sin(phi), -math.sin(delta)])
    n = np.array([math.sin(delta)*math.cos(phi),
                  -math.sin(delta)*math.sin(phi), math.cos(delta)])
    O = np.array([args.x0, args.y0, -args.ztop])
    P = (O + X[..., None]*s + Z[..., None]*d + h[..., None]*n)
    return P[::-1]  # j = 0 is the bottom edge; j increases up dip


def write_geom(filename, P, header_lines):
    nj, ni, _ = P.shape
    with open(filename, 'w') as f:
        f.write('# marmot fault geometry: ni nj, then ni*nj points x y z, '
                'i fastest\n')
        for line in header_lines:
            f.write('# %s\n' % line)
        f.write('%d %d\n' % (ni, nj))
        np.savetxt(f, P.reshape(-1, 3), fmt='%23.16e')


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        epilog=__doc__.split('\n', 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('roughness (as rough3D_fft_minlambda_twoH.m)')
    g.add_argument('--Hx', type=float, default=0.8, help='Hurst exponent '
                   'along x [0.8]')
    g.add_argument('--Hz', type=float, default=0.8, help='Hurst exponent '
                   'along z [0.8]')
    g.add_argument('--Nx', type=int, default=129, help='samples along x [129]')
    g.add_argument('--Nz', type=int, default=65, help='samples along z [65]')
    g.add_argument('--Lx', type=float, default=2.0, help='length along x [2]')
    g.add_argument('--Lz', type=float, default=1.0, help='length down dip [1]')
    g.add_argument('--alpha', type=float, default=0.01,
                   help='RMS roughness [0.01]')
    g.add_argument('--lambda-min-x', type=float, default=None,
                   help='shortest wavelength along x [none]')
    g.add_argument('--lambda-min-z', type=float, default=None,
                   help='shortest wavelength along z [none]')
    g.add_argument('--aniso', type=float, default=1.0,
                   help='z-wavenumber scale; > 1 makes z smoother [1]')
    g.add_argument('--seed', type=int, default=None,
                   help='random seed [none: random]')
    g.add_argument('--h-file', default=None,
                   help='read h (Nz rows x Nx columns, row k at z = k*Lz/'
                        '(Nz-1)) from .npy or text/csv instead of '
                        'generating it; Nx, Nz are taken from the file')
    e = p.add_argument_group('embedding in 3D')
    e.add_argument('--frame', choices=['yxz', 'dip'], default='yxz',
                   help='yxz: vertical y = y0 + h; dip: plane with strike '
                        'and dip [yxz]')
    e.add_argument('--strike', type=float, default=90.0,
                   help='dip frame: strike azimuth, degrees clockwise from '
                        'north (+y) [90]')
    e.add_argument('--dip', type=float, default=90.0,
                   help='dip frame: dip angle in degrees, (0, 90] [90]')
    e.add_argument('--ztop', type=float, default=0.0,
                   help='depth of the top edge (>= 0) [0]')
    e.add_argument('--x0', type=float, default=0.0,
                   help='dip frame: x of the top-edge start point [0]')
    e.add_argument('--y0', type=float, default=0.0,
                   help='y of the reference plane / top-edge start point [0]')
    p.add_argument('-o', '--out', default='geom.txt',
                   help='output geometry file [geom.txt]')
    p.add_argument('--save-h', default=None,
                   help='also save h (Nz x Nx) to this .npy or text file')
    args = p.parse_args(argv)

    if args.frame == 'dip' and not (0 < args.dip <= 90):
        p.error('--dip must be in (0, 90]')
    if args.ztop < 0:
        p.error('--ztop must be >= 0')

    if args.h_file:
        h = read_h(args.h_file)
        Nz, Nx = h.shape
        x = np.linspace(0, args.Lx, Nx)
        z = np.linspace(0, args.Lz, Nz)
        source = 'h from %s' % args.h_file
    else:
        h, x, z, _ = rough3d_fft_minlambda_twoh(
            args.Hx, args.Hz, args.Nx, args.Nz, args.Lx, args.Lz, args.alpha,
            args.lambda_min_x, args.lambda_min_z, args.aniso, args.seed)
        Nz, Nx = h.shape
        source = ('rough3D_fft_minlambda_twoH Hx=%g Hz=%g Nx=%d Nz=%d Lx=%g '
                  'Lz=%g alpha=%g lambda_min_x=%s lambda_min_z=%s aniso=%g '
                  'seed=%s' % (args.Hx, args.Hz, Nx, Nz, args.Lx, args.Lz,
                               args.alpha, args.lambda_min_x,
                               args.lambda_min_z, args.aniso, args.seed))

    P = embed(h, x, z, args)
    if args.frame == 'yxz':
        frame = ('frame yxz: (x, y0 + h, -(ztop + z)), y0=%g ztop=%g; i along '
                 '+x, j down dip; normal +y' % (args.y0, args.ztop))
    else:
        frame = ('frame dip: strike=%g dip=%g x0=%g y0=%g ztop=%g (ENU); i '
                 'along strike, j up dip from the bottom edge; normal into '
                 'the hanging wall' % (args.strike, args.dip, args.x0,
                                       args.y0, args.ztop))
    write_geom(args.out, P, [source, frame])
    if args.save_h:
        if args.save_h.endswith('.npy'):
            np.save(args.save_h, h)
        else:
            np.savetxt(args.save_h, h, fmt='%.17g', delimiter=',')

    dx, dz = x[1] - x[0], z[1] - z[0]
    hz, hx = np.gradient(h, dz, dx)
    print('wrote %s: %d x %d points (ni x nj), %s' % (args.out, Nx, Nz,
                                                     args.frame))
    print('h: rms %.4g  max|h| %.4g  rms slope dh/dx %.4g dh/dz %.4g  '
          'max slope %.4g' % (h.std(ddof=1), np.abs(h).max(),
                              np.sqrt(np.mean(hx**2)),
                              np.sqrt(np.mean(hz**2)),
                              np.sqrt(hx**2 + hz**2).max()))
    zmax = P[..., 2].max()
    if zmax > 0:
        print('warning: max z = %.4g > 0; marmot halfspace=1 requires '
              'z <= 0 (increase --ztop)' % zmax, file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
