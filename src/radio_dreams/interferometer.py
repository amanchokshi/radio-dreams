# -*- coding: utf-8 -*-

"""Imaginary Interferometers."""

__author__ = "Aman Chokshi"
__date__ = "2021-06-13"
__cite__ = "https://github.com/amanchokshi"

import numpy as np
from numba import njit, prange
from scipy.constants import c


def read_layout(layout_path=None):
    """Read array layout from txt file.

    Antenna positions are defined with respect to the array centre.

    E - East of the center in metres
    N - North of center in metres
    H - Height above sea level in metres

    :param str layout_path: Path to file with array layout in E, N, H columns

    :returns: Array of shape [3, n], for E, N, H respectively
    :rtype: :class:`~numpy.ndarray`
    """
    return np.loadtxt(layout_path).T


@njit(parallel=True, nogil=True)
def enh_xyz(layout=None, latitude=None):
    r"""Convert from local E, N, H to X, Y, Z coordinates.

    Antenna positions are defined with respect to the array centre.

    E - East of the center in metres
    N - North of center in metres
    H - Height above sea level in metres

    Convert these coordinates to a Earth Centered Earth Fixed (ECEF) cartesian
    system with axes pointing towards

    X - (h = 0, δ = 0)
    Y - (h = -6, δ = 0)
    Z - (δ = 90)

    h - hour angle
    δ - declination

    :param layout: object from :func:`read_layout`
    :type layout: :class:`~numpy.ndarray`
    :param float latitude: Latitude of array in radians

    :returns: Array of shape [3, n], for X, Y, Z respectively
    :rtype: :class:`~numpy.ndarray`
    """
    east, north, height = layout[0], layout[1], layout[2]

    sin_lat = np.sin(latitude)
    cos_lat = np.cos(latitude)

    x = height * cos_lat - north * sin_lat
    y = east
    z = height * sin_lat + north * cos_lat

    xyz = np.vstack((x, y, z))

    return xyz


@njit(parallel=True, nogil=True)
def xyz_uvw(xyz=None, freq=None, dec0=None, ha0=None):
    r"""Convert local XYZ to UVU coordinates.

    U, V, W are coordinates used to represent interferometric baselines

    .. math::

        \begin{bmatrix}
            u \\
            v \\
            w
        \end{bmatrix} =
        \begin{bmatrix}
            \sin(H_0) & \cos(H_0) & 0 \\
            -\sin(\delta_0)\cos(H_0) & \sin(\delta_0)\sin(H_0) & \cos(\delta_0) \\
            \cos(\delta_0)\cos(H_0) & -\cos(\delta_0)\sin(H_0) & \sin(\delta_0)
        \end{bmatrix}
        \begin{bmatrix}
            X_\lambda \\
            Y_\lambda \\
            Z_\lambda
        \end{bmatrix},

    :param xyz:  object from :func:`enh_xyz`
    :type xyz: :class:`~numpy.ndarray`
    :param float freq: 1D of frequencies in Hz
    :param float dec0: Declination of phase centre in radians
    :param float ha0: Hour Angle of phase centre in radians

    :returns: UVW cube, with 0 axis for frequency and 1, 2 for UVWs
    :rtype: :class:`~numpy.ndarray`
    """
    # All possible baseline distances, in metres
    # This is equivalent to two nested for loops

    # This is the numpy version. Have moved to numba
    # lx = np.concatenate(xyz[0] - xyz[0][:, None])
    # ly = np.concatenate(xyz[1] - xyz[1][:, None])
    # lz = np.concatenate(xyz[2] - xyz[2][:, None])

    # wavelengths = c / freqs

    # lx_lambda = np.array(lx) / np.atleast_2d(wavelengths).T
    # ly_lambda = np.array(ly) / np.atleast_2d(wavelengths).T
    # lz_lambda = np.array(lz) / np.atleast_2d(wavelengths).T

    # xyz_lambda = np.swapaxes(np.array([lx_lambda, ly_lambda, lz_lambda]), 0, 1)

    # xyz_uvw_mat = np.array(
    #     [
    #         [np.sin(ha0), np.cos(ha0), 0],
    #         [-np.sin(dec0) * np.cos(ha0), np.sin(dec0) * np.sin(ha0), np.cos(dec0)],
    #         [np.cos(dec0) * np.cos(ha0), -np.cos(dec0) * np.sin(ha0), np.sin(dec0)],
    #     ]
    # )

    # uvw = np.matmul(xyz_uvw_mat, xyz_lambda)

    lx = []
    for i in xyz[0]:
        for j in xyz[0]:
            lx.append(i - j)
    ly = []
    for i in xyz[1]:
        for j in xyz[1]:
            ly.append(i - j)
    lz = []
    for i in xyz[2]:
        for j in xyz[2]:
            lz.append(i - j)

    wavelength = c / freq

    lx_lambda = np.array(lx) / wavelength
    ly_lambda = np.array(ly) / wavelength
    lz_lambda = np.array(lz) / wavelength

    xyz_lambda = np.vstack((lx_lambda, ly_lambda, lz_lambda))

    xyz_uvw_mat = np.array(
        [
            [np.sin(ha0), np.cos(ha0), 0],
            [-np.sin(dec0) * np.cos(ha0), np.sin(dec0) * np.sin(ha0), np.cos(dec0)],
            [np.cos(dec0) * np.cos(ha0), -np.cos(dec0) * np.sin(ha0), np.sin(dec0)],
        ]
    )

    uvw = np.dot(xyz_uvw_mat, xyz_lambda)

    return uvw


@njit()
def gauss_kernel(sigma, kersize):
    """Create 2D gaussian kernel.

    :param float sigma: Standard deviation of gaussian kernel
    :param float kersize: Kernel size in pixels.
        *Must be odd number for symmetry*

    :returns: 2D gaussian kernal
    :rtype: :class:`~numpy.ndarray`
    """
    x, y = np.arange(kersize), np.arange(kersize)

    cen = int(kersize / 2)
    norm = 1 / (2 * np.pi * sigma ** 2)

    # Looks instead of meshgrid as
    # Numba doesn't support it
    gauss = np.zeros((kersize, kersize))
    for i in x:
        for j in y:
            gauss[i, j] = norm * np.exp(
                (-1 / (2 * sigma ** 2)) * ((i - cen) ** 2 + (j - cen) ** 2)
            )

    return gauss


@njit(parallel=True, nogil=True)
def uv_degrid(
    max_lambda=1400, nside=511, uvw=None, sigma=3, kersize=21, kernel="gaussian"
):
    """Degrid continuous uv baselines onto regular uv grid.

    :param int max_lambda: Maximum baseline to evaluate,
        defaults to [1400]
    :param int nside: Number of pixels per side,
        defaults to [511]
    :param uvw: UVW array object from :func:`xyz_uvw`
    :type uvw: :class:`~numpy.ndarray`
    :param float sigma: Standard deviation of gaussian kernel,
        defaults to [3]
    :param int kernel: Kernel size in pixel,
        *Must be odd number for symmetry*,
        defaults to [21]
    :param str kernel: Kernel type, `gaussian` or `None`,
        defaults to ["gaussian"]

    :returns: UV grid of size [nside, nside]
    :rtype: :class:`~numpy.ndarray`
    """
    # Define a grid of u, v coords
    u_range = np.linspace(-1 * max_lambda, max_lambda, nside)
    v_range = np.linspace(-1 * max_lambda, max_lambda, nside)

    # Create empty uv grid onto which uv samples will be degridded
    uv_grid = np.zeros((v_range.shape[0], u_range.shape[0]))

    # Continuous uv coords of array to be degridded onto uv_grid
    u, v = uvw[0], uvw[1]

    # This numpy magic doesn't work with numba - fallback to for loops
    # u_closest_ind = np.argmin(np.abs(u_range - u[:, None]), axis=1)
    # v_closest_ind = np.argmin(np.abs(v_range - v[:, None]), axis=1)

    for i in prange(u.shape[0]):

        # Indicies of the closest pixel
        u_ind = np.argmin(np.abs(u_range - u[i]))
        v_ind = np.argmin(np.abs(v_range - v[i]))

        # Use gaussian kernal by default
        if kernel == "gaussian":

            # Generate the gaussian kernel
            gauss = gauss_kernel(sigma, kersize)
            ker_2 = int(kersize / 2)

            # Indices of uv_grid where kernel overlaps, centered at (u_ind, v_ind)
            v_grid_inds = np.arange(v_ind - ker_2, v_ind + ker_2 + 1)
            u_grid_inds = np.arange(u_ind - ker_2, u_ind + ker_2 + 1)

            # Append kernel values to uv_grid
            for vi in prange(v_grid_inds.shape[0]):
                for ui in prange(u_grid_inds.shape[0]):
                    if (v_grid_inds[vi] >= 0) & (v_grid_inds[vi] < nside):
                        if (u_grid_inds[ui] >= 0) & (u_grid_inds[ui] < nside):
                            uv_grid[v_grid_inds[vi], u_grid_inds[ui]] += gauss[vi, ui]

        # If not gaussian snap to closest gridpoint
        # All flux moved to nearest pixel
        else:
            uv_grid[int(v_ind), int(u_ind)] += 1

    return uv_grid


@njit(parallel=True, nogil=True)
def radec_lmn(ra=None, ra0=None, dec=None, dec0=None):
    r"""Calculate LMN direction cosines from Ra/Dec with respect to a phase center.

    .. math::
        :nowrap:

        \begin{eqnarray}
            & l =& \, \cos \, \delta  \sin \, \Delta \alpha  \\
            & m =& \, \sin \, \delta \cos \, \delta 0 -
                         \cos \delta \sin \delta 0 \cos \Delta \alpha \\
            & n =& \, \sin \, \delta \sin \, \delta 0 +
                         \cos \delta \cos \delta 0 \cos \Delta \alpha \\
            &   =& \, \sqrt{1 - l^2 - m^2} - 1
        \end{eqnarray}

    Here :math:`\Delta \alpha = \alpha - \alpha 0` is the difference between
    the Right Ascension of each coordinate and the phase centre and
    :math:`\delta 0` is the Declination of the phase centre.

    :param ra: Right ascension in `radians`
    :type ra: :class:`~numpy.ndarray`
    :param dec: Declination in `radians`
    :type dec: :class:`~numpy.ndarray`
    :param float ra0: Right ascension of phase centre in `radians`
    :param float dec0: Declination of phase centre in `radians`

    :returns: lmn direction cosines of shape (3, n)
    :rtype: :class:`~numpy.ndarray`
    """

    del_ra = ra - ra0

    l = np.cos(dec) * np.sin(del_ra)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(del_ra)
    n = np.sin(dec) * np.sin(dec0) - np.cos(dec) * np.cos(dec0) * np.cos(del_ra)

    lmn = np.vstack((l, m, n))

    return lmn
