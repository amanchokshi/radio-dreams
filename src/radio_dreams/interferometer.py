# -*- coding: utf-8 -*-

"""Imaginary Interferometers."""

__author__ = "Aman Chokshi"
__date__ = "2021-06-13"
__cite__ = "https://github.com/amanchokshi"

import numpy as np
from numba import njit
from scipy.constants import c


def read_layout(layout_txt):
    """Read array layout from txt file.

    Antenna positions are defined with respect to the array centre.

    E - East of the center in metres
    N - North of center in metres
    H - Height above sea level in metres

    :param str layout_txt: Path to file with array layout in E, N, H columns

    :returns: Array of shape [3, n], for E, N, H respectively
    :rtype: :class:`numpy.ndarray`
    """
    return np.loadtxt(layout_txt).T


@njit()
def enh_xyz(layout, latitude):
    """Convert from local E, N, H to X, Y, Z coordinates.

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

    :param layout: :class:`~numpy.ndarray` object from :func:`read_layout`
    :param latitude: Latitude of array in radians

    :returns: Array of shape [3, n], for X, Y, Z respectively
    :rtype: :class:`numpy.ndarray`
    """
    east, north, height = layout[0], layout[1], layout[2]

    sin_lat = np.sin(latitude)
    cos_lat = np.cos(latitude)

    x = height * cos_lat - north * sin_lat
    y = east
    z = height * sin_lat + north * cos_lat

    xyz = np.vstack((x, y, z))

    return xyz


@njit()
def xyz_uvw(xyz, freqs, dec0, ha0):
    """Convert local XYZ to UVU coordinates.

    U, V, W are coordinates used to represent interferometric baselines

    :param xyz: :class:`~numpy.ndarray` object from :func:`enh_xyz`
    :param freqs: :class:`~numpy.ndarray` 1D of frequencies in Hz
    :param dec0: Declination of phase centre in radians `float`
    :param ha0: Hour Angle of phase centre in radians `float`

    :returns: UVW cube, with 0 axis for frequency and 1, 2 for UVWs
    :rtype: :class:`numpy.ndarray`
    """
    # All possible baseline distances, in metres
    # This is equivalent to two nested for loops
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

    wavelengths = c / freqs

    uvw = np.zeros((freqs.shape[0], 3, len(lx)))
    #  uvw = []
    for i in range(wavelengths.shape[0]):

        lx_lambda = np.array(lx) / wavelengths[i]
        ly_lambda = np.array(ly) / wavelengths[i]
        lz_lambda = np.array(lz) / wavelengths[i]

        xyz_lambda = np.vstack((lx_lambda, ly_lambda, lz_lambda))

        xyz_uvw_mat = np.array(
            [
                [np.sin(ha0), np.cos(ha0), 0],
                [-np.sin(dec0) * np.cos(ha0), np.sin(dec0) * np.sin(ha0), np.cos(dec0)],
                [np.cos(dec0) * np.cos(ha0), -np.cos(dec0) * np.sin(ha0), np.sin(dec0)],
            ]
        )

        #  uvw.append(np.dot(xyz_uvw_mat, xyz_lambda))
        uvw[i] = np.dot(xyz_uvw_mat, xyz_lambda)

    return uvw
