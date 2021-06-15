"""Test radio_dreams.interferometer ."""

import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

import numpy as np
from radio_dreams.interferometer import (
    enh_xyz,
    gauss_kernel,
    radec_lmn,
    read_layout,
    uv_degrid,
    xyz_uvw,
)
from skyfield.api import wgs84

mwa_geo = wgs84.latlon(-26.703319, 116.670815, 337.83)
freq = 200e6

# Save the path to this directory
dirpath = os.path.dirname(__file__)

# Obtain path to directory with test_data
test_data = os.path.abspath(os.path.join(dirpath, "./test_data"))


def test_read_layout():
    """Check read values match data."""

    layout = read_layout(layout_path=f"{test_data}/test_mwa.txt")

    assert layout.shape[0] == 3
    assert layout[0][0] == -1.497849999999999966e02
    assert layout[1][0] == 2.658140000000000214e02
    assert layout[2][0] == 3.770110000000000241e02


def test_enh_xyz():
    """Check output values and shape."""

    layout = read_layout(layout_path=f"{test_data}/test_mwa.txt")
    xyz = enh_xyz(layout=layout, latitude=mwa_geo.latitude.radians)

    assert xyz.shape[0] == 3
    assert xyz.shape[1] == 3
    assert xyz[0, 0] == 456.25006328090495
    assert xyz[1, 0] == -149.785
    assert xyz[2, 0] == 68.04598792853452


def test_xyz_uvw():
    """Check output values and shape."""

    layout = read_layout(layout_path=f"{test_data}/test_mwa.txt")
    xyz = enh_xyz(layout=layout, latitude=mwa_geo.latitude.radians)
    uvw = xyz_uvw(xyz=xyz, freq=freq, dec0=mwa_geo.latitude.radians, ha0=0)

    assert uvw.shape == (3, 9)

    assert uvw[0][0] == 0.0
    assert uvw[1][0] == 0.0
    assert uvw[2][0] == 0.0


def test_gauss_kernel():
    """Check output values and shape."""

    gauss = gauss_kernel(2, 5)

    assert gauss.shape == (5, 5)
    assert gauss[2, 2] == 0.039788735772973836


def test_uv_degrid():
    """Check output values and shape."""

    layout = read_layout(layout_path=f"{test_data}/test_mwa.txt")
    xyz = enh_xyz(layout=layout, latitude=mwa_geo.latitude.radians)
    uvw = xyz_uvw(xyz=xyz, freq=freq, dec0=mwa_geo.latitude.radians, ha0=0)
    uv = uv_degrid(max_lambda=1400, nside=20, uvw=uvw, sigma=3, kersize=21, kernel=None)

    assert uv.shape == (20, 20)
    assert uv[0, 0] == 0.0


def test_uv_degrid_gaussian_kernel():
    """Check output values and shape."""

    layout = read_layout(layout_path=f"{test_data}/test_mwa.txt")
    xyz = enh_xyz(layout=layout, latitude=mwa_geo.latitude.radians)
    uvw = xyz_uvw(xyz=xyz, freq=freq, dec0=mwa_geo.latitude.radians, ha0=0)
    uv = uv_degrid(
        max_lambda=1400, nside=20, uvw=uvw, sigma=3, kersize=21, kernel="gaussian"
    )

    assert uv.shape == (20, 20)
    assert uv[0, 0] == 1.295932713086053e-05


def test_radec_lmn():
    """Check output values and shape."""

    lmn = radec_lmn(ra=np.arange(4), ra0=3, dec=np.arange(-2, 2), dec0=0)

    out = np.array(
        [
            [0.05872664, -0.4912955, -0.84147098, 0.0],
            [-0.90929743, -0.84147098, 0.0, 0.84147098],
            [-0.41198225, 0.2248451, -0.54030231, -0.54030231],
        ]
    )

    assert lmn.shape == (3, 4)
    assert np.all(np.isclose(out, lmn))
