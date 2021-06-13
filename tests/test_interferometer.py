"""Test radio_dreams.interferometer ."""

from os import path

import numpy as np
from radio_dreams.interferometer import enh_xyz, read_layout, xyz_uvw
from skyfield.api import wgs84

mwa_geo = wgs84.latlon(-26.703319, 116.670815, 337.83)
freqs = np.linspace(200e6, 300e6, 2)


# Save the path to this directory
dirpath = path.dirname(__file__)

# Obtain path to directory with test_data
test_data = path.abspath(path.join(dirpath, "./test_data"))


def test_read_layout():
    """Check read values match data."""

    layout = read_layout(f"{test_data}/test_mwa.txt")

    assert layout.shape[0] == 3
    assert layout[0][0] == -1.497849999999999966e02
    assert layout[1][0] == 2.658140000000000214e02
    assert layout[2][0] == 3.770110000000000241e02


def test_enh_xyz():
    """Check output values and shape."""

    layout = read_layout(f"{test_data}/test_mwa.txt")
    xyz = enh_xyz(layout, mwa_geo.latitude.radians)

    assert xyz.shape[0] == 3
    assert xyz.shape[1] == 3
    assert xyz[0, 0] == 456.25006328090495
    assert xyz[1, 0] == -149.785
    assert xyz[2, 0] == 68.04598792853452

def test_xyz_uvw():
    """Check output values and shape."""

    layout = read_layout(f"{test_data}/test_mwa.txt")
    xyz = enh_xyz(layout, mwa_geo.latitude.radians)
    uvw = xyz_uvw(xyz, freqs, 0, mwa_geo.latitude.radians)

    assert uvw.shape == (2, 3, 9)

    assert uvw[0][0][0] == 0.0
    assert uvw[0][1][0] == 0.0
    assert uvw[0][2][0] == 0.0
