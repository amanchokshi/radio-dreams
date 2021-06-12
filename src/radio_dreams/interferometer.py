"""Classes and methods describing the functionality of an interferometer."""

from pathlib import Path

import numpy as np
from scipy.constants import c


class Telescope:
    """A class used to represent the positions of antennas in an interferometer.

    Antenna positions are usually defined with respect to the array centre.

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

    .. code-block:: python

        from radio_dreams import interferometer

        mwa_latitude = -26.7033194444

        # An instance of the interferometer.ArrayConfig class for the MWA
        mwa = interferometer.ArrayConfig(array_csv="../arrays/mwa_phase2.csv")

        # Access the original E, N, H positions of tiles and tile names
        E = mwa.east
        N = mwa.north
        H = mwa.height
        T = mwa.tiles

        # Access X, Y, Z coordinates of tiles
        x, y, z = mwa.enh_xyz(latitude=mwa_latitude)


    """

    def __init__(self, telescope_name):

        self.name = telescope_name

    def _read_layout(self, layout):

        self.layout = np.loadtxt(layout).T

    def _enh_xyz(self):
        """Convert from local E, N, H to X, Y, Z coordinates."""

        east, north, height = self.layout[0], self.layout[1], self.layout[2]

        sin_lat = np.sin(self.location.latitude.radians)
        cos_lat = np.cos(self.location.latitude.radians)

        x = height * cos_lat - north * sin_lat
        y = east
        z = height * sin_lat + north * cos_lat

        self.xyz = np.array([x, y, z])

    def configure(self, location=None, layout=None, freqs=None):
        self.location = location

        if freqs is not None:
            self.freqs = freqs
        try:
            self.freqs
        except Exception as e:
            print(f"FreqMissingError : {e}")

        try:
            if not Path(layout).is_file():
                raise FileNotFoundError

            try:
                self._read_layout(layout)
            except Exception as e:
                print(e)

        except FileNotFoundError:
            print(f"FileNotFoundError : Layout file '{layout}' doesn't exist")

        try:
            self._enh_xyz()
        except Exception as e:
            print(f"TelescopeConfigError : {e}")

        try:
            self._lengths_xyz()
        except Exception as e:
            print(f"TelescopeConfigError : {e}")

    def _lengths_xyz(self):

        # All possible baseline distances, in metres
        # This is equivalent to two nested for loops
        lx = np.concatenate(self.xyz[0] - self.xyz[0][:, None])
        ly = np.concatenate(self.xyz[1] - self.xyz[1][:, None])
        lz = np.concatenate(self.xyz[2] - self.xyz[2][:, None])

        wavelengths = c / self.freqs

        lx_lambda = lx / wavelengths[:, None]
        ly_lambda = ly / wavelengths[:, None]
        lz_lambda = lz / wavelengths[:, None]

        self.xyz_lambda = np.swapaxes(np.array([lx_lambda, ly_lambda, lz_lambda]), 0, 1)

    def _uvw(self, ha0, dec0):

        self.ha0 = ha0
        self.dec0 = dec0

        xyz_uvw_mat = np.array(
            [
                [np.sin(self.ha0), np.cos(self.ha0), 0],
                [
                    -np.sin(self.dec0) * np.cos(self.ha0),
                    np.sin(self.dec0) * np.sin(self.ha0),
                    np.cos(self.dec0),
                ],
                [
                    np.cos(self.dec0) * np.cos(self.ha0),
                    -np.cos(self.dec0) * np.sin(self.ha0),
                    np.sin(self.dec0),
                ],
            ]
        )
        self.uvw = np.matmul(xyz_uvw_mat, self.xyz_lambda)
