"""Classes and methods describing the functionality of an interferometer."""

import numpy as np
import pandas as pd
from skyfield.api import wgs84

#  from scipy import constants as const


class Layout:
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

    def __init__(self, array_csv, latitude=None, longitude=None, elevation=None):
        """Assign variables and read array layout file."""
        self.array_csv = array_csv

        # Read array layout csv file
        df = pd.read_csv(self.array_csv)
        self.east = df["East"].to_numpy()
        self.north = df["North"].to_numpy()
        self.height = df["Height"].to_numpy()
        self.tiles = df["Tile"]

        # MWA latitude
        if latitude is None:
            print(
                " ** INFO: No latitude provided - defaults to MWA: -26.7033194444 deg"
            )
            self.latitude = -26.7033194444
        else:
            self.latitude = latitude

        # MWA longitude
        if longitude is None:
            print(" ** INFO: No longitude provided - defaults to MWA: 116.670815 deg")
            self.longitude = 116.670815
        else:
            self.longitude = longitude

        # MWA elevation
        if elevation is None:
            print(" ** INFO: No elevation provided - defaults to MWA: 337.83m")
            self.elevation = 337.83
        else:
            self.elevation = elevation

        # Array gps coords
        self.gps = wgs84.latlon(self.latitude, self.longitude, self.elevation)

        # Convert from local E, N, H to X, Y, Z coordinates
        # Latitude in radians
        sin_lat = np.sin(self.gps.latitude.radians)
        cos_lat = np.cos(self.gps.latitude.radians)

        self.x = self.height * cos_lat - self.north * sin_lat
        self.y = self.east
        self.z = self.height * sin_lat + self.north * sin_lat


class Synthesis(Layout):
    """Synthesis class."""

    def __init__(
        self,
        array_csv,
        latitude=None,
        longitude=None,
        elevation=None,
        freq_start=None,
        freq_bands=None,
        bandwidth=None,
    ):
        """Assign Synth class variables."""
        super().__init__(array_csv, latitude, longitude, elevation)

        self.freq_start = freq_start
        self.freq_bands = freq_bands
        self.bandwidth = bandwidth

    def freqs(self):
        """Create frequency array for interferometer."""
        if None not in [self.freq_start, self.freq_bands, self.bandwidth]:
            freqs = np.arange(
                self.freq_start,
                self.freq_start + self.freq_bands * self.bandwidth,
                self.bandwidth,
            )

            return freqs

        else:
            print(
                " ** INFO: freqs() missing required arguments:"
                " 'freq_start', 'freq_bands', 'bandwidth'"
            )
