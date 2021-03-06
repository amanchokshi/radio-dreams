"""Tests radio_dreams.interferometer."""

from os import path

from radio_dreams.interferometer import Freqs, Layout  # Synthesis

# Save the path to this directory
dirpath = path.dirname(__file__)

# Obtain path to directory with test_data
test_data = path.abspath(path.join(dirpath, "../test_data"))


def test_Layout():
    """Instance of Layout with test_mwa.csv array to test outputs."""
    mwa = Layout(
        array_csv=f"{test_data}/test_mwa.csv",
        latitude=-26.7033194444,
        longitude=116.670815,
        elevation=337.83,
    )

    # Test __init__
    assert mwa.east[0] == -1999.81
    assert mwa.north[0] == 206.85
    assert mwa.height[0] == 372.921
    assert mwa.tiles[0] == "LBA1"

    # Test enh_xyz
    assert mwa.x[0] == 426.09958407519065
    assert mwa.y[0] == -1999.81
    assert mwa.z[0] == -260.53213231153575

    # Test gps
    assert mwa.gps.latitude.degrees == -26.7033194444
    assert mwa.gps.longitude.degrees == 116.670815
    assert mwa.gps.elevation.m == 337.83


def test_Layout_no_gps(capfd):
    """It outputs latitude not provided error."""
    Layout(array_csv=f"{test_data}/test_mwa.csv")
    out, err = capfd.readouterr()

    assert "No latitude provided" in out
    assert "No longitude provided" in out
    assert "No elevation provided" in out


def test_Freqs():
    """It outputs freq array and compares results."""
    mwa_freqs = Freqs(
        freq_start=160e6,
        freq_bands=24,
        bandwidth=1e6,
    )

    assert mwa_freqs.freqs[0] == 160.0e6


def test_Freqs_no_args(capfd):
    """It prints exception for missing args."""
    Freqs()

    out, err = capfd.readouterr()
    assert "missing required arguments" in out


def test_Freqs_to_lamba():
    """It outputs wavelength array and compares results."""
    mwa_freqs = Freqs(
        freq_start=160e6,
        freq_bands=24,
        bandwidth=1e6,
    )

    lambdas = mwa_freqs.to_lambda()

    assert lambdas[0] == 1.8737028625
