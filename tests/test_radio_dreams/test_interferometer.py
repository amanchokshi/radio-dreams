"""Tests radio_dreams.interferometer."""

from os import path

from radio_dreams.interferometer import Layout, Synthesis

# Save the path to this directory
dirpath = path.dirname(__file__)

# Obtain path to directory with test_data
test_data = path.abspath(path.join(dirpath, "../test_data"))


def test_Layout():
    """Instance of Layout with test_mwa.csv array to test outputs."""
    mwa = Layout(array_csv=f"{test_data}/test_mwa.csv", latitude=-26.7033194444)

    # Test __init__
    assert mwa.east[0] == -1999.81
    assert mwa.north[0] == 206.85
    assert mwa.height[0] == 372.921
    assert mwa.tiles[0] == "LBA1"

    # Test enh_xyz
    assert mwa.x[0] == 426.09958407519065
    assert mwa.y[0] == -1999.81
    assert mwa.z[0] == -260.53213231153575


def test_Layout_no_latitude(capfd):
    """It outputs latitude not provided error."""
    Layout(array_csv=f"{test_data}/test_mwa.csv")
    out, err = capfd.readouterr()

    assert "No latitude provided" in out


def test_Synthesis_freqs():
    """It outputs freq array and compares results."""
    mwa = Synthesis(
        array_csv=f"{test_data}/test_mwa.csv",
        freq_start=160e6,
        freq_bands=24,
        bandwidth=1 * +6,
    )

    assert mwa.freqs()[0] == 160.0e6


def test_Synthesis_freqs_no_args(capfd):
    """It prints exception for missing args."""
    Synthesis(array_csv=f"{test_data}/test_mwa.csv").freqs()

    out, err = capfd.readouterr()
    assert "missing required arguments" in out
