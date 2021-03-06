"""Nox sessions."""

import tempfile

import nox


package = "radio_dreams"
nox.options.sessions = "lint", "tests"
locations = "src", "tests", "noxfile.py", "docs/conf.py"


def install_with_constraints(session, *args, **kwargs):
    """Install packages constrained by Poetry's lock file.

    This function is a wrapper for nox.sessions.Session.install. It
    invokes pip to install packages inside of the session's virtualenv.
    Additionally, pip is passed a constraints file generated from
    Poetry's lock file, to ensure that the packages are pinned to the
    versions specified in poetry.lock. This allows you to manage the
    packages as Poetry development dependencies.

    Arguments:
        session: The Session object.
        args: Command-line arguments for pip.
        kwargs: Additional keyword arguments for Session.install.
    """
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True,
        )
        session.install(f"--constraint={requirements.name}", *args, **kwargs)


# Black - the uncompromising Python code formatter
@nox.session(python="3.8")
def black(session):
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


# Flake8 - linting of various kinds
@nox.session(python=["3.9", "3.8", "3.7"])
def lint(session):
    """Lint using flake8."""
    args = session.posargs or locations
    install_with_constraints(
        session,
        "flake8",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
    )
    session.run("flake8", *args)


# Pytest - unit tests
@nox.session(python=["3.9", "3.8", "3.7"])
def tests(session):
    """Run the test suite."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session, "coverage[toml]", "pytest", "pytest-cov", "pytest-mock"
    )
    session.run("pytest", "--cov")


@nox.session(python="3.8")
def coverage(session):
    """Upload coverage data."""
    install_with_constraints(session, "coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")
    session.run("codecov", *session.posargs)


# Sphinx - build documentation
@nox.session(python="3.8")
def docs(session):
    """Build the documentation."""
    install_with_constraints(session, "sphinx")
    session.run("sphinx-build", "docs", "docs/_build")
