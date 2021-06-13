import tempfile

import nox

nox.options.sessions = "lint", "tests"
locations = "src", "tests", "noxfile.py"


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


@nox.session(python=["3.10", "3.9", "3.8", "3.7"])
def tests(session):
    args = session.posargs or ["--cov"]
    install_with_constraints(session, "coverage[toml]", "pytest", "pytest-cov")
    session.run("pytest", *args)


@nox.session(python=["3.10", "3.9", "3.8", "3.7"])
def lint(session):
    args = session.posargs or locations
    install_with_constraints(session, "flake8", "flake8-black")
    session.run("flake8", *args)


@nox.session(python="3.8")
def black(session):
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)
