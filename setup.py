"""Only needed so versioneer can derive the version from git tags; metadata lives in pyproject.toml."""

import versioneer
from setuptools import setup

setup(version=versioneer.get_version(), cmdclass=versioneer.get_cmdclass())
