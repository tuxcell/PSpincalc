import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PSpincalc",
    version="0.2.3",
    author="Jose Gama",
    author_email="josephgama@yahoo.com",
    description="Package for converting between attitude representations (DCM, Euler angles, Quaternions, and Euler vectors)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuxcell/PSpincalc",
    packages=setuptools.find_packages(),
    license="GNU General Public License v3 or later (GPLv3+)"
)

