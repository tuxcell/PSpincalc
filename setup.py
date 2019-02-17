import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PSpincalc",
    version="0.2.0",
    author="Jose Gama",
    author_email="rxprtgama@gmail.com",
    description="Package for converting between attitude representations: DCM, Euler angles, Quaternions, and Euler vectors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tuxcell/PSpincalc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL (>= 3)",
        "Operating System :: OS Independent",
    ],
)

