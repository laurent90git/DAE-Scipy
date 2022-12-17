from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="scipyDAE",
    version="0.0.1",
    author="Laurent François",
    author_email="laurent.francois@polytechnique.edu",
    description="Modification of Scipy's Radau solver for the solution of DAEs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/laurent90git/DAE-Scipy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','scipy'],
)

# run "python setup.py develop" once !

