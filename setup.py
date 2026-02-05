from setuptools import setup, find_packages

setup(
    name="bab_datasets",
    version="0.1.0",
    description="Lightweight dataset loaders for nonlinear system ID experiments",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib"],
)
