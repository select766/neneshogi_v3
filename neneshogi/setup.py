# Add to import path:
# python setup.py develop

from setuptools import setup, find_packages, Extension

setup(
    name="neneshogi",
    version="3.0.0",
    packages=find_packages(),
    test_suite='test'
)
