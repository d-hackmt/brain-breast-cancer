# to make our modular logic into package

from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
    
setup(
    name="oncoscan",
    version="0.1",
    author="divesh",
    packages=find_packages(),
    install_requires = requirements,
)