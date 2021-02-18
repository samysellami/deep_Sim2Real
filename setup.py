from setuptools import setup, find_packages
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()

setup(
    name='DeepReinforcement',
    version='0.1.0',
    packages=find_packages(include=['deep_calibration']),
    install_requires=load_requirements("requirements.txt")
)