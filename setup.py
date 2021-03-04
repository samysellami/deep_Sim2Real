from setuptools import setup, find_packages
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()

setup(
    name='Deep_Sim2Real',
    version='0.1.0',
    packages=find_packages(include=['deep_calibration', 
    								'deep_simulation']),
    entry_points={
        "console_scripts": [
            "deep-calibration = deep_calibration.__main__:main",
        ],
    },
    install_requires=load_requirements("requirements.txt")
)