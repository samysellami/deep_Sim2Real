
import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        return f.read().splitlines()



print(load_requirements("requirements.txt"))