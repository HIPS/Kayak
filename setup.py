import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "kayak",
    version = "0.1",
    author = "Ryan Adams, Dougal MacLaurin, Scott Linderman, Jasper Snoek, and David Duvenaud",
    author_email = "rpa@seas.harvard.edu, macLaurin@physics.harvard.edu, slinderman@seas.harvard.edu, jsnoek@seas.harvard.edu, dduvenaud@seas.harvard.edu",
    description = ("A package for automatic differentiation in deep learning models."),
    keywords = "automatic differentiation, deep learning, neural networks",
    packages=['kayak'],
    long_description=read('README.md'),
)