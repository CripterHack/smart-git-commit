#!/usr/bin/env python3
"""
Setup script for smart-git-commit package.
"""

from setuptools import setup, find_packages
import os

# Read the content of README.md for the long description
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="smart-git-commit",
    version="0.2.3",
    description="AI-powered Git commit workflow tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Edgar Zorrilla",
    author_email="edgar@izignamx.com",
    url="https://github.com/CripterHack/smart-git-commit",
    packages=find_packages(),
    py_modules=["smart_git_commit"],
    install_requires=[
        # No external dependencies beyond Python standard library
    ],
    entry_points={
        "console_scripts": [
            "smart-git-commit=smart_git_commit:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Version Control :: Git",
    ],
    python_requires=">=3.7",
)
