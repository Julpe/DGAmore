from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="DGAmore",
    version="1.0.0",
    description="Self-consistent multi-orbital ladder-DGA code for the multi-band Hubbard Model",
    author="Julian Peil",
    author_email="julian.peil@tuwien.ac.at",
    packages=find_packages(),
    install_requires=requirements,
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: Linux",
    ],
    scripts=["dgamore/DGAmore.py", "dgamore/symmetrize_new.py"],
    python_requires=">=3.12",
)
