import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyslingshot",
    version="0.0.1",
    author="Jacob Moss",
    author_email="jm2311@cam.ac.uk",
    description="Python implementation of the Slingshot pseudotime algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mossjacob/pyslingshot",
    project_urls={
        "Bug Tracker": "https://github.com/mossjacob/pyslingshot/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)
