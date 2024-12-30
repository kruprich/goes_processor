from setuptools import setup, find_packages

setup(
    name="your_library",
    version="0.1.0",
    description="A library for downloading and processing GOES-16 data",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "aiohttp",
        "google-cloud-storage",
        "pandas",
        "numpy",
        "netCDF4"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)