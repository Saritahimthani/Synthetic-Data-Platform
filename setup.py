from setuptools import setup, find_packages

setup(
    name="synthetic-data-platform",
    version="0.1.0",
    packages=find_packages(),
    description="A platform for generating synthetic data with privacy guarantees",
    author="Your Name",
    author_email="your.email@example.com",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "matplotlib",
        "flask"
    ],
)
