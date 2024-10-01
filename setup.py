# setup.py

from setuptools import setup, find_packages

setup(
    name="automf",
    version="0.1.0",
    description="AutoMF: Automatic Model Fusion for model selection and weighted averaging",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ye Su",
    author_email="ye.su@siat.ac.cn",
    url="https://github.com/SIAT-Suye/AutoMF",  
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
