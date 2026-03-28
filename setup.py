from setuptools import setup, find_packages

setup(
    name="cissn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "rich>=13.0.0",
        "shap>=0.41.0",
        "networkx>=2.8.0"
    ],
    author="Emanuele Nardone",
    author_email="emanuele.nardone@unicas.it",
    description="Conformally Calibrated Interpretable State-Space Networks (CISSN)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Narden91/CISSN",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)