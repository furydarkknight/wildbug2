"""
setup.py — MachinaSense package
"""
from setuptools import setup, find_packages

setup(
    name="machina_sense",
    version="1.0.0",
    description="Intelligent Industrial Fault Detection and Predictive Maintenance",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "cwru>=0.0.4",
        "joblib>=1.3",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "machina-sense=machina_sense.pipeline:main",
            "machina-edge-profile=machina_sense.edge_profile:main",
        ],
    },
)
