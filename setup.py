"""setup.py for this project"""

from setuptools import find_packages, setup

PACKAGE_VERSION = "0.1.0"

requirements = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "python-dotenv",
    "autopep8",
    "pylint",
    "mlops",
    "fastapi",
    "streamlit",
    "scikit-learn"
]

setup(
    name="lending-club-mlops",
    version=PACKAGE_VERSION,
    packages=find_packages(where=".", exclude=["tests", "tests.*"]),
    install_requires=requirements,
    python_requires="==3.10.6",
    keywords="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "xyz = lending_club_mlops.cli:main"
        ]
    }
)
