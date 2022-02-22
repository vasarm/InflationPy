from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="inflationpy",
        versrion="0.0.1",
        author="Martin Vasar",
        packages=[package for package in find_packages() if package.startswith("inflationpy")],
        package_data={"inflationpy": ["py.typed", "version.txt"]},
        install_requires=[
            "numpy",
            "scipy",
            "sympy",
            "mpmath",
            "matplotlib",
        ],
        extras_require={
            "tests": [
                # Run tests
                "pytest>=6.0",
                # Linter
                "flake8>=3.8",
                # Reformat
                "black",
            ],
            "docs": [
                "sphinx",
                "sphinx-autobuild",
                "sphinx-rtd-theme",
                # For spelling
                "sphinxcontrib.spelling",
                # Type hints support
                "sphinx-autodoc-typehints",
            ],
        },
        python_requires=">=3.7",
    )
