"""
Setup file for the dewolf logic module.

Based on https://github.com/pypa/sampleproject
"""
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()


setup(
    name="delogic",
    version="0.2",
    description="Logic engine for assembly conditions.",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/fkie-cad/dewolf-logic",
    author="Fraunhofer FKIE, DSO National Laboratories",
    author_email="dewolf@fkie.frauhofer.de",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Programming Language :: Python :: 3.8",
    ],
    keywords="logic, assembly, decompiler, conditions",
    packages=find_packages(include=["simplifier", "simplifier.*"]),
    package_data={"": ["*.lark"]},
    include_package_data=True,
    python_requires=">=3.8, <4",
    install_requires=["networkx", "lark~=0.11.1"],
    extras_require={
        "dev": ["black", "mypy", "pydocstyle", "isort"],
        "test": ["coverage", "pytest>=6.0.0"],
    },
    project_urls={
        "Bug Reports": "https://github.com/fkie-cad/dewolf-logic/issues",
        "Source": "https://github.com/fkie-cad/dewolf-logic",
        "DSO": "https://www.dso.org.sg/",
        "FKIE": "https://www.fkie.fraunhofer.de/",
    },
)
