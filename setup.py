import setuptools
from pathlib import Path


# Reading the long description from README.md
def read_long_description():
    try:
        return Path("README.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "A description of MiniRAG is currently unavailable."


# Reading dependencies from requirements.txt
def read_requirements():
    deps = []
    try:
        with open("./requirements.txt") as f:
            deps = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(
            "Warning: 'requirements.txt' not found. No dependencies will be installed."
        )
    return deps


def read_api_requirements():
    api_deps = []
    try:
        with open("./minirag/api/requirements.txt") as f:
            api_deps = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Warning: API requirements.txt not found.")
    return api_deps


long_description = read_long_description()
requirements = read_requirements()

setuptools.setup(
    name="minirag-hku",
    url="https://github.com/HKUDS/MiniRAG",
    version="0.0.2",
    author="HKUDS",
    description="MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        exclude=("tests*", "docs*")
    ),  # Automatically find packages
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",  # rec: 3.9.19
    install_requires=requirements,
    include_package_data=True,  # Includes non-code files from MANIFEST.in
    extras_require={
        "api": read_api_requirements(),  # API requirements as optional
    },
    entry_points={
        "console_scripts": [
            "minirag-server=minirag.api.minirag_server:main [api]",
        ],
    },
)
