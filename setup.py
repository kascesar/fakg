import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fukg-kascesar",
    version="0.0.1",
    author="Cesar Augusto Munoz Araya",
    author_email="kas.cesar@gmail.com",
    description="tensorflow generators with data augmentation for tensorflow models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kascesar/fakg.git",
    project_urls={
        "Bug Tracker": "https://github.com/kascesar/fakg.git/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3.0",
        "Operating System :: Linux",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
