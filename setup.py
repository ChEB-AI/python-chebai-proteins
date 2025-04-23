from setuptools import find_packages, setup

packages = find_packages()
print(packages)
setup(
    name="chebai-proteins",
    version="0.0.2.dev0",
    packages=packages,
    package_data={"": ["**/*.txt", "**/*.json"]},
    include_package_data=True,
    url="",
    license="",
    author="MGlauer",
    author_email="martin.glauer@ovgu.de",
    description="",
    zip_safe=False,
    python_requires=">=3.9, <3.13",
    install_requires=[
        "chebai @ git+https://github.com/ChEB-AI/python-chebai.git",
        "biopython",
        "fair-esm",
    ],
    extras_require={"dev": ["black", "isort", "pre-commit"]},
)
