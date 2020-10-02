import setuptools

setuptools.setup(
    name="pochade",
    packages=setuptools.find_packages(include=["pochade"]),
    version="0.1.0",
    description="Python tool to extract colors from images",
    author="Michael Lusher",
    license="MIT",
    test_require=["nose"],
    install_require=["PIL"],
    test_suite="nose.collector"
)