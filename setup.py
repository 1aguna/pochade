import setuptools
import os
import sys

base_dir = os.path.dirname(__file__)
src_dir = os.path.join(base_dir, "src")

# When executing the setup.py, we need to be able to import ourselves, this
# means that we need to add the src/ directory to the sys.path.
sys.path.insert(0, src_dir)

setuptools.setup(
    package_dir={"": "src"},
    name="pochade",
    packages=setuptools.find_packages(where="src"),
    version="0.1.0",
    description="Python tool to extract colors from images",
    author="Michael Lusher",
    license="MIT",
    test_require=["nose"],
    install_require=["PIL"],
    test_suite="nose.collector"
)