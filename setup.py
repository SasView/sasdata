# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
    Setup for SasData
"""
import os
import subprocess
import shutil
import sys
from distutils.core import Command

from setuptools import setup

# Manage version number ######################################
with open(os.path.join("sasdata", "__init__.py")) as fid:
    for line in fid:
        if line.startswith('__version__'):
            print(line)
            VERSION = line.split('"')[1]
            break
    else:
        raise ValueError("Could not find version in src/sas/sasview/__init__.py")
##############################################################

package_dir = {}
package_data = {}
packages = []
ext_modules = []

# Remove all files that should be updated by this setup
# We do this here because application updates these files from .sasdata
# except when there is no such file

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAS_DATA_BUILD = os.path.join(CURRENT_SCRIPT_DIR, "build")

# Optionally clean before build.
dont_clean = 'update' in sys.argv
if dont_clean:
    sys.argv.remove('update')
elif os.path.exists(SAS_DATA_BUILD):
    print("Removing existing build directory", SAS_DATA_BUILD, "for a clean build")
    shutil.rmtree(SAS_DATA_BUILD)


class BuildSphinxCommand(Command):
    description = "Build Sphinx documentation."
    user_options = []

    def initialize_options(self):
        self.cwd = None

    def finalize_options(self):
        self.cwd = os.getcwd()

    def run(self):
        """Build the docs from source"""
        sys.path.append("docs/sphinx-docs")
        import build_sphinx
        build_sphinx.rebuild()


# _standard_ commands which should trigger the Qt build
build_commands = [
    'install', 'build', 'build_py', 'bdist', 'bdist_egg', 'bdist_rpm',
    'bdist_wheel', 'develop', 'test'
]

# sasdata module
package_dir["sasdata"] = os.path.join("sasdata")
packages.append("sasdata")

# sas.sascalc.dataloader
package_dir["sasdata.dataloader"] = os.path.join("sasdata", "dataloader")
package_data["sasdata.dataloader.readers"] = ['schema/*.xsd']
packages.extend(["sasdata.dataloader", "sasdata.dataloader.readers", "sasdata.dataloader.readers.schema"])

# sas.sascalc.file_converter
package_dir["sasdata.file_converter"] = os.path.join("sasdata", "file_converter")
packages.append("sasdata.file_converter")

required = ['lxml', 'h5py', 'numpy']

if os.name == 'nt':
    required.extend(['html5lib', 'reportlab'])
else:
    # 'pil' is now called 'pillow'
    required.extend(['pillow'])

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Set up SasView
setup(
    name="sasdata",
    version=VERSION,
    description="Sas Data Loader application",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="SasView Team",
    author_email="developers@sasview.org",
    url="http://sasview.org",
    license="PSF",
    keywords="small-angle x-ray and neutron scattering data loading",
    download_url="https://github.com/SasView/sasdata.git",
    package_dir=package_dir,
    packages=packages,
    package_data=package_data,
    ext_modules=ext_modules,
    install_requires=required,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            "sasview=sasdata.loader.Loader:load",
        ]
    }
)
