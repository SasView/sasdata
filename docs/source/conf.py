# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

from sasdata import __version__ as sasdata_version

project = 'sasdata'
copyright = '2023, SasView'
author = 'SasView'
release = sasdata_version

# -- General configuration ---------------------------------------------------

extensions = []

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'
html_static_path = ['_static']
