# Configuration file for the Sphinx documentation builder.

# -- Project information ----------------------------------------------------
import datetime
import os

from sphinx.domains.python import PythonDomain

from sasdata import __version__ as sasdata_version

if os.path.exists('rst_prolog'):
    with open('rst_prolog') as fid:
        rst_prolog = fid.read()

# General information about the project.
year = datetime.datetime.now().year

project = 'SasData'
copyright = f'{year}, The SasView Project'
author = 'SasView'
release = sasdata_version

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'default'

# Ignore missing references to sections in SasView documentation
def on_missing_reference(app, env, node, contnode):
    if node["reftarget"] in [
        "file_converter_tool",
        "image_viewer_tool",
        "sans_calculator_tool",
    ]:
        return contnode
    else:
        return None

# Bypass stupid sphinx handling of multiple classes with members named *type*
class PatchedPythonDomain(PythonDomain):
    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        if 'refspecific' in node:
            del node['refspecific']
        return super(PatchedPythonDomain, self).resolve_xref(
            env, fromdocname, builder, typ, target, node, contnode)

def setup(app):
    app.connect("missing-reference", on_missing_reference)
    app.add_domain(PatchedPythonDomain, override=True)
