# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import datetime


# Add source folder to path for autodoc
sys.path.insert(0, os.path.abspath("../../../python/gigl"))

project = 'GiGL'
copyright = f'{datetime.date.today().year}, Snap Inc'
author = 'Snap Inc'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_design",
]
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "../../../docs/assets/images/gigl.png"
html_static_path = ['_static']
source_suffix = ['.rst', '.md']

html_css_files = [
    "css/custom.css",
    "https://fonts.googleapis.com/css?family=Inter:100,200,300,regular,500,600,700,800,900",
]

myst_enable_extensions = [
    'colon_fence',
]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Snapchat/GiGL",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/gigl/",
            "icon": "fa-custom fa-pypi",
        },
    ],
    # alternative way to set twitter and github header icons
    # "github_url": "https://github.com/pydata/pydata-sphinx-theme",
    # "twitter_url": "https://twitter.com/PyData",
    "logo": {
        "text": "GiGL",
        "image_dark": "../../../docs/assets/images/gigl.png",
    },
    # "navbar_start": ["navbar-logo"],
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    # "navbar_persistent": ["search-button"],
    # "primary_sidebar_end": ["custom-template", "sidebar-ethical-ads"],
    # "article_footer_items": ["test", "test"],
    # "content_footer_items": ["test", "test"],
    #"footer_start": ["copyright"],
    #"footer_center": ["sphinx-version"],
    #"secondary_sidebar_items": {
    #    "**/*": ["page-toc", "edit-this-page", "sourcelink"],
    #    "examples/no-sidebar": [],
    #},
}
