# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib

curr_file_dir = pathlib.Path(__file__).parent.resolve()

project = 'GiGL'
copyright = '2025, Snap Inc'
author = 'Snap Inc'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# https://www.sphinx-doc.org/en/master/usage/extensions/index.html
extensions = [
    "sphinx.ext.autodoc",  # Pull in documentation from docstrings
    "sphinx.ext.autosummary", # Generates function/method/attribute summary lists
    "myst_parser", # Parsing markdown files: https://myst-parser.readthedocs.io/en/v0.15.1/sphinx/intro.html
    "sphinx_design", # needed by themes
]

myst_enable_extensions = [
    "html_image", # Convert <img> tags in markdown files; https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#html-images
]

include_patterns = [
    "docs/**",
    "python/**",
    "snapchat/**",
    "index.rst",
]


templates_path = [
    'gh_pages_source/_templates'
]
html_static_path = [
    'gh_pages_source/_static',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme" # https://pydata-sphinx-theme.readthedocs.io/en/stable/
html_logo = "docs/assets/images/gigl.png"
html_theme_options = {
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/header-links.html#icon-links
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Snapchat/GiGL",
            "icon": "fa-brands fa-github",
        }
    ],
    "logo": {
        "text": "GiGL",
        "image_dark": "docs/assets/images/gigl.png",
    }
}
