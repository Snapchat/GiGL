# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import pathlib
import datetime


curr_file_dir = pathlib.Path(__file__).parent.resolve()

project = "GiGL"
author = "Snap Inc"
copyright = f"{datetime.datetime.now().year}, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# https://www.sphinx-doc.org/en/master/usage/extensions/index.html
extensions = [
    "sphinx.ext.autodoc",  # Pull in documentation from docstrings
    "sphinx.ext.viewcode",  # Add links to the source code
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

autodoc_default_options = {
    # Generate automatic documentation for all members of the target module
    'members': True,
    # Generate automatic documentation for members of the target module that don’t have a docstring or doc-comment
    'undoc-members': True,
    # Insert the class’s base classes after the class signature
    'show-inheritance': True,
   # Generate automatic documentation for special members of the target module
    'special-members': '__init__',
    # Exclude the given names from the members to document
    'exclude-members': "__weakref__,__dict__,__module__,__class__,__abstractmethods__",
}

templates_path = [
    'gh_pages_source/_templates'
]
html_static_path = [
    'gh_pages_source/_static',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme" # https://pydata-sphinx-theme.readthedocs.io/en/stable/
# Disable showing the rst source link - its not useful, neither is it asthetically pleasing.
# We enable the edit button instead so the src code can be seen and edited conveniently; see use of use_edit_page_button below.
html_show_sourcelink = False # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/source-buttons.html#view-source-link
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
    },
    # Allow user to directly edit the documentation
    # https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/source-buttons.html#add-an-edit-button
    "use_edit_page_button": True,
}

# Allow user to directly edit the documentation
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/source-buttons.html#github
html_context = {
    "github_user": "Snapchat",
    "github_repo": "GiGL",
    "github_version": "main",
    "doc_path": "/",
}
