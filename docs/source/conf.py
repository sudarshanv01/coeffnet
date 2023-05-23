project = "minimal-basis"
copyright = "2023, Sudarshan Vijay"
author = "Sudarshan Vijay"
release = "0.0.1"

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))


extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.autosummary"]
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
