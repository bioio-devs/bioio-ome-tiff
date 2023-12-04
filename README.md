# bioio-ome-tiff

[![Build Status](https://github.com/bioio-devs/bioio-ome-tiff/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-ome-tiff/actions)
[![PyPI version](https://badge.fury.io/py/bioio-ome-tiff.svg)](https://badge.fury.io/py/bioio-ome-tiff)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.9+](https://img.shields.io/badge/python-3.9,3.10,3.11-blue.svg)](https://www.python.org/downloads/release/python-390/)

A BioIO reader plugin for reading OME TIFF files using `tifffile`

---


## Documentation

[See the full documentation on our GitHub pages site](https://bioio-devs.github.io/bioio/OVERVIEW.html) - the generic use and installation instructions there will work for this package.

Information about the base reader this package relies on can be found in the `bioio-base` repository [here](https://github.com/bioio-devs/bioio-base)

## Installation

**Stable Release:** `pip install bioio-ome-tiff`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-ome-tiff.git`

## Example Usage (see full documentation for more examples)

Install bioio-ome-tiff alongside bioio:

`pip install bioio bioio-ome-tiff`


This example shows a simple use case for just accessing the pixel data of the image
by explicitly passing this `Reader` into the `BioImage`. Passing the `Reader` into
the `BioImage` instance is optional as `bioio` will automatically detect installed
plug-ins and auto-select the most recently installed plug-in that supports the file
passed in.
```python
from bioio import BioImage
import bioio_ome_tiff

img = BioImage("my_file.ome.tiff", reader=bioio_ome_tiff.Reader)
img.data
```

## Issues
[_Click here to view all open issues in bioio-devs organization at once_](https://github.com/search?q=user%3Abioio-devs+is%3Aissue+is%3Aopen&type=issues&ref=advsearch) or check this repository's issue tab.


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.
