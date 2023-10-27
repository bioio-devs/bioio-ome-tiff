# bioio-ome-tiff

[![Build Status](https://github.com/bioio-devs/bioio-ome-tiff/actions/workflows/ci.yml/badge.svg)](https://github.com/bioio-devs/bioio-ome-tiff/actions)
[![Documentation](https://github.com/bioio-devs/bioio-ome-tiff/actions/workflows/docs.yml/badge.svg)](https://bioio-devs.github.io/bioio-ome-tiff)

A BioIO reader plugin for reading tiff files in the OME format.

This plugin is intended to be used in conjunction with [bioio](https://github.com/bioio-devs/bioio)
---

## Installation

**Stable Release:** `pip install bioio-ome-tiff`<br>
**Development Head:** `pip install git+https://github.com/bioio-devs/bioio-ome-tiff.git`

## Quickstart

```python
from bioio_ome_tiff import Reader 

r = Reader("my-image.ext")
r.dims
```

## Documentation

For full package documentation please visit [bioio-devs.github.io/bioio-ome-tiff](https://bioio-devs.github.io/bioio-ome-tiff).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT License**
