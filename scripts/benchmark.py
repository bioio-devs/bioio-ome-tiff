"""
    Runs bioio_base's benchmark function against the test resources in this repository
"""
import pathlib

import bioio_base.benchmark

import bioio_ome_tiff


# This file is under /scripts while the test resourcess are under /bioio_ome_tiff/tests/resources
test_resources_dir = pathlib.Path(__file__).parent.parent / "bioio_ome_tiff" / "tests" / "resources"
assert test_resources_dir.exists(), f"Test resources directory {test_resources_dir} does not exist"
test_files = list(test_resources_dir.iterdir())
bioio_base.benchmark.benchmark(bioio_ome_tiff.reader.Reader, test_files)
