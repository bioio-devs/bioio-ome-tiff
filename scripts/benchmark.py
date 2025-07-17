"""
    Runs bioio_base's benchmark function against the test resources in this repository
"""
import pathlib

import bioio_base.benchmark

import bioio_ome_tiff


# This file is under /scripts while the test resourcess are under /bioio_ome_tiff/tests/resources
test_resources_dir = pathlib.Path(__file__).parent.parent / "bioio_ome_tiff" / "tests" / "resources"
assert test_resources_dir.exists(), f"Test resources directory {test_resources_dir} does not exist"
test_files = [
    test_file
    for test_file in test_resources_dir.iterdir()
    # Exclude some files that are -intended- to fail the reader
    if test_file.name not in {"image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos001_000.ome.tif", "s_1_t_1_c_2_z_1.lif", "example.txt"}
]
bioio_base.benchmark.benchmark(bioio_ome_tiff.reader.Reader, test_files)
