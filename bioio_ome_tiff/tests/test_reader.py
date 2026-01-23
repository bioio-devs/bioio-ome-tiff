#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
from bioio_base import dimensions, exceptions, test_utilities
from ome_types import OME

from bioio_ome_tiff import Reader

from .conftest import LOCAL_RESOURCES_DIR

COMPANION_RESOURCES_DIR = LOCAL_RESOURCES_DIR / "companion_resources"


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Bright"],
            (None, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 10, 1, 1736, 1776),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [f"C:{i}" for i in range(10)],  # This is the actual metadata
            (None, None, None),
        ),
        (
            # This is actually an OME-TIFF file
            # Shows we don't just work off of extensions
            # But the content of the file
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (1, 2, 1, 32, 32, 3),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0", "Channel:0:1"],
            (None, None, None),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            ("Image:0", "Image:1", "Image:2"),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            ("Image:0", "Image:1", "Image:2"),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:2",
            ("Image:0", "Image:1", "Image:2"),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "s_1_t_1_c_2_z_1.lif",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:1",
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=IndexError),
        ),
        pytest.param(
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:3",
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=IndexError),
        ),
    ],
)
def test_ome_tiff_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
    )


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "pre-variance-cfe.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 9, 65, 600, 900),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [
                "Bright_2",
                "EGFP",
                "CMDRP",
                "H3342",
                "SEG_STRUCT",
                "SEG_Memb",
                "SEG_DNA",
                "CON_Memb",
                "CON_DNA",
            ],
            (0.29, 0.10833333333333334, 0.10833333333333334),
        ),
        (
            "variance-cfe.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 9, 65, 600, 900),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [
                "CMDRP",
                "EGFP",
                "H3342",
                "Bright_2",
                "SEG_STRUCT",
                "SEG_Memb",
                "SEG_DNA",
                "CON_Memb",
                "CON_DNA",
            ],
            (0.29, 0.10833333333333332, 0.10833333333333332),
        ),
        (
            "actk.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 6, 65, 233, 345),
            np.float64,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [
                "nucleus_segmentation",
                "membrane_segmentation",
                "dna",
                "membrane",
                "structure",
                "brightfield",
            ],
            (0.29, 0.29, 0.29),
        ),
    ],
)
def test_ome_tiff_reader_large_files(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
    )


@pytest.mark.parametrize(
    "filename, "
    "first_scene_id, "
    "first_scene_shape, "
    "second_scene_id, "
    "second_scene_shape",
    [
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            (1, 3, 5, 325, 475),
            "Image:1",
            (1, 3, 5, 325, 475),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            (1, 3, 5, 325, 475),
            "Image:2",
            (1, 3, 5, 325, 475),
        ),
    ],
)
def test_multi_scene_ome_tiff_reader(
    filename: str,
    first_scene_id: str,
    first_scene_shape: Tuple[int, ...],
    second_scene_id: str,
    second_scene_shape: Tuple[int, ...],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_multi_scene_image_read_checks(
        ImageContainer=Reader,
        image=uri,
        first_scene_id=first_scene_id,
        first_scene_shape=first_scene_shape,
        first_scene_dtype=np.dtype(np.uint16),
        second_scene_id=second_scene_id,
        second_scene_shape=second_scene_shape,
        second_scene_dtype=np.dtype(np.uint16),
    )


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        # TODO:
        # Select a different level besides level 0
        # TiffReader / Reader defaults to reading level 0
        (
            "variable_scene_shape_first_scene_pyramid.ome.tiff",
            "Image:0",
            ("Image:0", "Image:1"),
            (1, 3, 1, 6184, 7712),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "mCher", "PGC"],
            (None, 0.9082107048835328, 0.9082107048835328),
        ),
        (
            "variable_scene_shape_first_scene_pyramid.ome.tiff",
            "Image:1",
            ("Image:0", "Image:1"),
            (1, 1, 1, 2030, 422),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:1:0"],
            (None, 0.9082107048835328, 0.9082107048835328),
        ),
    ],
)
def test_multi_resolution_ome_tiff_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
    )


@pytest.mark.parametrize(
    "filename",
    [
        # Pipline 4 is valid, :tada:
        "pipeline-4.ome.tiff",
        # Some of our test files are valid, :tada:
        "s_1_t_1_c_1_z_1.ome.tiff",
        "s_3_t_1_c_3_z_5.ome.tiff",
        # A lot of our files aren't valid, :upside-down-smiley:
        # These files have invalid schema / layout
        # but recently ome-types default settings are more lenient!
        "3d-cell-viewer.ome.tiff",
        "pre-variance-cfe.ome.tiff",
        "variance-cfe.ome.tiff",
    ],
)
def test_known_errors_without_cleaning(filename: str) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    Reader(uri, clean_metadata=False, fs_kwargs=dict(anon=True))


@pytest.mark.parametrize(
    "filename",
    [
        "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif",
    ],
)
def test_micromanager_ome_tiff_main_file(filename: str) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # MicroManager will split up multi-scene image sets into multiple files
    # tifffile will then read in all of the scenes at once when it detects
    # the file is a micromanager file set
    # resulting in this single file truly only containing the binary for a
    # single scene but containing the metadata for all files in the set
    # and, while this file only contains the binary for itself, tifffile will
    # read the image data for the linked files

    # Run image read checks on the first scene
    # (this files binary data)
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene="Image:0",
        expected_scenes=("Image:0", "Image:1"),
        expected_current_scene="Image:0",
        expected_shape=(50, 3, 5, 256, 256),
        expected_dtype=np.dtype(np.uint16),
        expected_dims_order=dimensions.DEFAULT_DIMENSION_ORDER,
        expected_channel_names=["Cy5", "DAPI", "FITC"],
        expected_physical_pixel_sizes=(1.75, 2.0, 2.0),
        expected_metadata_type=OME,
    )


@pytest.mark.parametrize(
    "filename",
    [
        "actk.ome.tiff",
    ],
)
def test_ome_metadata(filename: str) -> None:
    # Get full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Init image
    img = Reader(uri)

    # Test the transform
    assert isinstance(img.ome_metadata, OME)


@pytest.mark.parametrize(
    "filename",
    [
        "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif",
    ],
)
def test_micromanager_metadata(filename: str) -> None:
    # Get full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Init image
    img = Reader(uri)
    metadata = img.micromanager_metadata

    # Test the transform
    assert isinstance(metadata, dict)
    assert not (278 in metadata)  # non-mm keys do not exist
    assert metadata["ChNames"] == ["Cy5", "DAPI", "FITC"]
    assert metadata["MicroManagerVersion"] == "2.0.0-gamma1-20201209"


# --- Companion OME tests ------------------------------------------------------
def _companion_paths(main_tiff: str, companion_ome: str) -> tuple[Path, Path]:
    return (
        COMPANION_RESOURCES_DIR / main_tiff,
        COMPANION_RESOURCES_DIR / companion_ome,
    )


@pytest.mark.parametrize(
    "main_tiff",
    [
        "20250910_Test4ch_2ROI_3Z_1_w1confCy5_sg1_s1.ome.tif",
    ],
)
def test_companion_binary_only_requires_companion(main_tiff: str) -> None:
    uri = COMPANION_RESOURCES_DIR / main_tiff
    with pytest.raises(exceptions.UnsupportedFileFormatError, match=r"Binary-only"):
        Reader(uri)


@pytest.mark.parametrize(
    "main_tiff, companion_ome, set_scene, expected_scenes, expected_shape, "
    "expected_dtype, expected_dims_order, expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        pytest.param(
            "20250910_Test4ch_2ROI_3Z_1_w1confCy5_sg1_s1.ome.tif",
            "20250910_test4ch_2roi_3z_1_sg1.companion.ome",
            "Image:0",
            (
                "Image:0",
                "Image:1",
                "Image:2",
                "Image:3",
                "Image:4",
                "Image:5",
                "Image:6",
                "Image:7",
                "Image:8",
            ),
            (1, 4, 3, 512, 512),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,  # "TCZYX"
            ["confCy5", "confmCherry", "confGFP", "confDAPI"],
            (2.0, 1.29721379, 1.29721379),
            id="sg1_roi_companion_contract",
        ),
        pytest.param(
            "20250910_Test4ch_2ROI_3Z_1_w1confCy5_sg2_s1.ome.tif",
            "20250910_test4ch_2roi_3z_1_sg2.companion.ome",
            "Image:0",
            (
                "Image:0",
                "Image:1",
                "Image:2",
                "Image:3",
                "Image:4",
                "Image:5",
            ),
            (1, 4, 3, 512, 512),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["confCy5", "confmCherry", "confGFP", "confDAPI"],
            (2.0, 1.29721379, 1.29721379),
            id="sg2_roi_companion_contract",
        ),
    ],
)
def test_companion_ome_tiff_reader_contract(
    main_tiff: str,
    companion_ome: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    """
    Companion datasets should satisfy the same reader contract checks.
    """
    uri, comp = _companion_paths(main_tiff, companion_ome)

    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=OME,
        reader_kwargs=dict(
            fs_kwargs=dict(anon=True),
            companion_path=comp,
        ),
    )


@pytest.mark.parametrize(
    "main_tiff, companion_ome",
    [
        (
            "20250910_Test4ch_2ROI_3Z_1_w1confCy5_sg1_s1.ome.tif",
            "20250910_test4ch_2roi_3z_1_sg1.companion.ome",
        ),
    ],
)
def test_companion_mode_sanity_selected_image_has_tiffdata(
    main_tiff: str, companion_ome: str
) -> None:
    uri, comp = _companion_paths(main_tiff, companion_ome)

    img = Reader(uri, companion_path=comp)

    assert isinstance(img.ome_metadata, OME)
    assert img.scenes is not None and len(img.scenes) > 0

    assert getattr(img, "_using_companion_ome", False) is True
    assert isinstance(getattr(img, "_ome_image_index", None), int)

    ome_idx = img._ome_image_index
    tds = list(img.ome_metadata.images[ome_idx].pixels.tiff_data_blocks or [])
    assert len(tds) > 0


@pytest.mark.parametrize(
    "main_tiff_a, companion_ome_a, main_tiff_b, companion_ome_b",
    [
        (
            "20250910_Test4ch_2ROI_3Z_1_w1confCy5_sg1_s1.ome.tif",
            "20250910_test4ch_2roi_3z_1_sg1.companion.ome",
            "20250910_Test4ch_2ROI_3Z_1_w1confCy5_sg2_s1.ome.tif",
            "20250910_test4ch_2roi_3z_1_sg2.companion.ome",
        ),
    ],
)
def test_companion_rois_match_shape_and_dims(
    main_tiff_a: str,
    companion_ome_a: str,
    main_tiff_b: str,
    companion_ome_b: str,
) -> None:
    uri_a, comp_a = _companion_paths(main_tiff_a, companion_ome_a)
    uri_b, comp_b = _companion_paths(main_tiff_b, companion_ome_b)

    a = Reader(uri_a, companion_path=comp_a)
    b = Reader(uri_b, companion_path=comp_b)

    da0 = a.xarray_dask_data
    da1 = b.xarray_dask_data

    assert da0 is not None and da1 is not None
    assert tuple(da0.dims) == tuple("TCZYX")
    assert tuple(da1.dims) == tuple("TCZYX")
    assert tuple(da0.shape) == tuple(da1.shape)
