"""Native TIFF chunks through the public lazy reader APIs."""

import math
import pickle
import tracemalloc
from pathlib import Path
from typing import Any

import fsspec
import imagecodecs
import numpy as np
import psutil
import pytest
import tifffile
from bioio_base.constants import METADATA_UNPROCESSED
from fsspec.implementations.local import LocalFileOpener
from ome_types import OME, model
from ome_types.model.simple_types import PixelType

from bioio_ome_tiff import Reader


def open_tiff_count(*paths: Path) -> int:
    targets = {str(path.resolve()) for path in paths}
    return sum(entry.path in targets for entry in psutil.Process().open_files())


@pytest.mark.parametrize("planar", ["contig", "separate"])
def test_tiled_rgb(tmp_path: Path, planar: str) -> None:
    path = tmp_path / "rgb.ome.tif"
    yxs = np.random.default_rng(43).integers(0, 256, (700, 900, 3), dtype=np.uint8)
    raw = yxs if planar == "contig" else yxs.transpose(2, 0, 1)
    tifffile.imwrite(path, raw, photometric="rgb", planarconfig=planar, tile=(256, 256))

    reader = Reader(path, chunk_dims=["Y", "X"])
    lazy = reader.get_image_dask_data("YXS")
    assert lazy.chunks == ((256, 256, 188), (256, 256, 256, 132), (3,))
    roi = np.s_[250:267, 760:, :]
    np.testing.assert_array_equal(lazy[roi].compute(), yxs[roi])
    np.testing.assert_array_equal(lazy.compute(), yxs)
    np.testing.assert_array_equal(reader.get_image_data("YXS"), yxs)
    np.testing.assert_array_equal(tifffile.imread(path), raw)


@pytest.mark.parametrize(
    "chunk_dims, c_chunks, z_chunks",
    [
        (None, (1, 1, 1), (2,)),
        (["Y", "X"], (1, 1, 1), (1, 1)),
        (["C", "Y", "X"], (3,), (1, 1)),
    ],
)
def test_tiled_volume_chunk_dims(
    tmp_path: Path,
    chunk_dims: list[str] | None,
    c_chunks: tuple[int, ...],
    z_chunks: tuple[int, ...],
) -> None:
    path = tmp_path / "volume.ome.tif"
    raw = np.random.default_rng(43).integers(0, 65535, (2, 3, 2, 70, 93), np.uint16)
    tifffile.imwrite(
        path, raw, photometric="minisblack", tile=(32, 48), metadata={"axes": "TCZYX"}
    )

    reader = Reader(path) if chunk_dims is None else Reader(path, chunk_dims=chunk_dims)
    lazy = reader.get_image_dask_data("XTZCY")
    assert lazy.chunks == ((48, 45), (1, 1), z_chunks, c_chunks, (32, 32, 6))
    expected = raw.transpose(4, 0, 2, 1, 3)
    roi = np.s_[43:55, 1, :, :, 30:]
    np.testing.assert_array_equal(lazy[roi].compute(), expected[roi])
    np.testing.assert_array_equal(lazy.compute(), expected)
    np.testing.assert_array_equal(reader.get_image_data("XTZCY"), expected)
    np.testing.assert_array_equal(tifffile.imread(path), raw)


@pytest.mark.parametrize("rowsperstrip", [75, 16])
def test_small_strip_chunk_dims(tmp_path: Path, rowsperstrip: int) -> None:
    path = tmp_path / "strips.ome.tif"
    raw = np.arange(3 * 75 * 100, dtype=np.uint16).reshape(3, 75, 100)
    tifffile.imwrite(
        path,
        raw,
        rowsperstrip=rowsperstrip,
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )

    for chunk_dims, expected in [("YX", (1, 1, 1)), ("CYX", (3,))]:
        reader = Reader(path, chunk_dims=chunk_dims)
        lazy = reader.get_image_dask_data("CYX")
        assert lazy.chunks == (expected, (75,), (100,))
        np.testing.assert_array_equal(
            lazy[:, 12:20, 7:14].compute(), raw[:, 12:20, 7:14]
        )
        np.testing.assert_array_equal(lazy.compute(), raw)
        np.testing.assert_array_equal(reader.get_image_data("CYX"), raw)


def test_large_multistrip(tmp_path: Path) -> None:
    path = tmp_path / "wide.ome.tif"
    shape = (8192, 8192)
    strip = np.full((128, shape[1]), 71, dtype=np.uint8)
    encoded = bytes(imagecodecs.zlib_encode(strip, level=1))
    tifffile.imwrite(
        path,
        (encoded for _ in range(shape[0] // strip.shape[0])),
        shape=shape,
        dtype=strip.dtype,
        compression="deflate",
        rowsperstrip=strip.shape[0],
        photometric="minisblack",
    )
    reader = Reader(path)
    lazy = reader.get_image_dask_data("YX")
    assert lazy.chunksize == strip.shape
    np.testing.assert_array_equal(lazy[125:132, 14:18].compute(), strip[:7, :4])


def test_multiscene_pyramid(tmp_path: Path) -> None:
    path = tmp_path / "pyramid.ome.tif"
    base = np.arange(256 * 320, dtype=np.uint16).reshape(256, 320)
    lower = base[::2, ::2].copy()
    other = base[:65, :91].copy() ^ 0xFFFF
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(
            base,
            tile=(64, 96),
            subifds=1,
            metadata={"PhysicalSizeY": 0.25, "PhysicalSizeX": 0.5},
        )
        writer.write(lower, tile=(32, 48), subfiletype=1)
        writer.write(other, tile=(16, 32))

    reader = Reader(path)
    assert reader.resolution_levels == (0, 1)
    base_graph = reader.get_image_dask_data("YX")
    assert base_graph.chunksize == (64, 96)

    reader.set_resolution_level(1)
    level_graph = reader.get_image_dask_data("XY")
    assert level_graph.chunks == ((48, 48, 48, 16), (32, 32, 32, 32))
    assert reader.physical_pixel_sizes.Y == 0.5
    assert reader.physical_pixel_sizes.X == 1.0
    tags = reader.xarray_dask_data.attrs[METADATA_UNPROCESSED]
    assert tags[256] == lower.shape[1]  # ImageWidth
    assert tags[257] == lower.shape[0]  # ImageLength
    np.testing.assert_array_equal(
        reader.xarray_dask_data.coords["Y"], np.arange(128) * 0.5
    )
    np.testing.assert_array_equal(
        level_graph[43:85, 27:65].compute(), lower.T[43:85, 27:65]
    )
    np.testing.assert_array_equal(reader.get_image_data("XY"), lower.T)
    np.testing.assert_array_equal(tifffile.imread(path, series=0, level=1), lower)

    reader.set_scene(1)
    assert reader.current_resolution_level == 0
    assert reader.resolution_levels == (0,)
    other_graph = reader.get_image_dask_data("YX")
    assert other_graph.chunksize == (16, 32)
    np.testing.assert_array_equal(other_graph.compute(), other)
    np.testing.assert_array_equal(reader.get_image_data("YX"), other)
    np.testing.assert_array_equal(tifffile.imread(path, series=1), other)
    np.testing.assert_array_equal(base_graph.compute(), base)
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(level_graph)).compute(), lower.T
    )
    assert open_tiff_count(path) == 0


@pytest.mark.parametrize("companion_tile", [(32, 32), (48, 48)])
def test_local_multifile_tiled(tmp_path: Path, companion_tile: tuple[int, int]) -> None:
    paths = [tmp_path / f"plane{index}.ome.tif" for index in range(2)]
    uuids = [f"urn:uuid:08cd4300-0000-0000-0000-{index:012}" for index in range(2)]
    raw = np.arange(2 * 65 * 87, dtype=np.uint16).reshape(2, 65, 87)
    pixels = model.Pixels(
        dimension_order=model.Pixels_DimensionOrder.XYZCT,
        type=PixelType.UINT16,
        size_z=2,
        size_y=65,
        size_x=87,
        size_c=1,
        size_t=1,
        channels=[model.Channel(id="Channel:0:0")],
        tiff_data_blocks=[
            model.TiffData(
                ifd=0,
                first_z=index,
                plane_count=1,
                uuid=model.TiffData.UUID(file_name=path.name, value=uuids[index]),
            )
            for index, path in enumerate(paths)
        ],
    )
    for index, path in enumerate(paths):
        ome = OME(uuid=uuids[index], images=[model.Image(id="Image:0", pixels=pixels)])
        tifffile.imwrite(
            path,
            raw[index],
            tile=(32, 32) if index == 0 else companion_tile,
            metadata=None,
            ome=False,
            description=ome.to_xml().encode(),
        )
    np.testing.assert_array_equal(tifffile.imread(paths[0]), raw)
    reader = Reader(paths[0])
    assert open_tiff_count(*paths) == 0
    lazy = reader.get_image_dask_data("ZYX")
    assert open_tiff_count(*paths) == 0
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(lazy))[:, 29:35, 60:].compute(), raw[:, 29:35, 60:]
    )
    np.testing.assert_array_equal(lazy.compute(), raw)
    np.testing.assert_array_equal(reader.get_image_data("ZYX"), raw)
    assert open_tiff_count(*paths) == 0
    yx = (32, 32) if companion_tile == (32, 32) else raw.shape[-2:]
    assert lazy.chunksize == (2, *yx)


def test_memory_filesystem_tiled(tmp_path: Path) -> None:
    local = tmp_path / "memory.ome.tif"
    raw = np.arange(50 * 68, dtype=np.uint16).reshape(50, 68)
    tifffile.imwrite(local, raw, tile=(32, 32))
    fs = fsspec.filesystem("memory")
    with fs.open("/bioio-native-chunks.ome.tif", "wb") as stream:
        stream.write(local.read_bytes())

    reader = Reader("memory:///bioio-native-chunks.ome.tif")
    lazy = reader.get_image_dask_data("YX")
    assert lazy.chunks == ((32, 18), (32, 32, 4))
    restored = pickle.loads(pickle.dumps(lazy))
    np.testing.assert_array_equal(restored[29:, 61:].compute(), raw[29:, 61:])
    np.testing.assert_array_equal(restored.compute(), raw)
    np.testing.assert_array_equal(reader.get_image_data("YX"), raw)


def test_large_tiled_roi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "large.ome.tif"
    tile_size = 512
    tiles_per_axis = 16
    side = tiles_per_axis * tile_size
    colors = np.array(
        [[17, 27, 37], [50, 60, 70], [81, 91, 101], [120, 130, 140]], dtype=np.uint8
    )
    encoded = [
        bytes(
            imagecodecs.zlib_encode(np.tile(color, (tile_size, tile_size, 1)), level=1)
        )
        for color in colors
    ]
    tifffile.imwrite(
        path,
        (
            encoded[(y % 2) * 2 + x % 2]
            for y in range(tiles_per_axis)
            for x in range(tiles_per_axis)
        ),
        shape=(side, side, 3),
        dtype=colors.dtype,
        compression="deflate",
        photometric="rgb",
        tile=(tile_size, tile_size),
    )
    with tifffile.TiffFile(path) as tiff:
        page = tiff.series[0].keyframe
        assert page.is_tiled
        assert (page.tilelength, page.tilewidth) == (tile_size, tile_size)
        segments = list(zip(page.dataoffsets, page.databytecounts))
        assert len(segments) == tiles_per_axis**2
        assert open_tiff_count(path) >= 1  # Positive control for the FD sensor.

    reads: list[tuple[int, int]] = []
    original_read = LocalFileOpener.read

    def track_read(handle: LocalFileOpener, *args: Any, **kwargs: Any) -> bytes:
        offset = handle.tell()
        result = original_read(handle, *args, **kwargs)
        if handle.path == str(path):
            reads.append((offset, offset + len(result)))
        return result

    def tiles_touched() -> set[int]:
        return {
            index
            for index, (offset, length) in enumerate(segments)
            if any(start < offset + length and offset < end for start, end in reads)
        }

    monkeypatch.setattr(LocalFileOpener, "read", track_read)
    reader = Reader(path)
    assert open_tiff_count(path) == 0
    lazy = reader.get_image_dask_data("YXS")
    assert open_tiff_count(path) == 0
    assert reader.dask_data is reader.xarray_dask_data.data
    assert open_tiff_count(path) == 0
    assert not tiles_touched()

    restored = pickle.loads(pickle.dumps(lazy))
    reads.clear()
    tracemalloc.start()
    try:
        pixel = restored[0, 0, 0].compute(scheduler="synchronous")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert open_tiff_count(path) == 0
    assert pixel == colors[0, 0]
    assert tiles_touched() == {0}
    assert peak < 16 * 1024**2
    assert lazy.chunksize == (tile_size, tile_size, 3)

    reads.clear()
    crop = restored[500:524, 500:524].compute(scheduler="synchronous")
    quadrants = np.indices((24, 24)) // 12
    np.testing.assert_array_equal(crop, colors[quadrants[0] * 2 + quadrants[1]])
    assert tiles_touched() == {0, 1, tiles_per_axis, tiles_per_axis + 1}
    assert open_tiff_count(path) == 0
    assert math.prod(lazy.shape) == 201_326_592
