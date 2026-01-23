import os
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

import dask.array as da
import numpy as np
from bioio_base import exceptions, types
from fsspec.spec import AbstractFileSystem
from ome_types import OME
from tifffile.tifffile import TiffFile

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PlaneMapEntry:
    """
    Mapping from a TIFF plane (global IFD index) to semantic Z/C/T coordinates.
    """

    ifd: int
    z: int
    c: int
    t: int


@dataclass(frozen=True)
class OmeResolution:
    """
    Result of resolving which OME metadata applies to a TIFF file.

    Attributes
    ----------
    ome : OME
        The resolved OME metadata object.
    ome_image_index : int
        Index into ome.images corresponding to this TIFF.
    using_companion_ome : bool
        Whether metadata was resolved from a companion OME file.
    """

    ome: OME
    ome_image_index: int
    using_companion_ome: bool


# -----------------------------------------------------------------------------
# OME resolution helpers
# -----------------------------------------------------------------------------


def _read_text_file(fs: AbstractFileSystem, path: str) -> str:
    """Read a UTF-8 text file from an fsspec filesystem."""
    with fs.open(path, "rb") as f:
        return f.read().decode("utf-8", errors="replace")


def _get_tiffdata_uuid_filename(tiffdata: object) -> Optional[str]:
    """
    Extract the filename referenced by a <TiffData>/<UUID> block.
    """
    uuid = getattr(tiffdata, "uuid", None)
    if uuid is None:
        return None

    return getattr(uuid, "file_name", None) or getattr(uuid, "value", None)


def select_companion_ome_image_for_tiff(ome: OME, tiff_path: str) -> int:
    """
    Select the OME <Image> whose <TiffData> UUID references this TIFF.

    Returns
    -------
    image_index : int
        Index into ome.images matching this TIFF.
    """
    tiff_basename = os.path.basename(tiff_path)

    for i, img in enumerate(ome.images):
        for td in list(img.pixels.tiff_data_blocks or []):
            ref = _get_tiffdata_uuid_filename(td)
            if ref and os.path.basename(ref) == tiff_basename:
                return i

    return 0


def resolve_ome_metadata_for_tiff(
    *,
    tiff: TiffFile,
    fs: AbstractFileSystem,
    tiff_path: str,
    companion_path: Optional[str],
    clean_metadata: bool,
    get_ome_fn: Callable[[str, bool], OME],
) -> OmeResolution:
    """
    Resolve which OME metadata applies to a TIFF file.

    Returns
    -------
    resolution : OmeResolution
        The resolved OME metadata and selected <Image> index.
    """
    embedded_ome = get_ome_fn(tiff.pages[0].description, clean_metadata)

    if not embedded_ome.binary_only:
        return OmeResolution(embedded_ome, 0, False)

    if companion_path is None:
        raise exceptions.UnsupportedFileFormatError(
            "bioio-ome-tiff",
            tiff_path,
            "Binary-only embedded OME requires a companion OME file.",
        )

    companion_ome = get_ome_fn(
        _read_text_file(fs, companion_path),
        clean_metadata,
    )

    image_index = select_companion_ome_image_for_tiff(companion_ome, tiff_path)

    return OmeResolution(companion_ome, image_index, True)


# -----------------------------------------------------------------------------
# Plane mapping helpers
# -----------------------------------------------------------------------------


def _iterate_zct_indices(
    *,
    first_z: int,
    first_c: int,
    first_t: int,
    plane_count: int,
    size_z: int,
    size_c: int,
    size_t: int,
) -> Iterator[Tuple[int, int, int]]:
    """
    Generate (z, c, t) coordinates assuming Z-fastest ordering.
    """
    start = (first_t * size_c * size_z) + (first_c * size_z) + first_z

    for i in range(plane_count):
        idx = start + i
        z = idx % size_z
        rem = idx // size_z
        c = rem % size_c
        t = rem // size_c
        yield z, c, t


def build_ifd_to_zct_plane_map(ome: OME, image_index: int) -> list[PlaneMapEntry]:
    """
    Build a per-plane mapping from TIFF IFD indices to Z/C/T coordinates.

    Returns
    -------
    mapping : list[PlaneMapEntry]
        One entry per TIFF plane.
    """
    pixels = ome.images[image_index].pixels
    tds = list(pixels.tiff_data_blocks or [])

    if not tds:
        raise exceptions.UnsupportedFileFormatError(
            "bioio-ome-tiff",
            "",
            "OME metadata contains no <TiffData> blocks.",
        )

    out: list[PlaneMapEntry] = []

    for td in tds:
        if td.ifd is None:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-ome-tiff",
                "",
                "<TiffData> block missing required IFD attribute.",
            )

        for offset, (z, c, t) in enumerate(
            _iterate_zct_indices(
                first_z=int(td.first_z or 0),
                first_c=int(td.first_c or 0),
                first_t=int(td.first_t or 0),
                plane_count=int(td.plane_count or 1),
                size_z=int(pixels.size_z),
                size_c=int(pixels.size_c),
                size_t=int(pixels.size_t),
            )
        ):
            out.append(PlaneMapEntry(td.ifd + offset, z, c, t))

    return out


# -----------------------------------------------------------------------------
# Data remapping
# -----------------------------------------------------------------------------


def remap_plane_axis_to_zct(
    *,
    tiff: TiffFile,
    tiff_scene_index: int,
    image_data: types.ArrayLike,
    tiff_axes: list[str],
    ome: OME,
    ome_image_index: int,
) -> Tuple[types.ArrayLike, list[str]]:
    """
    Remap a TIFF plane axis ('I') into explicit (T, C, Z) axes using OME metadata.

    Returns
    -------
    image_data : ArrayLike
        Data reshaped to (T, C, Z, ...).
    axes : list[str]
        Updated axis labels.

    """
    if "I" not in tiff_axes:
        return image_data, tiff_axes

    mapping = build_ifd_to_zct_plane_map(ome, ome_image_index)
    pixels = ome.images[ome_image_index].pixels

    size_t, size_c, size_z = (
        int(pixels.size_t),
        int(pixels.size_c),
        int(pixels.size_z),
    )

    pages = tiff.series[tiff_scene_index].pages
    first_ifd = pages[0].index
    plane_lut = np.full((size_t, size_c, size_z), -1, dtype=np.int64)

    for m in mapping:
        idx = m.ifd - first_ifd
        if 0 <= idx < len(pages):
            plane_lut[m.t, m.c, m.z] = idx

    if (plane_lut < 0).any():
        raise exceptions.UnsupportedFileFormatError(
            "bioio-ome-tiff",
            "",
            "OME <TiffData> mapping does not align with TIFF planes.",
        )

    i_axis = tiff_axes.index("I")
    if i_axis != 0:
        image_data = image_data.transpose(
            [i_axis] + [i for i in range(image_data.ndim) if i != i_axis]
        )

    flat = plane_lut.reshape(-1)

    if isinstance(image_data, da.Array):
        image_data = da.take(image_data, flat, axis=0).reshape(
            (size_t, size_c, size_z) + image_data.shape[1:]
        )
    else:
        image_data = np.take(image_data, flat, axis=0).reshape(
            (size_t, size_c, size_z) + image_data.shape[1:]
        )

    return image_data, ["T", "C", "Z"] + [ax for ax in tiff_axes if ax != "I"]
