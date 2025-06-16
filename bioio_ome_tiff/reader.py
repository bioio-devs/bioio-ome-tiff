#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import dask.array as da
import numpy as np
import xarray as xr
from bioio_base import constants, dimensions, exceptions, io, reader, transforms, types
from dask import delayed
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from ome_types import OME, from_xml
from pydantic import ValidationError
from tifffile.tifffile import TiffFile, TiffFileError, TiffTags, imread
from xmlschema import XMLSchemaValidationError
from xmlschema.exceptions import XMLSchemaValueError

from .utils import (
    clean_ome_xml_for_known_issues,
    get_coords_from_ome,
    get_dims_from_ome,
    physical_pixel_sizes,
)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Reader(reader.Reader):
    """
    Wraps the tifffile and ome-types APIs to provide the same BioIO Reader Plugin
    for volumetric OME-TIFF images.
    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    chunk_dims: List[str]
        Which dimensions to create chunks for.
        Default: DEFAULT_CHUNK_DIMS
        Note: Dimensions.SpatialY, Dimensions.SpatialX, and DimensionNames.Samples,
        will always be added to the list if not present during dask array
        construction.
    clean_metadata: bool
        Should the OME XML metadata found in the file be cleaned for known
        AICSImageIO 3.x and earlier created errors.
        Default: True (Clean the metadata for known errors)
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    Notes
    -----
    If the OME metadata in your file isn't OME schema compilant or does not validate
    this will fail to read your file and raise an exception.
    If the OME metadata in your file doesn't use the latest OME schema (2016-06),
    this reader will make a request to the referenced remote OME schema to validate.
    """

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _micromanager_metadata: Optional[Dict[str | int, Any]] = None
    _mosaic_xarray_dask_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[dimensions.Dimensions] = None
    _metadata: Optional[Any] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    # Do not provide default value because
    # they may not need to be used by your reader (i.e. input param is an array)
    _fs: "AbstractFileSystem"
    _path: str

    @staticmethod
    def _get_ome(ome_xml: str, clean_metadata: bool = True) -> OME:
        # To clean or not to clean, that is the question
        if clean_metadata:
            ome_xml = clean_ome_xml_for_known_issues(ome_xml)

        return from_xml(ome_xml, parser="lxml")

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem, path: str, clean_metadata: bool = True, **kwargs: Any
    ) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    # Get first page description (aka the description tag in general)
                    # after Tifffile version 2023.3.15 mmstack images read all scenes
                    # into tiff.pages[0]
                    xml = tiff.pages[0].description
                    ome = Reader._get_ome(xml, clean_metadata)

                    # Handle no images in metadata
                    # this commonly means it is a "BinaryData" OME file
                    # i.e. a non-main OME-TIFF from MicroManager or similar
                    # in this case, because it's not the main file we want to just role
                    # back to TiffReader
                    if ome.binary_only:
                        raise exceptions.UnsupportedFileFormatError(
                            "bioio-ome-tiff",
                            path,
                            "The OME metadata indicates this is a binary OME-TIFF.",
                        )
                    return True

        # tifffile exceptions
        except (TiffFileError, TypeError) as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-ome-tiff",
                path,
                str(e),
            )

        # xml parse errors
        except ET.ParseError as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-ome-tiff",
                path,
                f"Failed to parse XML for the provided file. Error: {e}",
            )

        # invalid OME XMl
        except (XMLSchemaValueError, XMLSchemaValidationError, ValidationError) as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-ome-tiff",
                path,
                f"OME XML validation failed. Error: {e}",
            )

        # cant connect to external schema resource (no internet conection)
        except URLError as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-ome-tiff",
                path,
                f"Could not validate OME XML against referenced schema "
                f"(no internet connection). "
                f"Error: {e}",
            )

        except Exception as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-ome-tiff",
                path,
                str(e),
            )

    @staticmethod
    def _guess_ome_dim_order(tiff: TiffFile, ome: OME, scene_index: int) -> List[str]:
        """
        Guess the dimension order based on OME metadata and actual TIFF data.
        Parameters
        -------
        tiff: TiffFile
            A constructed TIFF object to retrieve data from.
        ome: OME
            A constructed OME object to retrieve data from.
        scene_index: int
            The current operating scene index to pull metadata from.
        Returns
        -------
        dims: List[str]
            Educated guess of the dimension order for the file
        """
        dims_from_ome = get_dims_from_ome(ome, scene_index)

        # Assumes the dimensions coming from here are align semantically
        # with the dimensions specified in this package. Possible T dimension
        # is not equivalent to T dimension here. However, any dimensions
        # not also found in OME will be omitted.
        dims_from_tiff_axes = list(tiff.series[scene_index].axes)

        # If the OME metadata does not have a "S" dimension but the tiff axes
        # does, and the OME metadata has a "C" dimension but the tiff axes does
        # not, then we can assume that the "S" dimension in the tiff axes is
        # actually the "C" dimension in the OME metadata.
        if (
            "S" in dims_from_tiff_axes
            and "S" not in dims_from_ome
            and "C" in dims_from_ome
            and "C" not in dims_from_tiff_axes
        ):
            dims_from_tiff_axes = [
                dim if dim != "S" else "C" for dim in dims_from_tiff_axes
            ]

        # Adjust the guess of what the dimensions are based on the combined
        # information from the tiff axes and the OME metadata.
        # Necessary since while OME metadata should be source of truth, it
        # does not provide enough data to guess which dimension is Samples
        # for RGB files
        dims = [dim for dim in dims_from_ome if dim not in dims_from_tiff_axes]
        dims += [dim for dim in dims_from_tiff_axes if dim in dims_from_ome]
        return dims

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = dimensions.DEFAULT_CHUNK_DIMS,
        clean_metadata: bool = True,
        fs_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        # Expand details of provided image
        self._fs, self._path = io.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        self.chunk_dims = chunk_dims
        self.clean_metadata = clean_metadata

        # Enforce valid image
        self._is_supported_image(self._fs, self._path, clean_metadata)

        # Get ome-types object and warn of other behaviors
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                # Get and store OME
                self._ome = self._get_ome(
                    tiff.pages[0].description, self.clean_metadata
                )

                # Get and store scenes
                self._scenes: Tuple[str, ...] = tuple(
                    image_meta.id for image_meta in self._ome.images
                )

                # Log a warning stating that if this is a MM OME-TIFF, don't read
                # many series
                if tiff.is_micromanager and not isinstance(self._fs, LocalFileSystem):
                    log.warning(
                        "**Remote reading** (S3, GCS, HTTPS, etc.) of multi-image "
                        "(or scene) OME-TIFFs created by MicroManager has limited "
                        "support with the scene API. "
                        "It is recommended to use independent AICSImage or Reader "
                        "objects for each remote file instead of the `set_scene` API. "
                        "Track progress on support here: "
                        "https://github.com/AllenCellModeling/aicsimageio/issues/196"
                    )

    @staticmethod
    def _expand_dims_to_match_ome(
        image_data: types.ArrayLike,
        ome: OME,
        dims: List[str],
        scene_index: int,
    ) -> types.ArrayLike:
        # Expand image_data for empty dimensions
        ome_shape = []

        # need to correct channel count if this is a RGB image
        n_samples = ome.images[scene_index].pixels.channels[0].samples_per_pixel
        has_multiple_samples = n_samples is not None and n_samples > 1
        for d in dims:
            # SizeC can represent RGB (Samples) data rather
            # than channel data, whether or not this is the case depends
            # on what the SamplesPerPixel are for the channel
            if d == "C" and has_multiple_samples:
                count = len(ome.images[scene_index].pixels.channels)
            elif d == "S" and has_multiple_samples:
                count = n_samples
            else:
                count = getattr(ome.images[scene_index].pixels, f"size_{d.lower()}")
            ome_shape.append(count)

        # The file may not have all the data but OME requires certain dimensions
        # expand to fill
        expand_dim_ops: List[Optional[slice]] = []
        for d_size in ome_shape:
            # Add empty dimension where OME requires dimension but no data exists
            if d_size == 1:
                expand_dim_ops.append(None)
            # Add full slice where data exists
            else:
                expand_dim_ops.append(slice(None, None, None))

        # Apply operators to dask array
        return image_data[tuple(expand_dim_ops)]

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        retrieve_indices: Tuple[Union[int, slice]],
        transpose_indices: List[int],
    ) -> np.ndarray:
        """
        Open a file for reading, construct a Zarr store, select data, and compute to
        numpy.
        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to file to read.
        scene: int
            The scene index to pull the chunk from.
        retrieve_indices: Tuple[Union[int, slice]]
            The image indices to retrieve.
        transpose_indices: List[int]
            The indices to transpose to prior to requesting data.
        Returns
        -------
        chunk: np.ndarray
            The image chunk as a numpy array.
        """
        with fs.open(path) as open_resource:
            with imread(
                open_resource,
                aszarr=True,
                series=scene,
                level=0,
                chunkmode="page",
                is_mmstack=False,
            ) as store:
                arr = da.from_zarr(store)
                arr = arr.transpose(transpose_indices)

                # By setting the compute call to always use a "synchronous" scheduler,
                # it informs Dask not to look for an existing scheduler / client
                # and instead simply read the data using the current thread / process.
                # In doing so, we shouldn't run into any worker data transfer and
                # handoff _during_ a read.
                return arr[retrieve_indices].compute(scheduler="synchronous")

    def _general_data_array_constructor(
        self,
        image_data: types.ArrayLike,
        dims: List[str],
        coords: Dict[str, Union[List[Any], types.ArrayLike]],
        tiff_tags: TiffTags,
    ) -> xr.DataArray:
        # Expand the image data to match the OME empty dimensions
        image_data = self._expand_dims_to_match_ome(
            image_data=image_data,
            ome=self._ome,
            dims=dims,
            scene_index=self.current_scene_index,
        )

        # Always order array
        if dimensions.DimensionNames.Samples in dims:
            out_order = dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES
        else:
            out_order = dimensions.DEFAULT_DIMENSION_ORDER

        # Transform into order
        image_data = transforms.reshape_data(
            image_data,
            "".join(dims),
            out_order,
        )

        # Reset dims after transform
        dims = [d for d in out_order]

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,
            attrs={
                constants.METADATA_UNPROCESSED: tiff_tags,
                constants.METADATA_PROCESSED: self._ome,
            },
        )

    def _read_delayed(self) -> xr.DataArray:
        """
        Construct the delayed xarray DataArray object for the image.
        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully delayed image as a DataArray object.
            Metadata is attached in some cases as coords, dims, and attrs contains
            unprocessed tags and processed OME object.
        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags(tiff)

                # Unpack coords from OME
                coords = get_coords_from_ome(
                    ome=self._ome,
                    scene_index=self.current_scene_index,
                )

                # Guess the dim order based on metadata and actual tiff data
                dims = Reader._guess_ome_dim_order(
                    tiff, self._ome, self.current_scene_index
                )

                # Grab the tifffile axes to use for dask array construction
                # If any of the non-"standard" dims are present
                # they will be filtered out during later reshape data calls
                strictly_read_dims = list(tiff.series[self.current_scene_index].axes)

                # Create the delayed dask array
                image_data = self._create_dask_array(tiff, strictly_read_dims)

                return self._general_data_array_constructor(
                    image_data,
                    dims,
                    coords,
                    tiff_tags,
                )

    def _read_immediate(self) -> xr.DataArray:
        """
        Construct the in-memory xarray DataArray object for the image.
        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully read into memory image as a DataArray
            object. Metadata is attached in some cases as coords, dims, and attrs
            contains unprocessed tags and processed OME object.
        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags(tiff)

                # Unpack coords from OME
                coords = get_coords_from_ome(
                    ome=self._ome,
                    scene_index=self.current_scene_index,
                )

                # Guess the dim order based on metadata and actual tiff data
                dims = Reader._guess_ome_dim_order(
                    tiff, self._ome, self.current_scene_index
                )

                # Read image into memory
                image_data = tiff.series[self.current_scene_index].asarray()

                return self._general_data_array_constructor(
                    image_data,
                    dims,
                    coords,
                    tiff_tags,
                )

    @property
    def scenes(self) -> Optional[Tuple[str, ...]]:
        return self._scenes

    @property
    def ome_metadata(self) -> OME:
        return self.metadata

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            Using available metadata, the floats representing physical pixel sizes for
            dimensions Z, Y, and X.
        Notes
        -----
        We currently do not handle unit attachment to these values. Please see the file
        metadata for unit information.
        """
        return physical_pixel_sizes(self.metadata, self.current_scene_index)

    @property
    def micromanager_metadata(self) -> Dict[str | int, Any]:
        """
        Returns
        -------
        micromanager_metadata: dict[str|int, Any]
        Expose the data from Adobe private tag 50839.
        Notes
        -----
        this is in response to a user request:
            https://github.com/bioio-devs/bioio-ome-tiff/issues/5
        """
        if self._micromanager_metadata is not None:
            return self._micromanager_metadata

        self._micromanager_metadata = {}
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                # Iterate over tiff tags
                tiff_tags = self._get_tiff_tags(tiff)
                for k, v in tiff_tags.items():
                    # break up key 50839 which is where MM metadata lives
                    # 50839 is a private tag registered with Adobe
                    if k == 50839:
                        try:
                            for kk, vv in json.loads(v["Info"]).items():
                                self._micromanager_metadata[kk] = vv
                        except Exception:
                            # if we can't parse the json, just ignore it
                            pass
        return self._micromanager_metadata

    def _get_tiff_tags(self, tiff: TiffFile, process: bool = True) -> TiffTags:
        unprocessed_tags = tiff.series[self.current_scene_index].pages[0].tags
        if not process:
            return unprocessed_tags

        # Create dict of tag and value
        tags: Dict[int, str] = {}
        for code, tag in unprocessed_tags.items():
            tags[code] = tag.value

        return tags

    def _create_dask_array(
        self, tiff: TiffFile, selected_scene_dims_list: List[str]
    ) -> da.Array:
        """
        Creates a delayed dask array for the file.
        Parameters
        ----------
        tiff: TiffFile
            An open TiffFile for processing.
        selected_scene_dims_list: List[str]
            The dimensions to use for constructing the array with.
            Required for managing chunked vs non-chunked dimensions.
        Returns
        -------
        image_data: da.Array
            The fully constructed and fully delayed image as a Dask Array object.
        """
        # Always add the plane dimensions if not present already
        for dim in dimensions.REQUIRED_CHUNK_DIMS:
            if dim not in self.chunk_dims:
                self.chunk_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_dims = [d.upper() for d in self.chunk_dims]

        # Construct delayed dask array
        selected_scene = tiff.series[self.current_scene_index]
        selected_scene_dims = "".join(selected_scene_dims_list)

        # Raise invalid dims error
        if len(selected_scene.shape) != len(selected_scene_dims):
            raise exceptions.ConflictingArgumentsError(
                f"Dimension string provided does not match the "
                f"number of dimensions found for this scene. "
                f"This scene shape: {selected_scene.shape}, "
                f"Provided dims string: {selected_scene_dims}"
            )

        # Constuct the chunk and non-chunk shapes one dim at a time
        # We also collect the chunk and non-chunk dimension order so that
        # we can swap the dimensions after we block out the array
        non_chunk_dim_order = []
        non_chunk_shape = []
        chunk_dim_order = []
        chunk_shape = []
        for dim, size in zip(selected_scene_dims, selected_scene.shape):
            if dim in self.chunk_dims:
                chunk_dim_order.append(dim)
                chunk_shape.append(size)
            else:
                non_chunk_dim_order.append(dim)
                non_chunk_shape.append(size)

        # Fill out the rest of the blocked shape with dimension sizes of 1 to
        # match the length of the sample chunk
        # When dask.block happens it fills the dimensions from inner-most to
        # outer-most with the chunks as long as the dimension is size 1
        blocked_dim_order = non_chunk_dim_order + chunk_dim_order
        blocked_shape = tuple(non_chunk_shape) + ((1,) * len(chunk_shape))

        # Construct the transpose indices that will be used to
        # transpose the array prior to pulling the chunk dims
        match_map = {dim: selected_scene_dims.find(dim) for dim in selected_scene_dims}
        transposer = []
        for dim in blocked_dim_order:
            transposer.append(match_map[dim])

        # Make ndarray for lazy arrays to fill
        lazy_arrays: np.ndarray = np.ndarray(blocked_shape, dtype=object)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            # All dimensions get their normal index except for chunk dims
            # which get filled with "full" slices
            indices_with_slices = np_index[: len(non_chunk_shape)] + (
                (slice(None, None, None),) * len(chunk_shape)
            )

            # Fill the numpy array with the delayed arrays
            lazy_arrays[np_index] = da.from_delayed(
                delayed(Reader._get_image_data)(
                    fs=self._fs,
                    path=self._path,
                    scene=self.current_scene_index,
                    retrieve_indices=indices_with_slices,
                    transpose_indices=transposer,
                ),
                shape=chunk_shape,
                dtype=selected_scene.dtype,
            )

        # Convert the numpy array of lazy readers into a dask array
        image_data = da.block(lazy_arrays.tolist())

        # Because we have set certain dimensions to be chunked and others not
        # we will need to transpose back to original dimension ordering
        # Example, if the original dimension ordering was "TZYX" and we
        # chunked by "T", "Y", and "X"
        # we created an array with dimensions ordering "ZTYX"
        transpose_indices = []
        for i, d in enumerate(selected_scene_dims):
            new_index = blocked_dim_order.index(d)
            if new_index != i:
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Transpose back to normal
        image_data = da.transpose(image_data, tuple(transpose_indices))

        return image_data
