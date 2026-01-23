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
from ome_types import OME
from pydantic import ValidationError
from tifffile.tifffile import TiffFile, TiffFileError, TiffTags, imread
from xmlschema import XMLSchemaValidationError
from xmlschema.exceptions import XMLSchemaValueError

from .companion import (
    remap_plane_axis_to_zct,
    resolve_ome_metadata_for_tiff,
)
from .utils import (
    expand_dims_to_match_ome,
    expand_missing_dims_to_match_target,
    get_coords_from_ome,
    get_dims_from_ome,
    get_ome,
    guess_ome_dim_order,
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

    # Companion-mode bookkeeping
    _using_companion_ome: bool = False
    _ome_image_index: int = 0
    _companion_path: Optional[str] = None

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem,
        path: str,
        clean_metadata: bool = True,
        companion_path: Optional[str] = None,
        **kwargs: Any,
    ) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    # Get first page description (aka the description tag in general)
                    # after Tifffile version 2023.3.15 mmstack images read all scenes
                    # into tiff.pages[0]
                    xml = tiff.pages[0].description
                    ome = get_ome(xml, clean_metadata)

                    # Handle no images in metadata
                    # this commonly means it is a "BinaryData" OME file
                    # i.e. a non-main OME-TIFF from MicroManager or similar
                    # in this case, because it's not the main file we want to just role
                    # back to TiffReader
                    if ome.binary_only and companion_path is None:
                        raise exceptions.UnsupportedFileFormatError(
                            "bioio-ome-tiff",
                            path,
                            (
                                "Binary-only embedded OME. Provide a companion OME "
                                "file to read this dataset."
                            ),
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

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = dimensions.DEFAULT_CHUNK_DIMS,
        clean_metadata: bool = True,
        companion_path: Optional[types.PathLike] = None,
        fs_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        # Expand details of provided image
        self._fs, self._path = io.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        self._companion_path = (
            str(companion_path) if companion_path is not None else None
        )

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)
        self.chunk_dims = chunk_dims
        self.clean_metadata = clean_metadata

        # Enforce valid image
        self._is_supported_image(
            self._fs,
            self._path,
            clean_metadata,
            companion_path=self._companion_path,
        )

        # Get ome-types object and warn of other behaviors
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource, is_mmstack=False) as tiff:
                res = resolve_ome_metadata_for_tiff(
                    tiff=tiff,
                    fs=self._fs,
                    tiff_path=self._path,
                    companion_path=self._companion_path,
                    clean_metadata=self.clean_metadata,
                    get_ome_fn=get_ome,
                )

                self._ome = res.ome
                self._using_companion_ome = res.using_companion_ome
                self._ome_image_index = res.ome_image_index

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
        # Only expand the image data if the data is actually missing axes.
        if image_data.ndim < len(dims):
            image_data = expand_dims_to_match_ome(
                image_data=image_data,
                ome=self._ome,
                dims=dims,
                scene_index=self.ome_scene_index,
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
                    ome=self._ome, scene_index=self.ome_scene_index
                )

                # Guess the dim order based on metadata and actual tiff data
                if self._using_companion_ome:
                    dims = get_dims_from_ome(self._ome, self.ome_scene_index)
                else:
                    dims = guess_ome_dim_order(tiff, self._ome, self.ome_scene_index)

                # Grab the tifffile axes to use for dask array construction
                # If any of the non-"standard" dims are present
                # they will be filtered out during later reshape data calls
                strictly_read_dims = list(tiff.series[self.current_scene_index].axes)
                # Create the delayed dask array
                image_data = self._create_dask_array(tiff, strictly_read_dims)

                # If tifffile collapsed planes into 'I' and OME has <TiffData>,
                # remap exactly.
                try:
                    image_data, strictly_read_dims = remap_plane_axis_to_zct(
                        tiff=tiff,
                        tiff_scene_index=self.current_scene_index,
                        image_data=image_data,
                        tiff_axes=strictly_read_dims,
                        ome=self._ome,
                        ome_image_index=self.ome_scene_index,
                    )
                except exceptions.UnsupportedFileFormatError:
                    pass

                # If OME includes singleton dims that tifffile omitted,
                # expand ONLY those.
                if image_data.ndim < len(dims) or any(
                    d not in strictly_read_dims for d in dims
                ):
                    image_data, strictly_read_dims = (
                        expand_missing_dims_to_match_target(
                            image_data=image_data,
                            current_dims=strictly_read_dims,
                            target_dims=dims,
                            ome=self._ome,
                            scene_index=self.ome_scene_index,
                        )
                    )

                # After remap/expansion, dims describing data are strictly_read_dims
                dims = strictly_read_dims

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
                    ome=self._ome, scene_index=self.ome_scene_index
                )

                if self._using_companion_ome:
                    dims = get_dims_from_ome(self._ome, self.ome_scene_index)
                else:
                    # Guess the dim order based on metadata and actual tiff data
                    dims = guess_ome_dim_order(tiff, self._ome, self.ome_scene_index)

                strictly_read_dims = list(tiff.series[self.current_scene_index].axes)
                # Read image into memory
                image_data = tiff.series[self.current_scene_index].asarray()

                # If tifffile collapsed planes into 'I' and OME has <TiffData>,
                # remap exactly.
                try:
                    image_data, strictly_read_dims = remap_plane_axis_to_zct(
                        tiff=tiff,
                        tiff_scene_index=self.current_scene_index,
                        image_data=image_data,
                        tiff_axes=strictly_read_dims,
                        ome=self._ome,
                        ome_image_index=self.ome_scene_index,
                    )
                except exceptions.UnsupportedFileFormatError:
                    pass

                if image_data.ndim < len(dims) or any(
                    d not in strictly_read_dims for d in dims
                ):
                    image_data, strictly_read_dims = (
                        expand_missing_dims_to_match_target(
                            image_data=image_data,
                            current_dims=strictly_read_dims,
                            target_dims=dims,
                            ome=self._ome,
                            scene_index=self.ome_scene_index,
                        )
                    )

                dims = strictly_read_dims

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
    def ome_scene_index(self) -> int:
        """
        The scene index into OME <Image> list.

         - Normal mode: current_scene_index is the OME image index
         - Companion mode: OME <Image> is selected by companion resolver
        """
        if self._using_companion_ome:
            return self._ome_image_index
        return self.current_scene_index

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
        return physical_pixel_sizes(self.metadata, self.ome_scene_index)

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
