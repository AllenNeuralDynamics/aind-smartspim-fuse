"""
Computes the multiscales of a zarr array
"""

import time
from typing import Dict, List, Optional, Tuple, cast

import dask.array as da
import numpy as np
import xarray_multiscale
import zarr
from blocked_zarr_writer import BlockedArrayWriter
from dask.distributed import Client, LocalCluster
from numcodecs import blosc
from ome_zarr.format import CurrentFormat
from ome_zarr.writer import write_multiscales_metadata


def compute_pyramid(
    data,
    n_lvls,
    scale_axis,
    chunks="auto",
):
    """
    Computes the pyramid levels given an input full resolution image data

    Parameters
    ------------------------

    data: dask.array.core.Array
        Dask array of the image data

    n_lvls: int
        Number of downsampling levels
        that will be applied to the original image

    scale_axis: Tuple[int]
        Scaling applied to each axis

    chunks: Union[str, Sequence[int], Dict[Hashable, int]]
        chunksize that will be applied to the multiscales
        Default: "auto"

    Returns
    ------------------------

    Tuple[List[dask.array.core.Array], Dict]:
        List with the downsampled image(s) and dictionary
        with image metadata
    """

    pyramid = xarray_multiscale.multiscale(
        array=data,
        reduction=xarray_multiscale.reducers.windowed_mean,  # func
        scale_factors=scale_axis,  # scale factors
        preserve_dtype=True,
        chunks=chunks,
    )[:n_lvls]

    return [pyramid_level.data for pyramid_level in pyramid]


def _compute_scales(
    scale_num_levels: int,
    scale_factor: Tuple[float, float, float],
    pixelsizes: Tuple[float, float, float],
    chunks: Tuple[int, int, int, int, int],
    data_shape: Tuple[int, int, int, int, int],
    translation: Optional[List[float]] = None,
) -> Tuple[List, List]:
    """
    Generate the list of coordinate transformations
    and associated chunk options.

    Parameters
    ----------
    scale_num_levels: the number of downsampling levels
    scale_factor: a tuple of scale factors in each spatial dimension (Z, Y, X)
    pixelsizes: a list of pixel sizes in each spatial dimension (Z, Y, X)
    chunks: a 5D tuple of integers with size of each
    chunk dimension (T, C, Z, Y, X)
    data_shape: a 5D tuple of the full resolution image's shape
    translation: a 5 element list specifying the offset
    in physical units in each dimension

    Returns
    -------
    A tuple of the coordinate transforms and chunk options
    """
    transforms = [
        [
            # the voxel size for the first scale level
            {
                "type": "scale",
                "scale": [
                    1.0,
                    1.0,
                    pixelsizes[0],
                    pixelsizes[1],
                    pixelsizes[2],
                ],
            }
        ]
    ]
    if translation is not None:
        transforms[0].append({"type": "translation", "translation": translation})
    chunk_sizes = []
    lastz = data_shape[2]
    lasty = data_shape[3]
    lastx = data_shape[4]
    opts = dict(
        chunks=(
            1,
            1,
            min(lastz, chunks[2]),
            min(lasty, chunks[3]),
            min(lastx, chunks[4]),
        )
    )
    chunk_sizes.append(opts)
    if scale_num_levels > 1:
        for i in range(scale_num_levels - 1):
            last_transform = transforms[-1][0]
            last_scale = cast(List, last_transform["scale"])
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            1.0,
                            last_scale[2] * scale_factor[0],
                            last_scale[3] * scale_factor[1],
                            last_scale[4] * scale_factor[2],
                        ],
                    }
                ]
            )
            if translation is not None:
                transforms[-1].append(
                    {"type": "translation", "translation": translation}
                )
            lastz = int(np.ceil(lastz / scale_factor[0]))
            lasty = int(np.ceil(lasty / scale_factor[1]))
            lastx = int(np.ceil(lastx / scale_factor[2]))
            opts = dict(
                chunks=(
                    1,
                    1,
                    min(lastz, chunks[2]),
                    min(lasty, chunks[3]),
                    min(lastx, chunks[4]),
                )
            )
            chunk_sizes.append(opts)

    return transforms, chunk_sizes


def _get_axes_5d(
    time_unit: str = "millisecond", space_unit: str = "micrometer"
) -> List[Dict]:
    """Generate the list of axes.

    Parameters
    ----------
    time_unit: the time unit string, e.g., "millisecond"
    space_unit: the space unit string, e.g., "micrometer"

    Returns
    -------
    A list of dictionaries for each axis
    """
    axes_5d = [
        {"name": "t", "type": "time", "unit": f"{time_unit}"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": f"{space_unit}"},
        {"name": "y", "type": "space", "unit": f"{space_unit}"},
        {"name": "x", "type": "space", "unit": f"{space_unit}"},
    ]
    return axes_5d


def _build_ome(
    data_shape: Tuple[int, ...],
    image_name: str,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[int]] = None,
    channel_minmax: Optional[List[Tuple[float, float]]] = None,
    channel_startend: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Create the necessary metadata for an OME tiff image

    Parameters
    ----------
    data_shape: A 5-d tuple, assumed to be TCZYX order
    image_name: The name of the image
    channel_names: The names for each channel
    channel_colors: List of all channel colors
    channel_minmax: List of all (min, max) pairs of channel pixel
    ranges (min value of darkest pixel, max value of brightest)
    channel_startend: List of all pairs for rendering where start is
    a pixel value of darkness and end where a pixel value is
    saturated

    Returns
    -------
    Dict: An "omero" metadata object suitable for writing to ome-zarr
    """
    if channel_names is None:
        channel_names = [f"Channel:{image_name}:{i}" for i in range(data_shape[1])]
    if channel_colors is None:
        channel_colors = [i for i in range(data_shape[1])]
    if channel_minmax is None:
        channel_minmax = [(0.0, 1.0) for _ in range(data_shape[1])]
    if channel_startend is None:
        channel_startend = channel_minmax

    ch = []
    for i in range(data_shape[1]):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": channel_names[i],
                "window": {
                    "end": float(channel_startend[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_startend[i][0]),
                },
            }
        )

    omero = {
        "id": 1,  # ID in OMERO
        "name": image_name,  # Name as shown in the UI
        "version": "0.4",  # Current version
        "channels": ch,
        "rdefs": {
            "defaultT": 0,  # First timepoint to show the user
            "defaultZ": data_shape[2] // 2,  # First Z section to show the user
            "model": "color",  # "color" or "greyscale"
        },
    }
    return omero


def write_ome_ngff_metadata(
    group: zarr.Group,
    arr: da.Array,
    image_name: str,
    n_lvls: int,
    scale_factors: tuple,
    voxel_size: tuple,
    channel_names: List[str] = None,
    channel_colors: List[str] = None,
    channel_minmax: List[float] = None,
    channel_startend: List[float] = None,
    metadata: dict = None,
):
    """
    Write OME-NGFF metadata to a Zarr group.

    Parameters
    ----------
    group : zarr.Group
        The output Zarr group.
    arr : array-like
        The input array.
    image_name : str
        The name of the image.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : tuple
        The scale factors for downsampling along each dimension.
    voxel_size : tuple
        The voxel size along each dimension.
    channel_names: List[str]
        List of channel names to add to the OMENGFF metadata
    channel_colors: List[str]
        List of channel colors to visualize the data
    chanel_minmax: List[float]
        List of channel min and max values based on the
        image dtype
    channel_startend: List[float]
        List of the channel start and end metadata. This is
        used for visualization. The start and end range might be
        different from the min max and it is usually inside the
        range
    metadata: dict
        Extra metadata to write in the OME-NGFF metadata
    """
    print("WRITING METADATA")
    if metadata is None:
        metadata = {}
    fmt = CurrentFormat()

    # Building the OMERO metadata
    ome_json = _build_ome(
        arr.shape,
        image_name,
        channel_names=channel_names,
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
    )
    group.attrs["omero"] = ome_json
    axes_5d = _get_axes_5d()
    coordinate_transformations, chunk_opts = _compute_scales(
        n_lvls, scale_factors, voxel_size, arr.chunksize, arr.shape, None
    )
    fmt.validate_coordinate_transformations(
        arr.ndim, n_lvls, coordinate_transformations
    )
    # Setting coordinate transfomations
    datasets = [{"path": str(i)} for i in range(n_lvls)]
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    # Writing the multiscale metadata
    write_multiscales_metadata(group, datasets, fmt, axes_5d, **metadata)


def compute_multiscale(
    orig_lazy_data,
    zarr_group,
    scale_factor: List[int],
    n_workers: int,
    voxel_size: List[float],
    image_name: str,
    n_levels: Optional[int] = 3,
    threads_per_worker: Optional[int] = 1,
):
    """
    Computes the multiscales of a zarr dataset.

    Parameters
    ----------
    orig_lazy_data: dask.array
        Lazy read data

    zarr_group: Zarr.Group
        Zarr group where the multiscales will be written

    scale_factor: List[int]
        List of integers for the different multiscales

    n_workers: int
        Number of workers that will be used to write the
        multiple scales

    voxel_size: List[float]
        Voxel size expected to be in ZYX order

    image_name: str
        Image name

    n_levels: Optional[int]
        Number of multiple scales. Default: 3

    threads_per_worker: Optional[int]
        Number of threads per worker. Default 1
    """

    # Instantiating local cluster for parallel writing
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)
    #     performance_report_path = f"/results/report.html"

    start_time = time.time()

    previous_scale = orig_lazy_data

    written_pyramid = []

    if np.issubdtype(previous_scale.dtype, np.integer):
        np_info_func = np.iinfo

    else:
        # Floating point
        np_info_func = np.finfo

    # Getting min max metadata for the dtype
    channel_minmax = [
        (
            np_info_func(np.uint16).min,
            np_info_func(np.uint16).max,
        )
        for _ in range(previous_scale.shape[1])
    ]

    # Setting values for SmartSPIM
    # Ideally we would use da.percentile(image_data, (0.1, 95))
    # However, it would take so much time and resources and it is
    # not used that much on neuroglancer
    channel_startend = [(0.0, 350.0) for _ in range(previous_scale.shape[1])]

    # Writing OME-NGFF metadata
    write_ome_ngff_metadata(
        group=zarr_group,
        arr=previous_scale,
        image_name=image_name,
        n_lvls=n_levels,
        scale_factors=scale_factor,
        voxel_size=voxel_size,
        channel_names=[image_name],
        channel_colors=[0x690AFE],
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
        metadata=None,
    )

    # Writing zarr and performance report
    #     with performance_report(filename=performance_report_path):
    for i in range(1, n_levels):
        print(f"Writing multiscale: {i} in path {zarr_group}")
        if i != 1:
            previous_scale = da.from_zarr(pyramid_group, orig_lazy_data.chunks)

        # Writing zarr
        block_shape = list(
            BlockedArrayWriter.get_block_shape(
                arr=previous_scale, target_size_mb=12800  # 51200,
            )
        )

        # Formatting to 5D block shape
        block_shape = ([1] * (5 - len(block_shape))) + block_shape

        new_scale_factor = (
            [1] * (len(previous_scale.shape) - len(scale_factor))
        ) + scale_factor

        previous_scale_pyramid = compute_pyramid(
            data=previous_scale,
            scale_axis=new_scale_factor,
            chunks=(1, 1, 128, 128, 128),
            n_lvls=2,
        )
        array_to_write = previous_scale_pyramid[-1]

        # Create the scale dataset
        pyramid_group = zarr_group.create_dataset(
            name=i,
            shape=array_to_write.shape,
            chunks=array_to_write.chunksize,
            dtype=np.uint16,
            compressor=blosc.Blosc(cname="zstd", clevel=3, shuffle=blosc.SHUFFLE),
            dimension_separator="/",
            overwrite=True,
        )

        # Block Zarr Writer
        BlockedArrayWriter.store(array_to_write, pyramid_group, block_shape)
        written_pyramid.append(array_to_write)

    end_time = time.time()
    print(f"Time to write the dataset: {end_time - start_time}")
    print(f"Written pyramid: {written_pyramid}")

    client.shutdown()


def main():
    """
    Usage example
    """
    dataset_path = "../scratch/Ex_488_Em_525.ome.zarr"
    scale_factor = [2, 2, 2]
    dataset_name = "Ex_488_Em_525.ome.zarr"

    store = zarr.DirectoryStore(dataset_path)
    zarr_group = zarr.open(store, mode="a")

    lazy_data = da.from_zarr(f"{dataset_path}/0")

    n_workers = 16
    voxel_size = [
        2.0,
        1.8,
        1.8,
    ]
    n_levels = 3
    threads_per_worker = 1

    compute_multiscale(
        orig_lazy_data=lazy_data,
        zarr_group=zarr_group,
        scale_factor=scale_factor,
        n_workers=n_workers,
        voxel_size=voxel_size,
        image_name=dataset_name,
        n_levels=n_levels,
        threads_per_worker=threads_per_worker,
    )


if __name__ == "__main__":
    main()
