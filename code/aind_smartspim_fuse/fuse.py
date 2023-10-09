"""
This file controls the fusion step
for a SmartSPIM dataset
"""

from pathlib import Path

import utilities

from .__init__ import __version__
from ._shared.types import PathLike


def terasticher(
    data_folder: PathLike,
    transforms_xml_path: PathLike,
    output_fused_path: PathLike,
    intermediate_fused_folder: PathLike,
    channel_regex=r"Ex_([0-9]*)_Em_([0-9]*)$",
):
    """
    This function fuses a SmartSPIM dataset.
    Currently, this module will do the following:

    1. Read the XML file generated with TeraStitcher
    that comes with the image transformations to
    reconstruct the volume.
    2. Fuse the dataset in chunks using TeraStitcher.
    3. Get the fused chunks and generate the OMEZarr
    volume.


    Parameters
    -----------
    data_folder: PathLike
        Path where the image data is located

    transforms_xml_path: PathLike
        Path where the XML with TeraStitcher
        format is located.

    output_fused_path: PathLike
        Path where the OMEZarr and metadata will
        live after fusion

    intermediate_fused_folder: PathLike
        Path where the intermediate files
        will live. These will not be in the final
        folder structure. e.g., 3D fused chunks
        from TeraStitcher
    """

    # Converting to path objects if necessary
    transforms_xml_path = Path(transforms_xml_path)
    output_fused_path = Path(output_fused_path)
    intermediate_fused_folder = Path(intermediate_fused_folder)

    if not output_fused_path.exists():
        raise FileNotFoundError(f"XML path {transforms_xml_path} does not exist")

    # Contains the paths where I'll place the
    # fused OMEZarr and TeraStitcher metadata
    # and fusion
    (
        fusion_folder,
        metadata_folder,
        teras_fusion_folder,
    ) = utilities.create_fusion_folder_structure(
        output_fused_path=output_fused_path,
        intermediate_fused_folder=intermediate_fused_folder,
    )
    logger.info(
        f"Output folders -> Fused image: {fusion_folder} -- Fusion metadata: {metadata_folder}"
    )

    # Looking for SmartSPIM channels on data folder
    smartspim_channels = utilities.find_smartspim_channels(
        path=data_folder, channel_regex=channel_regex
    )

    if not len(smartspim_channels):
        raise ValueError("No SmartSPIM channels found!")

    # Setting first found channel to reconstruct
    # This is intented to be compatible with CO pipelines
    # Therefore the channel must be in the data folder
    fuse_channel = smartspim_channels[0]

    # Logger pointing everything to the metadata path
    logger = utilities.create_logger(output_log_path=metadata_folder)

    logger.info(f"Generating derived data description")

    utilities.generate_data_description(
        raw_data_description_path=data_folder.joinpath("data_description.json"),
        dest_data_description=output_fused_path.joinpath("data_description.json"),
        process_name="stitched",
    )

    logger.info("Copying all available raw SmartSPIM metadata")

    # This is the AIND metadata
    utilities.copy_available_metadata(
        input_path=data_folder,
        output_path=output_fused_path,
        ignore_files=[
            "data_description.json",  # Ignoring data description since we're generating it above
            "processing.json",  # This is generated with all the steps
        ],
    )

    logger.info(f"Starting fusion for channel {fuse_channel}")

    teras_import_binary = ""

    channel_merge_xml_path = utilities.generate_new_channel_alignment_xml(
        informative_channel_xml=transforms_xml_path,
        channel_path=fuse_channel,
        metadata_folder=metadata_folder,
        teras_mdata_bin=teras_import_binary,
        encoding="utf-8",
        regex_expr=channel_regex,
    )

    merge_config = {
        "s": channel_merge_xml_path,
        "d": teras_fusion_folder,
        "sfmt": '"TIFF (unstitched, 3D)"',
        "dfmt": '"TIFF (tiled, 4D)"',
        "cpu_params": config["merge"]["cpu_params"],
        "width": config["merge"]["slice_extent"][0],
        "height": config["merge"]["slice_extent"][1],
        "depth": config["merge"]["slice_extent"][2],
        "additional_params": ["fixed_tiling"],
        "ch_dir": fuse_channel,
        # 'clist':'0'
    }
