"""
This file controls the fusion step
for a SmartSPIM dataset
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from .__init__ import __version__
from ._shared.types import PathLike
from .utils import utils


def terastitcher_import_cmd(
    input_path: PathLike,
    xml_output_path: PathLike,
    import_params: dict,
    channel_name: str,
) -> str:
    """
    Builds the terastitcher's import command based on
    a provided configuration dictionary. It outputs
    a json file in the xmls folder of the output
    directory with all the parameters
    used in this step.

    Parameters
    ------------------------
    import_params: dict
        Configuration dictionary used to build the
        terastitcher's import command.

    channel_name:str
        Name of the dataset channel that will be imported

    fuse_path:PathLike
        Path where the fused xml files will be stored.
        This will only be used in multichannel fusing.
        Default None

    Returns
    ------------------------
    Tuple[str, str]:
        Command that will be executed for terastitcher and
        the TeraStitcher import binary
    """

    xml_output_path = Path(xml_output_path)

    volume_input = f"--volin={input_path}"

    output_path = xml_output_path.joinpath(f"xml_import_{channel_name}.xml")

    import_params["mdata_bin"] = str(
        xml_output_path.joinpath(f"mdata_{channel_name}.bin")
    )

    output_folder = f"--projout={output_path}"

    parameters = utils.helper_build_param_value_command(import_params)

    additional_params = ""
    if len(import_params["additional_params"]):
        additional_params = utils.helper_additional_params_command(
            import_params["additional_params"]
        )

    cmd = f"terastitcher --import {volume_input} {output_folder} {parameters} {additional_params}"

    output_json = xml_output_path.joinpath(f"import_params_{channel_name}.json")
    utils.save_dict_as_json(f"{output_json}", import_params, True)

    return cmd, import_params["mdata_bin"]


def terasticher(
    data_folder: PathLike,
    transforms_xml_path: PathLike,
    output_fused_path: PathLike,
    intermediate_fused_folder: PathLike,
    smartspim_config: dict,
    channel_regex: Optional[str] = r"Ex_([0-9]*)_Em_([0-9]*)$",
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

    smartspim_config: dict
        Dictionary with the smartspim configuration
        for that dataset

    channel_regex: Optional[str]
        Regular expression to identify
        smartspim channels
    """

    # Converting to path objects if necessary
    transforms_xml_path = Path(transforms_xml_path)
    output_fused_path = Path(output_fused_path)
    intermediate_fused_folder = Path(intermediate_fused_folder)

    if not output_fused_path.exists():
        raise FileNotFoundError(f"XML path {transforms_xml_path} does not exist")

    # Looking for SmartSPIM channels on data folder
    smartspim_channels = utils.find_smartspim_channels(
        path=data_folder, channel_regex=channel_regex
    )

    if not len(smartspim_channels):
        raise ValueError("No SmartSPIM channels found!")

    # Setting first found channel to reconstruct
    # This is intented to be compatible with CO pipelines
    # Therefore the channel must be in the data folder
    channel_name = smartspim_channels[0]

    # Contains the paths where I'll place the
    # fused OMEZarr and TeraStitcher metadata
    # and fusion
    (
        fusion_folder,
        metadata_folder,
        teras_fusion_folder,
    ) = utils.create_fusion_folder_structure(
        output_fused_path=output_fused_path,
        intermediate_fused_folder=intermediate_fused_folder,
        channel_name=channel_name,
    )
    logger.info(
        f"Output folders -> Fused image: {fusion_folder} -- Fusion metadata: {metadata_folder}"
    )

    # Logger pointing everything to the metadata path
    logger = utils.create_logger(output_log_path=metadata_folder)

    logger.info(f"Generating derived data description")

    utils.generate_data_description(
        raw_data_description_path=data_folder.joinpath("data_description.json"),
        dest_data_description=output_fused_path.joinpath("data_description.json"),
        process_name="stitched",
    )

    logger.info("Copying all available raw SmartSPIM metadata")

    # This is the AIND metadata
    utils.copy_available_metadata(
        input_path=data_folder,
        output_path=output_fused_path,
        ignore_files=[
            "data_description.json",  # Ignoring data description since we're generating it above
            "processing.json",  # This is generated with all the steps
        ],
    )

    logger.info(f"Starting fusion for channel {channel_name}")

    teras_import_channel_cmd, teras_import_binary = terastitcher_import_cmd(
        input_path=transforms_xml_path,
        xml_output_path=metadata_folder,
        import_params=smartspim_config["import_data"],
        channel_name=channel_name,
    )
    logger.info(f"Executing TeraStitcher command: {terastitcher_import_cmd}")

    # Importing channel to generate binary file
    import_start_time = datetime.now()
    utils.execute_command(command=teras_import_channel_cmd, logger=logger, verbose=True)
    import_end_time = datetime.now()

    # Generating new displacements file based on the informative channel
    channel_merge_xml_path = utils.generate_new_channel_alignment_xml(
        informative_channel_xml=transforms_xml_path,
        channel_path=channel_name,
        metadata_folder=metadata_folder,
        teras_mdata_bin=teras_import_binary,
        encoding="utf-8",
        regex_expr=channel_regex,
    )

    # Merge configuration
    terastitcher_merge_config = {
        "s": channel_merge_xml_path,
        "d": teras_fusion_folder,
        "sfmt": '"TIFF (unstitched, 3D)"',
        "dfmt": '"TIFF (tiled, 4D)"',
        "cpu_params": smartspim_config["merge"]["cpu_params"],
        "width": smartspim_config["merge"]["slice_extent"][0],
        "height": smartspim_config["merge"]["slice_extent"][1],
        "depth": smartspim_config["merge"]["slice_extent"][2],
        "additional_params": ["fixed_tiling"],
        "ch_dir": channel_name,
        # 'clist':'0'
    }

    # Merging dataset
