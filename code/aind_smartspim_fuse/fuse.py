"""
This file controls the fusion step
for a SmartSPIM dataset
"""

import logging
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from aind_data_schema.core.processing import DataProcess, ProcessName

from .__init__ import __version__
from ._shared.types import PathLike
from .utils import utils
from .zarr_writer import smartspim_zarr_writer as spim_zarr


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
    input_path: PathLike
        Path where the input data is located

    xml_output_path: PathLike
        Path where the import XML will be saved

    import_params: dict
        Configuration dictionary used to build the
        terastitcher's import command.

    channel_name:str
        Name of the dataset channel that will be imported

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


def build_parallel_command(params: dict, tool: PathLike) -> str:
    """
    Builds a mpi command based on a provided configuration dictionary.

    Parameters
    ------------------------
    params: dict
        Configuration dictionary used to build
        the mpi command depending on the platform.

    tool: PathLike
        Parallel tool to be used in the command.
        (Parastitcher or Paraconverter)

    Returns
    ------------------------
    str:
        Command that will be executed for terastitcher.

    """

    cpu_params = params["cpu_params"]

    # mpiexec for windows, mpirun for linux or macs OS
    mpi_command = "mpirun -np"
    additional_params = ""
    hostfile = ""
    n_procs = cpu_params["number_processes"]

    # Additional params provided in the configuration
    if len(cpu_params["additional_params"]):
        additional_params = utils.helper_additional_params_command(
            cpu_params["additional_params"]
        )

    hostfile = f"--hostfile {cpu_params['hostfile']}"

    cmd = f"{mpi_command} {n_procs} {hostfile} {additional_params}"
    cmd += f"python {tool}"
    return cmd


def terastitcher_merge_cmd(
    xml_output_path: PathLike,
    merge_params: dict,
    channel_name: str,
    paraconverter_path: PathLike,
) -> str:
    """
    Builds the terastitcher's multivolume merge command based
    on a provided configuration dictionary. It outputs a json
    file in the xmls folder of the output directory with all
    the parameters used in this step. It is important to
    mention that the channels are fuse separately.

    Parameters
    ------------------------
    xml_output_path: PathLike
        Path where the displacements XML
        is located

    merge_params: dict
        Configuration dictionary used to build the
        terastitcher's multivolume merge command.

    channel_name: str
        string with the channel to generate the
        command

    paraconverter_path: PathLike
        Path where para converter is located

    Returns
    ------------------------
    str:
        Command that will be executed for terastitcher.

    """

    paraconverter_path = str(paraconverter_path)
    parallel_command = build_parallel_command(merge_params, paraconverter_path)

    parameters = utils.helper_build_param_value_command(merge_params)

    additional_params = ""
    if len(merge_params["additional_params"]):
        additional_params = utils.helper_additional_params_command(
            merge_params["additional_params"]
        )

    cmd = f"{parallel_command} {parameters} {additional_params}"  # > {self.xmls_path}/step6par_{channel}.txt"
    cmd = cmd.replace("--s=", "-s=")
    cmd = cmd.replace("--d=", "-d=")

    output_json = xml_output_path.joinpath(f"merge_volume_params_{channel_name}.json")
    utils.save_dict_as_json(f"{output_json}", merge_params, True)

    return cmd


def terasticher_fusion(
    data_folder: PathLike,
    transforms_xml_path: PathLike,
    metadata_folder: PathLike,
    teras_fusion_folder: PathLike,
    channel_name: str,
    smartspim_config: dict,
    logger: logging.Logger,
    channel_regex: Optional[str] = r"Ex_([0-9]*)_Em_([0-9]*)$",
    code_url: Optional[
        str
    ] = "https://github.com/AllenNeuralDynamics/aind-smartspim-stitch",
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

    logger: logging.Logger
        Logger object

    channel_regex: Optional[str]
        Regular expression to identify
        smartspim channels

    code_url: Optional[str]
        Github repository where this code is
        hosted to include in the metadata

    Returns
    ----------
    Tuple[PathLike, List[DataProcess]]:
        Tuple with the path where the fused data
        was stored and the AIND data processes
        from the schema
    """
    data_processes = []

    # parastitcher_path = Path(smartspim_config["pyscripts_path"]).joinpath(
    #     "Parastitcher.py"
    # )
    paraconverter_path = Path(smartspim_config["pyscripts_path"]).joinpath(
        "paraconverter.py"
    )

    channel_path = data_folder.joinpath(channel_name)

    if not channel_path.exists():
        raise FileExistsError(f"Path {channel_path} does not exist!")

    logger.info(f"Starting importing for channel {channel_name}")

    teras_import_channel_cmd, teras_import_binary = terastitcher_import_cmd(
        input_path=channel_path,
        xml_output_path=metadata_folder,
        import_params=smartspim_config["import_data"],
        channel_name=channel_name,
    )

    logger.info(f"TeraStitcher import binary located at: {teras_import_binary}")
    logger.info(f"Executing TeraStitcher command: {teras_import_channel_cmd}")

    # Importing channel to generate binary file
    import_start_time = datetime.now()
    utils.execute_command(command=teras_import_channel_cmd, logger=logger, verbose=True)
    import_end_time = datetime.now()

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_IMPORTING,
            software_version="1.11.10",
            start_date_time=import_start_time,
            end_date_time=import_end_time,
            input_location=str(channel_path),
            output_location=str(metadata_folder),
            outputs={
                "output_file": str(
                    metadata_folder.joinpath(f"xml_import_{channel_name}.xml")
                )
            },
            code_url=code_url,
            code_version=__version__,
            parameters=smartspim_config["import_data"],
            notes=f"TeraStitcher image import for channel {channel_name}",
        )
    )

    # Generating new displacements file based on the informative channel
    channel_merge_xml_path = utils.generate_new_channel_alignment_xml(
        informative_channel_xml=transforms_xml_path,
        channel_path=data_folder.joinpath(channel_name),
        metadata_folder=metadata_folder,
        teras_mdata_bin=teras_import_binary,
        encoding="utf-8",
        channel_regex=channel_regex,
    )

    logger.info(
        f"New alignment file in path {channel_merge_xml_path} based from {transforms_xml_path}"
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
        # "mdata_fname": teras_import_binary
        # 'clist':'0'
    }

    # Merging dataset
    logger.info(f"Starting fusion for channel {channel_name}")

    teras_merge_channel_cmd = terastitcher_merge_cmd(
        xml_output_path=metadata_folder,
        merge_params=terastitcher_merge_config,
        channel_name=channel_name,
        paraconverter_path=paraconverter_path,
    )

    logger.info(f"Executing TeraStitcher command: {teras_merge_channel_cmd}")

    # Merge channel with TeraStitcher
    merge_start_time = datetime.now()
    utils.execute_command(command=teras_merge_channel_cmd, logger=logger, verbose=True)
    merge_end_time = datetime.now()

    # Getting new top level folder after fusion
    teras_fusion_folder = [
        x for x in teras_fusion_folder.iterdir() if x.is_dir() and "RES" in str(x)
    ][0]

    data_processes.append(
        DataProcess(
            name=ProcessName.IMAGE_TILE_FUSING,
            software_version="1.11.10",
            start_date_time=merge_start_time,
            end_date_time=merge_end_time,
            input_location=str(channel_merge_xml_path),
            output_location=str(metadata_folder),
            outputs={"output_folder": str(teras_fusion_folder)},
            code_url=code_url,
            code_version=__version__,
            parameters=terastitcher_merge_config,
            notes=f"TeraStitcher image fusion for channel {channel_name}",
        )
    )

    return teras_fusion_folder, data_processes


def main(
    data_folder: PathLike,
    transforms_xml_path: PathLike,
    output_fused_path: PathLike,
    intermediate_fused_folder: PathLike,
    smartspim_config: dict,
    channel_regex: Optional[str] = r"Ex_([0-9]*)_Em_([0-9]*)$",
):
    """
    This function fuses a SmartSPIM dataset.

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
    data_folder = Path(data_folder)
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
        raise ValueError(f"No SmartSPIM channels found in path: {data_folder}")

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

    # Logger pointing everything to the metadata path
    logger = utils.create_logger(output_log_path=metadata_folder)
    utils.print_system_information(logger)

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    logger.info(f"{'='*40} SmartSPIM Stitching {'='*40}")
    logger.info(f"Output folders - Stitch metadata: {metadata_folder}")

    logger.info(f"{'='*40} SmartSPIM Fusion {'='*40}")

    logger.info(
        f"Output folders -> Fused image: {fusion_folder} -- Fusion metadata: {metadata_folder}"
    )

    terastitcher_fused_path, data_processes = terasticher_fusion(
        data_folder=data_folder,
        transforms_xml_path=transforms_xml_path,
        metadata_folder=metadata_folder,
        teras_fusion_folder=teras_fusion_folder,
        channel_name=channel_name,
        smartspim_config=smartspim_config,
        logger=logger,
        channel_regex=channel_regex,
    )

    logger.info(f"Fused dataset with TeraStitcher in path: {terastitcher_fused_path}")

    logger.info(f"Starting OMEZarr conversion in path: {output_fused_path}")

    voxel_size = [  # ZYX order
        smartspim_config["import_data"]["vxl3"],  # Z
        smartspim_config["import_data"]["vxl2"],  # Y
        smartspim_config["import_data"]["vxl1"],  # X
    ]
    zarr_chunksize = [128, 128, 128]

    (
        file_convert_start_time,
        file_convert_end_time,
    ) = spim_zarr.write_zarr_from_terastitcher(
        input_path=terastitcher_fused_path,
        output_path=fusion_folder,
        voxel_size=voxel_size,
        final_chunksize=zarr_chunksize,
        scale_factor=smartspim_config["ome_zarr_params"]["scale_factor"],
        codec=smartspim_config["ome_zarr_params"]["codec"],
        compression_level=smartspim_config["ome_zarr_params"]["clevel"],
        n_lvls=smartspim_config["ome_zarr_params"]["pyramid_levels"],
        logger=logger,
    )

    data_processes.append(
        DataProcess(
            name=ProcessName.FILE_CONVERSION,
            software_version=__version__,
            start_date_time=file_convert_start_time,
            end_date_time=file_convert_end_time,
            input_location=str(terastitcher_fused_path),
            output_location=str(fusion_folder),
            outputs={
                "output_file": str(fusion_folder.joinpath(f"{channel_name}.zarr"))
            },
            code_url="https://github.com/AllenNeuralDynamics/aind-smartspim-fuse",
            code_version=__version__,
            parameters={
                "ome_zarr_params": smartspim_config["ome_zarr_params"],
                "voxel_size": voxel_size,
                "ome_zarr_chunksize": zarr_chunksize,
            },
            notes=f"File format conversion from .tiff to OMEZarr for channel {channel_name}",
        )
    )

    utils.generate_processing(
        data_processes=data_processes,
        dest_processing=metadata_folder,
        processor_full_name="Camilo Laiton",
        pipeline_version="1.5.0",
    )

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            metadata_folder,
            "smartspim_fusion",
        )
