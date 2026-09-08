"""
Module for bigstitcher fusion. It assumes that the input
is a bigstitcher.xml with the transforms that need to be
applied for each of the stacks.

Codebase intended for GPU/CPU device.
No fallback to CPU written until required.

This fusion worker expects:
- preprocessed data directory of zarrs to fuse.
- complementary bigstitcher.xml
- named xml of the following format: SmartSPIM_dataset_num_datetime_stitching_channel_channel_info
  This information informs the output location of the multiscaled zarr.
"""

import json
import multiprocessing as mp
import os
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.io as io
import dask.array as da
import torch
import yaml
import zarr
from aind_data_schema.components.identifiers import Code
from aind_data_schema.core.processing import (DataProcess, ProcessName,
                                              ProcessStage)

from . import (__maintainers__, __pipeline_name__, __pipeline_version__,
               __title__, __url__, __version__)
from .utils import metadata_compat, utils
from .zarr_writer.create_multiscales import compute_multiscale


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.
    Parameters
    ------------------------
    filepath: PathLike
        Path where the json is located.
    Returns
    ------------------------
    dict:
        Dictionary with the data the json has.
    """

    dictionary = {}

    if os.path.exists(filepath):
        try:
            with open(filepath) as json_file:
                dictionary = json.load(json_file)

        except UnicodeDecodeError:
            print("Error reading json with utf-8, trying different approach")
            # This might lose data, verify with Jeff the json encoding
            with open(filepath, "rb") as json_file:
                data = json_file.read()
                data_str = data.decode("utf-8", errors="ignore")
                dictionary = json.loads(data_str)

    return dictionary


def modify_xml_with_channel_names(
    input_xml_path: str, modified_xml_path: str, channel_num: int
):
    """
    Channel names are an xml convention.
    """
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    for item in (
        root.find("SequenceDescription")
        .find("ImageLoader")
        .find("zgroups")
        .findall("zgroup")
    ):
        tile_name = item.find("path").text
        item.find("path").text = tile_name.replace(".zarr", f"_ch_{channel_num}.zarr")

    tree.write(modified_xml_path, encoding="utf-8", xml_declaration=True)


def get_tile_zyz_resolution(input_xml_path: str) -> list[int]:
    """
    Parse tile resolution to store in ome_ngff metadata
    """
    tree = ET.parse(input_xml_path)
    root = tree.getroot()

    res_xyz = (
        root.find("SequenceDescription")
        .find("ViewSetups")
        .find("ViewSetup")
        .find("voxelSize")
        .find("size")
        .text
    )
    res_zyx = [float(num) for num in res_xyz.split(" ")[::-1]]

    return res_zyx


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs


def get_resolution(acquisition_config) -> Tuple[int]:
    """
    Gets the image resolution from the acquisiton.json

    Parameters
    ----------
    acquisition_config: dict
        Dictionary with the acquisition metadata

    Returns
    -------
    Tuple[float]
        Tuple of floats with the image resolution
        in XYZ order
    """
    # Grabbing a tile with metadata from acquisition - we assume all dataset
    # was acquired with the same resolution
    return metadata_compat.get_voxel_resolution(acquisition_config)


def execute_job():
    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))
    scratch_folder = Path(os.path.abspath("../scratch"))

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/acquisition.json",
        f"{data_folder}/bigstitcher.xml",
    ]

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    acquisition_dict = read_json_as_dict(f"{data_folder}/acquisition.json")
    voxel_resolution = get_resolution(acquisition_dict)

    # Prep inputs
    # Reference Path
    # ../data/preprocessed_data/Ex_639_Em_667
    base_path = data_folder.joinpath("preprocessed_data")

    smartspim_channel = list(base_path.glob("Ex_*_Em_*"))

    if len(smartspim_channel):
        start_time = datetime.now(timezone.utc)
        resource_monitor = utils.ResourceMonitor(interval_seconds=2.0).start()

        input_path = smartspim_channel[0]
        output_path = results_folder.joinpath(f"{input_path.name}.zarr")

        xml_path = data_folder.joinpath("bigstitcher.xml")
        modified_xml_path = scratch_folder.joinpath("bigstitcher.xml")
        channel_num = 0
        modify_xml_with_channel_names(xml_path, modified_xml_path, channel_num)

        output_params = io.OutputParameters(
            path=output_path,
            resolution_zyx=[
                voxel_resolution[-1],
                voxel_resolution[-2],
                voxel_resolution[-3],
            ],
        )
        blend_option = "weighted_linear_blending"

        # Run fusion
        fusion.run_fusion(
            str(input_path),
            str(modified_xml_path),
            channel_num,
            output_params,
            blend_option,
            smartspim=True,
        )

        # Log 'done' file for next capsule in pipeline.
        # Unique log filename
        unique_id = str(uuid.uuid4())
        timestamp = int(time.time() * 1000)
        log_path = str(results_folder.joinpath(f"file_{timestamp}_{unique_id}.yml"))

        log_content = {}
        log_content["channel_name"] = Path(input_path).name
        log_content["resolution_zyx"] = list(output_params.resolution_zyx)
        with open(log_path, "w") as file:
            yaml.dump(log_content, file)

        # Downsampling factor
        scale_factor = [2, 2, 2]
        dataset_name = output_path.name

        store = zarr.DirectoryStore(output_path)
        zarr_group = zarr.open(store, mode="a")

        n_workers = int(utils.get_code_ocean_cpu_limit())
        n_levels = 4
        threads_per_worker = 1

        # Computing multiscales
        compute_multiscale(
            orig_lazy_data=da.from_zarr(f"{output_path}/0"),
            zarr_group=zarr_group,
            scale_factor=scale_factor,
            n_workers=n_workers,
            voxel_size=[
                voxel_resolution[-1],
                voxel_resolution[-2],
                voxel_resolution[-3],
            ],  # ZYX order
            image_name=dataset_name,
            n_levels=n_levels,
            threads_per_worker=threads_per_worker,
        )
        resource_monitor.stop()
        end_time = datetime.now(timezone.utc)

        data_process = DataProcess(
            process_type=ProcessName.IMAGE_TILE_FUSING,
            name="Image tile fusing",
            stage=ProcessStage.PROCESSING,
            code=Code(
                url=__url__,
                name=__title__,
                version=__version__,
            ),
            experimenters=__maintainers__,
            pipeline_name=__pipeline_name__,
            start_date_time=start_time,
            end_date_time=end_time,
            output_path=str(output_path),
            output_parameters={
                "input_location": str(xml_path),
                "voxel_resolution": voxel_resolution,
                "scale_factor": scale_factor,
                "pyramid_levels": n_levels,
                "n_workers": n_workers,
                "threads_per_worker": threads_per_worker,
            },
            resources=resource_monitor.to_resource_usage(cpu_cores=n_workers),
            notes=f"Fusing channel {dataset_name}",
        )

        utils.generate_processing(
            data_processes=[data_process],
            dest_processing=results_folder,
            pipeline_name=__pipeline_name__,
            pipeline_version=__pipeline_version__,
            pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
            prefix=output_path.stem,
        )

    else:
        print("No smartspim channels were provided!")


if __name__ == "__main__":
    # Some configurations helpful for GPU processing.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    print("Multiprocessing start method: ", mp.get_start_method(allow_none=False))
    print(
        "Multiprocessing start forkserver: ",
        mp.set_start_method("forkserver", force=True),
    )
    print("Multiprocessing start method: ", mp.get_start_method(allow_none=False))
    torch.cuda.empty_cache()

    execute_job()
