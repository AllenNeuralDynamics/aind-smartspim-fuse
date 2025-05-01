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
from pathlib import Path
from typing import List, Tuple

import aind_cloud_fusion.fusion as fusion
import aind_cloud_fusion.io as io
import dask.array as da
import psutil
import torch
import yaml
import zarr
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing, ProcessName)

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
    
    # Find all zgroup elements in the zgroups section
    zgroups_elem = root.find("SequenceDescription").find("ImageLoader").find("zgroups")
    
    if zgroups_elem is not None:
        for zgroup in zgroups_elem.findall("zgroup"):
            # Get the current path attribute
            zarr_path = zgroup.get("path")
            
            if zarr_path:
                # Insert channel number before the .ome.zarr extension
                full_extension = ''.join(Path(zarr_path).suffixes)
                modified_path = zarr_path.replace(full_extension, f"_ch_{channel_num}{full_extension}")
                zgroup.set("path", modified_path)
    
    else:
        raise ValueError(f"{input_xml_path} does not have zgroups")
    
    # Write the modified XML to the output path
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


def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
        cfs_quota_us = int(fp.read())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
        cfs_period_us = int(fp.read())
    container_cpus = cfs_quota_us // cfs_period_us
    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


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
    tile_coord_transforms = acquisition_config["tiles"][0]["coordinate_transformations"]

    scale_transform = [
        x["scale"] for x in tile_coord_transforms if x["type"] == "scale"
    ][0]

    x = float(scale_transform[0])
    y = float(scale_transform[1])
    z = float(scale_transform[2])

    return x, y, z


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: str,
    prefix: str,
    processor_full_name: str,
    pipeline_version: str,
):
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    processor_full_name: str
        Person in charged of running the pipeline
        for this data asset

    pipeline_version: str
        Terastitcher pipeline version

    """
    # flake8: noqa: E501
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
        note="Metadata for fusion step",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing only contains metadata about fusion \
            and needs to be compiled with other steps at the end",
    )

    processing.write_standard_file(output_directory=dest_processing, prefix=prefix)


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
        start_time = time.time()

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

        n_workers = int(get_code_ocean_cpu_limit())
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
        end_time = time.time()

        data_process = DataProcess(
            name=ProcessName.IMAGE_TILE_FUSING,
            software_version="0.0.2",
            start_date_time=start_time,
            end_date_time=end_time,
            input_location=str(xml_path),
            output_location=str(output_path),
            outputs={},
            code_url="",
            code_version="0.0.2",
            parameters={
                "voxel_resolution": voxel_resolution,
                "scale_factor": scale_factor,
                "pyramid_levels": n_levels,
                "n_workers": n_workers,
                "threads_per_worker": threads_per_worker,
            },
            notes=f"Fusing channel {dataset_name}",
        )

        generate_processing(
            data_processes=[data_process],
            dest_processing=results_folder,
            prefix=output_path.stem,
            processor_full_name="Camilo Laiton",
            pipeline_version="3.0.0",
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
