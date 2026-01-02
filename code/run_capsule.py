"""
Module for bigstitcher fusion. It assumes that the input
is a bigstitcher.xml with the transforms that need to be
applied for each of the stacks.

This fusion worker expects:
- preprocessed data directory of zarrs to fuse.
- complementary bigstitcher.xml
- named xml of the following format: SmartSPIM_dataset_num_datetime_stitching_channel_channel_info
  This information informs the output location of the multiscaled zarr.
"""

import json
import logging
import multiprocessing as mp
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import psutil
import yaml
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing, ProcessName)


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


def modify_xml_removing_nextflow_folder(
    input_xml_path: str, modified_xml_path: str, new_data_path: str
):
    """
    Channel names are an xml convention.
    """
    tree = ET.parse(input_xml_path)
    root = tree.getroot()
    for item in root.find("SequenceDescription").find("ImageLoader").findall("zarr"):
        tile_name = item.text
        print(tile_name)
        item.text = new_data_path

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


def execute_command_helper(command: str, print_command: bool = False) -> None:
    """
    Execute a shell command.

    Parameters
    ------------------------

    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------

    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def execute_command(
    command: str, logger: logging.Logger, verbose: Optional[bool] = False
):
    """
    Execute a shell command with a given configuration.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.

    logger: logging.Logger
        Logger object

    verbose: Optional[bool]
        Prints the command in the console

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """
    for out in execute_command_helper(command, verbose):
        if len(out):
            logger.info(out)


def main():
    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))
    scratch_folder = Path(os.path.abspath("../scratch"))

    BIGSTITCHER_PATH = os.getenv("BIGSTITCHER_HOME")
    if not BIGSTITCHER_PATH:
        raise ValueError("Please, set the BIGSTITCHER_HOME env value.")

    BIGSTITCHER_PATH = Path(BIGSTITCHER_PATH)
    env = os.environ.copy()
    print("Running from cwd:", os.getcwd())
    print("BIGSTITCHER_PATH:", BIGSTITCHER_PATH)
    print("Env JAVA_HOME:", os.environ.get("JAVA_HOME"))

    if not BIGSTITCHER_PATH.exists():
        raise ValueError("Please, set the BIGSTITCHER_PATH env value.")

    print(f"BigStitcher path: {BIGSTITCHER_PATH}")
    print(f"Os environ: {os.environ}")
    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/bigstitcher.xml",
    ]

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    # Prep inputs
    # Reference Path
    # ../data/preprocessed_data/Ex_639_Em_667
    print(list(data_folder.glob("*")))
    base_path = data_folder.joinpath("preprocessed_data")
    print("Base path: ", list(base_path.glob("*")))

    smartspim_channel = list(base_path.glob("Ex_*_Em_*"))

    if len(smartspim_channel):
        start_time = time.time()

        input_path = smartspim_channel[0]
        output_path = results_folder.joinpath(f"{input_path.name}.zarr")

        xml_path = data_folder.joinpath("bigstitcher.xml")
        modified_xml_path = scratch_folder.joinpath("bigstitcher.xml")
        channel_num = 0
        modify_xml_removing_nextflow_folder(
            xml_path, modified_xml_path, str(input_path)
        )

        output_dir = str(results_folder.joinpath(output_path))

        # Create output directory with multires folders
        process1 = subprocess.run(
            [
                "bash",
                f"./create-fusion-container",
                "-x",
                str(modified_xml_path),
                "-o",
                output_dir,
                "-d",
                "UINT16",
                "-ds",
                "1,1,1",
                "-ds",
                "2,2,2",
                "-ds",
                "4,4,4",
                "-ds",
                "8,8,8",
                "-ds",
                "16,16,16",
                "-ds",
                "32,32,32",
                "-ds",
                "64,64,64",
                "-ds",
                "128,128,128",
                "-ds",
                "256,256,256",
                "--anisotropyFactor",
                "1",
            ],
            check=True,
            cwd=BIGSTITCHER_PATH,
            env=env,
        )

        # Run fusion
        process2 = subprocess.run(
            [
                "bash",
                f"./affine-fusion",
                "-o",
                output_dir,
                "-s",
                "ZARR",
                "--prefetch",
            ],
            check=True,
            cwd=BIGSTITCHER_PATH,
            env=env,
        )

        end_time = time.time()

        data_process = DataProcess(
            name=ProcessName.IMAGE_TILE_FUSING,
            software_version="0.0.4",
            start_date_time=start_time,
            end_date_time=end_time,
            input_location=str(xml_path),
            output_location=str(output_path),
            outputs={
                "container_params": {
                    "parameters": [
                        "-x",
                        str(modified_xml_path),
                        "-o",
                        str(output_dir),
                        "-d",
                        "UINT16",
                        "-ds",
                        "1,1,1",
                        "-ds",
                        "2,2,2",
                        "-ds",
                        "4,4,4",
                        "-ds",
                        "8,8,8",
                        "-ds",
                        "16,16,16",
                        "-ds",
                        "32,32,32",
                        "-ds",
                        "64,64,64",
                        "-ds",
                        "128,128,128",
                        "-ds",
                        "256,256,256",
                        "--anisotropyFactor",
                        "1",
                    ]
                },
                "affine_fusion_params": {
                    "parameters": [
                        f"{BIGSTITCHER_PATH}/affine-fusion",
                        "-o",
                        str(output_dir),
                        "-s",
                        "ZARR",
                        "--prefetch",
                    ]
                },
            },
            code_url="https://github.com/AllenNeuralDynamics/aind-smartspim-fuse",
            code_version="0.0.4",
            parameters={},
            notes="Fusing channel with BigStitcher",
        )

        generate_processing(
            data_processes=[data_process],
            dest_processing=results_folder,
            prefix=Path(output_path).stem,
            processor_full_name="Camilo Laiton",
            pipeline_version="3.0.0",
        )

    else:
        print("No smartspim channels were provided!")


if __name__ == "__main__":
    main()
