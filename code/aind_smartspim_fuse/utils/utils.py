"""
Module to store utility functions
"""

import json
import logging
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import psutil
import xmltodict
from aind_data_schema.base import AindCoreModel
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing)

from .._shared.types import PathLike


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


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
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def copy_file(input_filename: PathLike, output_filename: PathLike):
    """
    Copies a file to an output path

    Parameters
    ----------
    input_filename: PathLike
        Path where the file is located

    output_filename: PathLike
        Path where the file will be copied
    """

    try:
        shutil.copy(input_filename, output_filename)

    except shutil.SameFileError:
        raise shutil.SameFileError(
            f"The filename {input_filename} already exists in the output path."
        )

    except PermissionError:
        raise PermissionError(
            f"Not able to copy the file. Please, check the permissions in the output path."
        )


def check_path_instance(obj: object) -> bool:
    """
    Checks if an objects belongs to pathlib.Path subclasses.

    Parameters
    ------------------------

    obj: object
        Object that wants to be validated.

    Returns
    ------------------------

    bool:
        True if the object is an instance of Path subclass, False otherwise.
    """

    for childclass in Path.__subclasses__():
        if isinstance(obj, childclass):
            return True

    return False


def helper_additional_params_command(params: List[str]) -> str:
    """
    Helper function to build a command based on values.

    Parameters
    ------------------------

    params: list
        List with additional command values used.

    Returns
    ------------------------

    str:
        String with the parameters.

    """
    additional_params = ""
    for param in params:
        additional_params += f"--{param} "

    return additional_params


def helper_build_param_value_command(
    params: dict, equal_con: Optional[bool] = True
) -> str:
    """
    Helper function to build a command based on key:value pairs.

    Parameters
    ------------------------

    params: dict
        Dictionary with key:value pairs used for building the command.

    equal_con: Optional[bool]
        Indicates if the parameter is followed by '='. Default True.

    Returns
    ------------------------

    str:
        String with the parameters.

    """
    equal = " "
    if equal_con:
        equal = "="

    parameters = ""
    for param, value in params.items():
        if type(value) in [str, float, int] or check_path_instance(value):
            parameters += f"--{param}{equal}{str(value)} "

    return parameters


def save_string_to_txt(txt: str, filepath: PathLike, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------

    txt: str
        String to be saved

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")


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


def generate_timestamp(time_format: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """
    Generates a timestamp in string format.

    Parameters
    ------------------------
    time_format: str
        String following the conventions
        to generate the timestamp (https://strftime.org/).

    Returns
    ------------------------
    str:
        String with the actual datetime
        moment in string format.
    """
    return datetime.now().strftime(time_format)


def wavelength_to_hex(wavelength: int) -> int:
    """
    Converts wavelength to corresponding color hex value.

    Parameters
    ------------------------
    wavelength: int
        Integer value representing wavelength.

    Returns
    ------------------------
    int:
        Hex value color.
    """
    # Each wavelength key is the upper bound to a wavelgnth band.
    # Wavelengths range from 380-750nm.
    # Color map wavelength/hex pairs are generated by sampling along a CIE diagram arc.
    color_map = {
        460: 0x690AFE,  # Purple
        470: 0x3F2EFE,  # Blue-Purple
        480: 0x4B90FE,  # Blue
        490: 0x59D5F8,  # Blue-Green
        500: 0x5DF8D6,  # Green
        520: 0x5AFEB8,  # Green
        540: 0x58FEA1,  # Green
        560: 0x51FF1E,  # Green
        565: 0xBBFB01,  # Green-Yellow
        575: 0xE9EC02,  # Yellow
        580: 0xF5C503,  # Yellow-Orange
        590: 0xF39107,  # Orange
        600: 0xF15211,  # Orange-Red
        620: 0xF0121E,  # Red
        750: 0xF00050,
    }  # Pink

    for ub, hex_val in color_map.items():
        if wavelength < ub:  # Exclusive
            return hex_val
    return hex_val


def generate_new_channel_alignment_xml(
    informative_channel_xml,
    channel_path: PathLike,
    metadata_folder: PathLike,
    teras_mdata_bin: PathLike,
    encoding: Optional[str] = "utf-8",
    channel_regex: str = r"Ex_([0-9]*)_Em_([0-9]*)$",
) -> str:
    """
    Generates an XML with the displacements
    that will be applied in a channel different
    than the informative one.

    Parameters
    -----------------
    informative_channel_xml: PathLike
        Path where the informative channel xml
        is located

    channel_path: PathLike
        Path where the image dataset
        is located

    metadata_folder: PathLike
        Path where the new alignment will
        be placed

    teras_mdata_bin: PathLike
        Path where the terastitcher binary
        was placed after importing the
        dataset

    encoding: str
        Encoding of the XML file.
        Default: UTF-8

    Returns
    -----------------
    str
        Path where the xml is stored
    """

    with open(informative_channel_xml, "r", encoding=encoding) as xml_reader:
        xml_file = xml_reader.read()

    xml_dict = xmltodict.parse(xml_file)

    new_stacks_folder = xml_dict["TeraStitcher"]["stacks_dir"]["@value"] = str(
        channel_path
    )
    new_bin_folder = xml_dict["TeraStitcher"]["mdata_bin"]["@value"] = str(
        teras_mdata_bin
    )

    xml_dict["TeraStitcher"]["stacks_dir"]["@value"] = new_stacks_folder
    xml_dict["TeraStitcher"]["mdata_bin"]["@value"] = new_bin_folder

    new_channel_name = re.search(channel_regex, str(channel_path)).group()

    modified_mergexml_path = str(
        metadata_folder.joinpath(f"xml_merging_{new_channel_name}.xml")
    )

    data_to_write = xmltodict.unparse(xml_dict, pretty=True)

    end_xml_header = "?>"
    len_end_xml_header = len(end_xml_header)
    xml_end_header = data_to_write.find(end_xml_header)

    new_data_to_write = data_to_write[: xml_end_header + len_end_xml_header]
    # Adding terastitcher doctype
    new_data_to_write += '\n<!DOCTYPE TeraStitcher SYSTEM "TeraStitcher.DTD">'
    new_data_to_write += data_to_write[xml_end_header + len_end_xml_header :]

    with open(modified_mergexml_path, "w", encoding=encoding) as xml_writer:
        xml_writer.write(new_data_to_write)

    return modified_mergexml_path


def copy_available_metadata(
    input_path: PathLike, output_path: PathLike, ignore_files: List[str]
) -> List[PathLike]:
    """
    Copies all the valid metadata from the aind-data-schema
    repository that exists in a given path.

    Parameters
    -----------
    input_path: PathLike
        Path where the metadata is located

    output_path: PathLike
        Path where we will copy the found
        metadata

    ignore_files: List[str]
        List with the filenames of the metadata
        that we need to ignore from the aind-data-schema

    Returns
    --------
    List[PathLike]
        List with the metadata files that
        were copied
    """

    # We get all the valid filenames from the aind core model
    metadata_to_find = [
        cls.default_filename() for cls in AindCoreModel.__subclasses__()
    ]

    # Making sure the paths are pathlib objects
    input_path = Path(input_path)
    output_path = Path(output_path)

    found_metadata = []

    for metadata_filename in metadata_to_find:
        metadata_filename = input_path.joinpath(metadata_filename)

        if metadata_filename.exists() and metadata_filename.name not in ignore_files:
            found_metadata.append(metadata_filename)

            # Copying file to output path
            output_filename = output_path.joinpath(metadata_filename.name)
            copy_file(metadata_filename, output_filename)

    return found_metadata


def find_smartspim_channels(
    path: PathLike, channel_regex: str = r"Ex_([0-9]*)_Em_([0-9]*)$"
):
    """
    Find image channels of a dataset using a regular expression.

    Parameters
    ------------------------

    path:PathLike
        Dataset path

    channel_regex:str
        Regular expression for filtering folders in dataset path.


    Returns
    ------------------------

    List[str]:
        List with the image channels. Empty list if
        it does not find any channels with the
        given regular expression.

    """
    return [path for path in os.listdir(path) if re.search(channel_regex, path)]


def copy_available_metadata(
    input_path: PathLike, output_path: PathLike, ignore_files: List[str]
) -> List[PathLike]:
    """
    Copies all the valid metadata from the aind-data-schema
    repository that exists in a given path.

    Parameters
    -----------
    input_path: PathLike
        Path where the metadata is located

    output_path: PathLike
        Path where we will copy the found
        metadata

    ignore_files: List[str]
        List with the filenames of the metadata
        that we need to ignore from the aind-data-schema

    Returns
    --------
    List[PathLike]
        List with the metadata files that
        were copied
    """

    # We get all the valid filenames from the aind core model
    metadata_to_find = [
        cls.default_filename() for cls in AindCoreModel.__subclasses__()
    ]

    # Making sure the paths are pathlib objects
    input_path = Path(input_path)
    output_path = Path(output_path)

    found_metadata = []

    for metadata_filename in metadata_to_find:
        metadata_filename = input_path.joinpath(metadata_filename)

        if metadata_filename.exists() and metadata_filename.name not in ignore_files:
            found_metadata.append(metadata_filename)

            # Copying file to output path
            output_filename = output_path.joinpath(metadata_filename.name)
            copy_file(metadata_filename, output_filename)

    return found_metadata


def create_logger(output_log_path: PathLike) -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/fusion_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


def create_fusion_folder_structure(
    output_fused_path: PathLike, intermediate_fused_folder: PathLike, channel_name: str
) -> Tuple:
    """
    Creates the fusion folder structure.

    Parameters
    -----------
    output_fused_path: PathLike
        Path where the OMEZarr and metadata will
        live after fusion

    intermediate_fused_folder: PathLike
        Path where the intermediate files
        will live. These will not be in the final
        folder structure. e.g., 3D fused chunks
        from TeraStitcher

    channel_name: str
        SmartSPIM channel name

    Returns
    -----------
    Tuple
        Tuple with the paths pointing
        to the final fusion folder, metadata folder
        and terastitcher intermediate fusion folder
    """

    # Creating folders if necessary
    if not output_fused_path.exists():
        logging.info(f"Path {output_fused_path} does not exists. We're creating one.")
        create_folder(dest_dir=output_fused_path)

    if not intermediate_fused_folder.exists():
        logging.info(
            f"Path {intermediate_fused_folder} does not exists. We're creating one."
        )
        create_folder(dest_dir=intermediate_fused_folder)

    output_fused_path = output_fused_path.joinpath(f"fusion_{channel_name}")
    fusion_folder = output_fused_path.joinpath("OMEZarr")
    metadata_folder = output_fused_path.joinpath(f"metadata")
    teras_fusion_folder = intermediate_fused_folder.joinpath("teras_stitched")

    create_folder(fusion_folder)
    create_folder(metadata_folder)
    create_folder(teras_fusion_folder)

    return fusion_folder, metadata_folder, teras_fusion_folder


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: PathLike,
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

    processing.write_standard_file(output_directory=dest_processing)


def save_dict_as_json(
    filename: str, dictionary: dict, verbose: Optional[bool] = False
) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------

    filename: str
        Name of the json file.

    dictionary: dict
        Dictionary that will be saved as json.

    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    else:
        for key, value in dictionary.items():
            # Converting path to str to dump dictionary into json
            if check_path_instance(value):
                # TODO fix the \\ encode problem in dump
                dictionary[key] = str(value)

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)

    if verbose:
        print(f"- Json file saved: {filename}")


def profile_resources(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    monitoring_interval: int,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    monitoring_interval: int
        Monitoring interval in seconds
    """
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        time_points.append(current_time)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=monitoring_interval)
        cpu_percentages.append(cpu_percent)

        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usages.append(memory_info.percent)

        time.sleep(monitoring_interval)


def generate_resources_graphs(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    output_path: str,
    prefix: str,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    output_path: str
        Path where the image will be saved

    prefix: str
        Prefix name for the image
    """
    time_len = len(time_points)
    memory_len = len(memory_usages)
    cpu_len = len(cpu_percentages)

    min_len = min([time_len, memory_len, cpu_len])
    if not min_len:
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_points[:min_len], cpu_percentages[:min_len], label="CPU Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_points[:min_len], memory_usages[:min_len], label="Memory Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{prefix}_compute_resources.png", bbox_inches="tight")


def stop_child_process(process: multiprocessing.Process):
    """
    Stops a process

    Parameters
    ----------
    process: multiprocessing.Process
        Process to stop
    """
    process.terminate()
    process.join()


def get_size(bytes, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Parameters
    ----------
    bytes: bytes
        Bytes to scale

    suffix: str
        Suffix used for the conversion
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


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
    
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())
        
        container_cpus = cfs_quota_us // cfs_period_us

    except FileNotFoundError as e:
        container_cpus = 0

    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def print_system_information(logger: logging.Logger):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object
    """
    co_memory = os.environ.get("CO_MEMORY")
    co_memory = int(co_memory) if co_memory else None

    # System info
    sep = "=" * 40
    logger.info(f"{sep} Code Ocean Information {sep}")
    logger.info(f"Code Ocean assigned cores: {get_code_ocean_cpu_limit()}")

    if co_memory:
        logger.info(f"Code Ocean assigned memory: {get_size(co_memory)}")

    logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
    logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
    logger.info(f"Is pipeline execution?: {bool(os.environ.get('AWS_BATCH_JOB_ID'))}")

    logger.info(f"{sep} System Information {sep}")
    uname = platform.uname()
    logger.info(f"System: {uname.system}")
    logger.info(f"Node Name: {uname.node}")
    logger.info(f"Release: {uname.release}")
    logger.info(f"Version: {uname.version}")
    logger.info(f"Machine: {uname.machine}")
    logger.info(f"Processor: {uname.processor}")

    # Boot info
    logger.info(f"{sep} Boot Time {sep}")
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    logger.info(
        f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}"
    )

    # CPU info
    logger.info(f"{sep} CPU Info {sep}")
    # number of cores
    logger.info(f"Physical node cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"Total node cores: {psutil.cpu_count(logical=True)}")

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    # CPU usage
    logger.info("CPU Usage Per Core before processing:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory info
    logger.info(f"{sep} Memory Information {sep}")
    # get the memory details
    svmem = psutil.virtual_memory()
    logger.info(f"Total: {get_size(svmem.total)}")
    logger.info(f"Available: {get_size(svmem.available)}")
    logger.info(f"Used: {get_size(svmem.used)}")
    logger.info(f"Percentage: {svmem.percent}%")
    logger.info(f"{sep} Memory - SWAP {sep}")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    logger.info(f"Total: {get_size(swap.total)}")
    logger.info(f"Free: {get_size(swap.free)}")
    logger.info(f"Used: {get_size(swap.used)}")
    logger.info(f"Percentage: {swap.percent}%")

    # Network information
    logger.info(f"{sep} Network Information {sep}")
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            logger.info(f"=== Interface: {interface_name} ===")
            if str(address.family) == "AddressFamily.AF_INET":
                logger.info(f"  IP Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == "AddressFamily.AF_PACKET":
                logger.info(f"  MAC Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    logger.info(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    logger.info(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")
