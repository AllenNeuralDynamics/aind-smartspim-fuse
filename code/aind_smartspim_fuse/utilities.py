import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import xmltodict
from aind_data_schema import DerivedDataDescription, Processing
from aind_data_schema.base import AindCoreModel
from aind_data_schema.data_description import (DataLevel, Funding, Institution,
                                               Modality, Platform,
                                               RawDataDescription)

from .types import PathLike


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


def save_string_to_txt(txt: str, filepath: PathLike, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------

    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")


def execute_command_helper(
    command: str,
    print_command: bool = False,
    stdout_log_file: Optional[PathLike] = None,
) -> None:
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

    if stdout_log_file and len(str(stdout_log_file)):
        save_string_to_txt("$ " + command, stdout_log_file, "a")

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def execute_command(config: dict) -> None:
    """
    Execute a shell command with a given configuration.

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
    # Command is not executed when info
    # is True
    if config["info"]:
        config["logger"].info(config["command"])
    else:
        for out in execute_command_helper(
            config["command"], config["verbose"], config["stdout_log_file"]
        ):
            if len(out):
                config["logger"].info(out)

            if config["exists_stdout"]:
                save_string_to_txt(out, config["stdout_log_file"], "a")


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
    channel_name: str,
    regex_expr: str,
    encoding: Optional[str] = "utf-8",
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

    channel_name: str
        String with the channel name

    regex_expr: str
        Regular expression to identify
        the name of the channel

    encoding: str
        Encoding of the XML file.
        Default: UTF-8

    Returns
    -----------------
    str
        Path where the xml is stored
    """
    informative_channel_name = re.search(regex_expr, informative_channel_xml)
    modified_mergexml_path = None

    if informative_channel_name:
        # Getting the channel name
        informative_channel_name = informative_channel_name.group()

        with open(informative_channel_xml, "r", encoding=encoding) as xml_reader:
            xml_file = xml_reader.read()

        xml_dict = xmltodict.parse(xml_file)

        new_stacks_folder = xml_dict["TeraStitcher"]["stacks_dir"]["@value"].replace(
            informative_channel_name, channel_name
        )

        new_bin_folder = xml_dict["TeraStitcher"]["mdata_bin"]["@value"].replace(
            informative_channel_name, channel_name
        )

        xml_dict["TeraStitcher"]["stacks_dir"]["@value"] = new_stacks_folder
        xml_dict["TeraStitcher"]["mdata_bin"]["@value"] = new_bin_folder

        modified_mergexml_path = str(informative_channel_xml).replace(
            informative_channel_name, channel_name
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


def generate_processing(
    data_processes: List[dict],
    dest_processing: PathLike,
    pipeline_version: str,
) -> None:
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    pipeline_version: str
        Terastitcher pipeline version

    """

    # flake8: noqa: E501
    processing = Processing(
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-stitch",
        pipeline_version=pipeline_version,
        data_processes=data_processes,
    )

    with open(dest_processing, "w") as f:
        f.write(processing.json(indent=3))


def generate_data_description(
    raw_data_description_path: PathLike,
    dest_data_description: PathLike,
    process_name: Optional[str] = "stitched",
):
    """
    Generates data description for the output folder.

    Parameters
    -------------

    raw_data_description_path: PathLike
        Path where the data description file is located.

    dest_data_description: PathLike
        Path where the new data description will be placed.

    process_name: str
        Process name of the new dataset


    Returns
    -------------
    str
        New folder name for the fused
        data
    """

    f = open(raw_data_description_path, "r")
    data = json.load(f)

    institution = data["institution"]
    if isinstance(data["institution"], dict) and "abbreviation" in data["institution"]:
        institution = data["institution"]["abbreviation"]

    funding_sources = [Funding.parse_obj(fund) for fund in data["funding_source"]]
    derived = DerivedDataDescription(
        creation_time=datetime.now(),
        input_data_name=data["name"],
        process_name=process_name,
        institution=Institution[institution],
        funding_source=funding_sources,
        group=data["group"],
        investigators=data["investigators"],
        platform=Platform.SMARTSPIM,
        project_name=data["project_name"],
        restrictions=data["restrictions"],
        modality=[Modality.SPIM],
        subject_id=data["subject_id"],
    )

    with open(dest_data_description, "w") as f:
        f.write(derived.json(indent=3))

    return derived.name


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
