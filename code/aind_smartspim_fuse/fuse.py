"""
This file controls the fusion step
for a SmartSPIM dataset
"""

import logging
import os
from pathlib import Path

import utilities
from natsort import natsorted
from ng_link import NgState

from .types import PathLike

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("test.log", "a"),
    ],
)
logging.disable("DEBUG")


def terasticher(
    transforms_xml_path: PathLike,
    output_fused_path: PathLike,
    intermediate_fused_folder: PathLike,
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

    # Creating folders if necessary
    if not output_fused_path.exists():
        logging.info(f"Path {output_fused_path} does not exists. We're creating one.")
        utilities.create_folder(dest_dir=output_fused_path)

    if not intermediate_fused_folder.exists():
        logging.info(
            f"Path {intermediate_fused_folder} does not exists. We're creating one."
        )
        utilities.create_folder(dest_dir=intermediate_fused_folder)
