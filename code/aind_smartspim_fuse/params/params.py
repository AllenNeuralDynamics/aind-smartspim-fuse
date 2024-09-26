"""
Module to declare the parameters for the stitching package
"""

import platform

import yaml
from argschema import ArgSchema
from argschema.fields import InputDir, InputFile, Str

from .._shared.types import PathLike


class InputFileBasedLinux(InputFile):
    """

    InputFileBasedOS is a :class:`argschema.fields.InputFile`
    subclass which is a path to a file location which can be
    read by the user depending if it's on Linux or not.

    """

    def _validate(self, value: str):
        """
        Validates the filesystem

        Parameters
        -------------
        value: str
            Path where the file is located
        """
        if platform.system() != "Windows":
            super()._validate(value)


class FusionParams(ArgSchema):
    """
    Parameters for fusion
    """

    data_folder = InputDir(
        required=True, metadata={"description": "Path where the data is located"}
    )

    transforms_xml_path = InputFileBasedLinux(
        required=True,
        metadata={"description": "Path where the transformations are"},
    )

    output_fused_path = InputDir(
        required=True,
        metadata={"description": "Path where the fused OMEZarr will be stored"},
    )

    intermediate_fused_folder = InputDir(
        required=True,
        metadata={"description": "Path where the intemediate results will be stored"},
    )

    channel_regex = Str(
        required=False,
        metadata={"description": "Regular expression to identify a SmartSPIM dataset"},
        dump_default="(Ex_[0-9]*_Em_[0-9]*)",
    )


def get_yaml(yaml_path: PathLike):
    """
    Gets the default configuration from a YAML file

    Parameters
    --------------
    filename: str
        Path where the YAML is located

    Returns
    --------------
    dict
        Dictionary with the yaml configuration
    """

    config = None
    try:
        with open(yaml_path, "r") as stream:
            config = yaml.safe_load(stream)
    except Exception as error:
        raise error

    return config
