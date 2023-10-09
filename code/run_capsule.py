""" top level run script """
import os
from pathlib import Path
from typing import List

from aind_smartspim_fuse import fuse


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
            missing_inputs.append(required_input_element)

    return missing_inputs


def run():
    """Function to start image fusion"""
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")
    scratch_folder = os.path.abspath("../scratch")

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/volume_alignments.xml",
    ]

    missing_files = validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    fuse.terasticher(
        transforms_xml_path=Path(results_folder).joinpath("volume_alignments.xml"),
        output_fused_path=Path(results_folder),
        intermediate_fused_folder=Path(scratch_folder),
    )


if __name__ == "__main__":
    run()
