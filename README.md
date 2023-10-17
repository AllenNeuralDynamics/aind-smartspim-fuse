# aind-smartspim-fuse

Repository that hosts the fusing step applied to the smartspim datasets. The current option at the moment is fusion with TeraStitcher.

The processing steps for this capsule are:

1. We read the volume_alignments.xml file which contains the image transformation steps to align the whole volume.
2. We validate capsule inputs in the data folder. The needed files are: "volume_alignments.xml", "processing_manifest.json" and "data_description.json".
    - volume_alignments.xml: This file contains the image transformations applied to the stacks. By default, we are using TeraStitcher.
    - processing_manifest.json: This file contains the image processing steps and some extra metadata necessary to fuse datasets.
    - data_description.json: This file contains metadata about the dataset itself. Please, check the [aind-data-schema](https://github.com/AllenNeuralDynamics/aind-data-schema) repository to know more about this file.
3. We create the fusion folder structure, generate new data description and copy all necessary metadata.
4. We start the chunked fusion using TeraStitcher. Please, check the default parameters defined on the params folder of this repository.
5. OMEZarr File Format Convertion: We configured TeraStitcher to output blocks of size (256, 256, 256) in most of the brain and in the border with whatever is left for that specific brain. These images are in TIFF format and need to be converted to OMEZarr for cloud visualization. In order to to this, we read the entire fused volume lazily, rechunk it to chunks of (128, 128, 128) and write it in a BigChunk approach. With BigChunk, our experiments show that to avoid a very large dask graph we can take n chunks (e.g., 4 chunks of 256) and then write those down. This is a good approach for very large datasets > 1 TB.
6. Generate neuroglancer link.