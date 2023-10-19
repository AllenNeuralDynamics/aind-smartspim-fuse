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

> Note: This repository is intented to work with Code Ocean pipelines. It means that we are executing a single instance per dataset channel and the generated folder structure will be:

fusion_{channel_name}:
    - OMEZarr: Folder where we will save the fused data.
    - metadata: Generated metadata for the fusion in this channel.

It is important to mention that there's another folder that is created. This is an intermediate fusion with the 3D fused chunked tiffs and it's pointing to the scratch folder in Code Ocean by default.

## Documentation
You can access the documentation for this module [here]().

## TeraStitcher Documentation
You can download TeraStitcher documentation from [here](https://unicampus365-my.sharepoint.com/:b:/g/personal/g_iannello_unicampus_it/EYT9KbapjBdGvTAD2_MdbKgB5gY_h9rlvHzqp6mUNqVhIw?e=s8GrFC)

## Contributing

To develop the code, run
```
pip install -e .[dev]
```

### Linters and testing

There are several libraries used to run linters, check documentation, and run tests.

- Please test your changes using the **coverage** library, which will run the tests and log a coverage report:

```
coverage run -m unittest discover && coverage report
```

- Use **interrogate** to check that modules, methods, etc. have been documented thoroughly:

```
interrogate .
```

- Use **flake8** to check that code is up to standards (no unused imports, etc.):
```
flake8 .
```

- Use **black** to automatically format the code into PEP standards:
```
black .
```

- Use **isort** to automatically sort import statements:
```
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repo and open a pull request from the fork. We'll primarily use [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style for commit messages. Roughly, they should follow the pattern:
```
<type>(<scope>): <short summary>
```

where scope (optional) describes the packages affected by the code changes and type (mandatory) is one of:

- **build**: Changes that affect the build system or external dependencies (example scopes: pyproject.toml, setup.py)
- **ci**: Changes to our CI configuration files and scripts (examples: .github/workflows/ci.yml)
- **docs**: Documentation only changes
- **feat**: A new feature
- **fix**: A bug fix
- **perf**: A code change that improves performance
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **test**: Adding missing tests or correcting existing tests

### Documentation
To generate the rst files source files for documentation, run
```
sphinx-apidoc -o doc_template/source/ src 
```
Then to create the documentation html files, run
```
sphinx-build -b html doc_template/source/ doc_template/build/html
```
More info on sphinx installation can be found here: https://www.sphinx-doc.org/en/master/usage/installation.html