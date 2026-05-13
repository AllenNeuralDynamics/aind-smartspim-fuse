# aind-smartspim-fuse

Repository that hosts the fusing step applied to SmartSPIM datasets.
The primary fusion algorithm is **BigStitcher** (`run_capsule.py`).
Alternative runners for **TeraStitcher** (`run_terastitcher_capsule.py`) and
**CloudFusion** (`run_cloudfusion.py`) are also available.

## BigStitcher pipeline (primary, `run_capsule.py`)

Required input in `../data/`:
- `bigstitcher.xml` — BigStitcher-format XML with tile transforms
- `preprocessed_data/Ex_*_Em_*/` — preprocessed zarr tile data

Processing steps:

1. Validate capsule inputs — confirms `bigstitcher.xml` is present.
2. Locate the SmartSPIM channel directory under `preprocessed_data/`.
3. Rewrite the XML to point at the actual data path on disk
   (`modify_xml_removing_nextflow_folder`).
4. Run `create-fusion-container` (BigStitcher) to create the output zarr
   structure with nine downsampling levels (1×–256×, UINT16).
5. Run `affine-fusion` (BigStitcher) to fill the zarr store with fused data.
6. Write AIND processing metadata (`processing.json`) to `../results/`.

Output in `../results/`:
```
Ex_*_Em_*.zarr/        # fused OME-Zarr (multi-resolution)
processing.json        # AIND processing provenance
```

## Alternative runners

| Runner | Algorithm | Key dependency |
|--------|-----------|----------------|
| `run_terastitcher_capsule.py` | TeraStitcher | `terastitcher` CLI + MPI |
| `run_cloudfusion.py` | CloudFusion | `aind_cloud_fusion` + GPU (PyTorch) |

TeraStitcher additionally requires `volume_alignments.xml`,
`processing_manifest.json`, `data_description.json`, and `acquisition.json`
in `../data/`.

## TeraStitcher documentation
You can download TeraStitcher documentation from [here](https://unicampus365-my.sharepoint.com/:b:/g/personal/g_iannello_unicampus_it/EYT9KbapjBdGvTAD2_MdbKgB5gY_h9rlvHzqp6mUNqVhIw?e=s8GrFC).

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