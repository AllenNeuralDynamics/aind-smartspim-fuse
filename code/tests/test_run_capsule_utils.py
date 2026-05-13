"""Tests for utility functions defined directly in run_capsule.py.

run_capsule.py is not a package module, so we add code/ to sys.path and
import it as a top-level module.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# run_capsule.py lives directly in code/, not inside a package.
_CODE_DIR = str(Path(__file__).parent.parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

try:
    import run_capsule

    RUN_CAPSULE_AVAILABLE = True
except ImportError:
    RUN_CAPSULE_AVAILABLE = False

# Minimal BigStitcher XML with one ViewSetup containing voxel size info.
_BIGSTITCHER_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<SpimData version="0.2">
  <SequenceDescription>
    <ViewSetups>
      <ViewSetup>
        <id>0</id>
        <voxelSize>
          <unit>um</unit>
          <size>0.748 0.748 2.0</size>
        </voxelSize>
      </ViewSetup>
    </ViewSetups>
    <ImageLoader format="bdv.n5">
      <zarr type="absolute">/old/data/path</zarr>
    </ImageLoader>
  </SequenceDescription>
</SpimData>"""


@unittest.skipUnless(RUN_CAPSULE_AVAILABLE, "run_capsule not importable")
class TestValidateCapsuleInputs(unittest.TestCase):
    def test_all_present_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "bigstitcher.xml"
            f.touch()
            result = run_capsule.validate_capsule_inputs([str(f)])
            self.assertEqual(result, [])

    def test_missing_file_appears_in_result(self):
        result = run_capsule.validate_capsule_inputs(["/nonexistent/file.xml"])
        self.assertEqual(len(result), 1)
        self.assertIn("file.xml", result[0])

    def test_mix_of_present_and_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            present = Path(tmpdir) / "exists.xml"
            present.touch()
            missing = "/nonexistent/missing.xml"
            result = run_capsule.validate_capsule_inputs([str(present), missing])
            self.assertEqual(len(result), 1)
            self.assertIn("missing.xml", result[0])

    def test_empty_list_returns_empty_list(self):
        result = run_capsule.validate_capsule_inputs([])
        self.assertEqual(result, [])

    def test_multiple_missing_all_returned(self):
        result = run_capsule.validate_capsule_inputs(
            ["/no/a.xml", "/no/b.xml", "/no/c.xml"]
        )
        self.assertEqual(len(result), 3)


@unittest.skipUnless(RUN_CAPSULE_AVAILABLE, "run_capsule not importable")
class TestGetTileZyzResolution(unittest.TestCase):
    def _write_xml(self, tmpdir):
        p = Path(tmpdir) / "bigstitcher.xml"
        p.write_text(_BIGSTITCHER_XML)
        return str(p)

    def test_returns_list_of_three_floats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_capsule.get_tile_zyz_resolution(self._write_xml(tmpdir))
            self.assertEqual(len(res), 3)
            for v in res:
                self.assertIsInstance(v, float)

    def test_z_is_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # XML size is "0.748 0.748 2.0" (X Y Z); reversed → Z Y X
            res = run_capsule.get_tile_zyz_resolution(self._write_xml(tmpdir))
            self.assertAlmostEqual(res[0], 2.0)

    def test_y_is_second(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_capsule.get_tile_zyz_resolution(self._write_xml(tmpdir))
            self.assertAlmostEqual(res[1], 0.748)

    def test_x_is_third(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_capsule.get_tile_zyz_resolution(self._write_xml(tmpdir))
            self.assertAlmostEqual(res[2], 0.748)


@unittest.skipUnless(RUN_CAPSULE_AVAILABLE, "run_capsule not importable")
class TestGetResolution(unittest.TestCase):
    def _acquisition(self, x=0.748, y=0.748, z=2.0):
        return {
            "tiles": [
                {
                    "coordinate_transformations": [
                        {"type": "scale", "scale": [str(x), str(y), str(z)]}
                    ]
                }
            ]
        }

    def test_returns_three_values(self):
        result = run_capsule.get_resolution(self._acquisition())
        self.assertEqual(len(result), 3)

    def test_x_value(self):
        x, _, _ = run_capsule.get_resolution(self._acquisition(x=1.5))
        self.assertAlmostEqual(x, 1.5)

    def test_y_value(self):
        _, y, _ = run_capsule.get_resolution(self._acquisition(y=2.0))
        self.assertAlmostEqual(y, 2.0)

    def test_z_value(self):
        _, _, z = run_capsule.get_resolution(self._acquisition(z=3.0))
        self.assertAlmostEqual(z, 3.0)

    def test_values_are_floats(self):
        x, y, z = run_capsule.get_resolution(self._acquisition())
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertIsInstance(z, float)

    def test_uses_only_first_tile(self):
        config = self._acquisition(x=1.0)
        config["tiles"].append(
            {
                "coordinate_transformations": [
                    {"type": "scale", "scale": ["9.9", "9.9", "9.9"]}
                ]
            }
        )
        x, _, _ = run_capsule.get_resolution(config)
        self.assertAlmostEqual(x, 1.0)

    def test_ignores_non_scale_transforms(self):
        config = {
            "tiles": [
                {
                    "coordinate_transformations": [
                        {"type": "translation", "translation": [0, 0, 0]},
                        {"type": "scale", "scale": ["0.748", "0.748", "2.0"]},
                    ]
                }
            ]
        }
        x, y, z = run_capsule.get_resolution(config)
        self.assertAlmostEqual(x, 0.748)
        self.assertAlmostEqual(z, 2.0)


@unittest.skipUnless(RUN_CAPSULE_AVAILABLE, "run_capsule not importable")
class TestReadJsonAsDictRunCapsule(unittest.TestCase):
    def test_reads_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"hello": "world"}, f)
            fname = f.name
        try:
            result = run_capsule.read_json_as_dict(fname)
            self.assertEqual(result, {"hello": "world"})
        finally:
            os.unlink(fname)

    def test_missing_file_returns_empty_dict(self):
        result = run_capsule.read_json_as_dict("/nonexistent/path.json")
        self.assertEqual(result, {})


@unittest.skipUnless(RUN_CAPSULE_AVAILABLE, "run_capsule not importable")
class TestGetCodeOceanCpuLimitRunCapsule(unittest.TestCase):
    def setUp(self):
        self._orig = {
            k: os.environ.pop(k, None) for k in ("CO_CPUS", "AWS_BATCH_JOB_ID")
        }

    def tearDown(self):
        for k, v in self._orig.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def test_co_cpus_returns_int(self):
        os.environ["CO_CPUS"] = "4"
        result = run_capsule.get_code_ocean_cpu_limit()
        self.assertIsInstance(result, int)
        self.assertEqual(result, 4)

    def test_aws_batch_returns_1(self):
        os.environ["AWS_BATCH_JOB_ID"] = "job-123"
        result = run_capsule.get_code_ocean_cpu_limit()
        self.assertEqual(result, 1)

    def test_result_is_never_a_string(self):
        os.environ["CO_CPUS"] = "16"
        result = run_capsule.get_code_ocean_cpu_limit()
        self.assertNotIsInstance(result, str)


@unittest.skipUnless(RUN_CAPSULE_AVAILABLE, "run_capsule not importable")
class TestModifyXmlRemovingNextflowFolder(unittest.TestCase):
    def test_updates_zarr_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_xml = Path(tmpdir) / "input.xml"
            input_xml.write_text(_BIGSTITCHER_XML)
            output_xml = Path(tmpdir) / "modified.xml"

            run_capsule.modify_xml_removing_nextflow_folder(
                str(input_xml), str(output_xml), "/new/data/path"
            )

            content = output_xml.read_text()
            self.assertIn("/new/data/path", content)

    def test_output_file_is_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_xml = Path(tmpdir) / "input.xml"
            input_xml.write_text(_BIGSTITCHER_XML)
            output_xml = Path(tmpdir) / "modified.xml"

            run_capsule.modify_xml_removing_nextflow_folder(
                str(input_xml), str(output_xml), "/new/path"
            )

            self.assertTrue(output_xml.exists())

    def test_old_path_no_longer_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_xml = Path(tmpdir) / "input.xml"
            input_xml.write_text(_BIGSTITCHER_XML)
            output_xml = Path(tmpdir) / "modified.xml"

            run_capsule.modify_xml_removing_nextflow_folder(
                str(input_xml), str(output_xml), "/new/path"
            )

            content = output_xml.read_text()
            self.assertNotIn("/old/data/path", content)


if __name__ == "__main__":
    unittest.main()
