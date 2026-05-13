"""Tests for aind_smartspim_fuse/utils/utils.py."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from aind_smartspim_fuse.utils.utils import (
        check_path_instance, find_smartspim_channels, generate_timestamp,
        get_code_ocean_cpu_limit, get_size, helper_additional_params_command,
        helper_build_param_value_command, read_json_as_dict, save_dict_as_json,
        wavelength_to_hex)

    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestWavelengthToHex(unittest.TestCase):
    def test_below_first_bound_returns_purple(self):
        self.assertEqual(wavelength_to_hex(400), 0x690AFE)

    def test_at_first_bound_skips_to_next_bucket(self):
        # 460 is NOT < 460, falls into the 470 bucket (0x3F2EFE)
        self.assertEqual(wavelength_to_hex(460), 0x3F2EFE)

    def test_at_second_bound_skips_to_next_bucket(self):
        # 470 is NOT < 470, falls into the 480 bucket (0x4B90FE)
        self.assertEqual(wavelength_to_hex(470), 0x4B90FE)

    def test_red_channel_wavelength(self):
        # 667 < 750 → pink/red bucket 0xF00050
        self.assertEqual(wavelength_to_hex(667), 0xF00050)

    def test_above_max_bound_returns_last_color(self):
        # Loop exhausted, fallback returns last iterated value (750 bucket)
        self.assertEqual(wavelength_to_hex(800), 0xF00050)

    def test_green_channel(self):
        # 525 is between 520 and 540 → 540 bucket (0x58FEA1)
        self.assertEqual(wavelength_to_hex(525), 0x58FEA1)


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestHelperBuildParamValueCommand(unittest.TestCase):
    def test_with_equal_connector(self):
        params = {"vxl1": 0.748, "vxl2": 0.748}
        result = helper_build_param_value_command(params, equal_con=True)
        self.assertIn("--vxl1=0.748", result)
        self.assertIn("--vxl2=0.748", result)

    def test_without_equal_connector(self):
        params = {"vxl1": 0.748}
        result = helper_build_param_value_command(params, equal_con=False)
        self.assertIn("--vxl1 0.748", result)

    def test_empty_dict_returns_empty_string(self):
        result = helper_build_param_value_command({})
        self.assertEqual(result.strip(), "")

    def test_skips_non_scalar_values(self):
        params = {"list_key": [1, 2, 3], "str_key": "value"}
        result = helper_build_param_value_command(params)
        self.assertNotIn("list_key", result)
        self.assertIn("--str_key=value", result)

    def test_integer_value(self):
        result = helper_build_param_value_command({"depth": 256})
        self.assertIn("--depth=256", result)

    def test_path_value_included(self):
        params = {"out": Path("/tmp/output")}
        result = helper_build_param_value_command(params)
        self.assertIn("--out=", result)
        self.assertIn("/tmp/output", result)


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestHelperAdditionalParamsCommand(unittest.TestCase):
    def test_builds_flag_style_params(self):
        result = helper_additional_params_command(["fixed_tiling", "sparse"])
        self.assertIn("--fixed_tiling", result)
        self.assertIn("--sparse", result)

    def test_empty_list_returns_empty_string(self):
        result = helper_additional_params_command([])
        self.assertEqual(result, "")

    def test_single_param(self):
        result = helper_additional_params_command(["overwrite"])
        self.assertIn("--overwrite", result)


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestGetSize(unittest.TestCase):
    def test_bytes(self):
        result = get_size(512)
        self.assertIn("B", result)

    def test_kilobytes(self):
        result = get_size(1024)
        self.assertIn("KB", result)

    def test_megabytes(self):
        result = get_size(1024 * 1024)
        self.assertIn("MB", result)

    def test_gigabytes(self):
        result = get_size(1024**3)
        self.assertIn("GB", result)

    def test_zero_bytes(self):
        result = get_size(0)
        self.assertIn("B", result)
        self.assertIn("0.00", result)


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestGenerateTimestamp(unittest.TestCase):
    def test_returns_non_empty_string(self):
        ts = generate_timestamp()
        self.assertIsInstance(ts, str)
        self.assertTrue(len(ts) > 0)

    def test_custom_year_format(self):
        ts = generate_timestamp("%Y")
        self.assertTrue(ts.isdigit())
        self.assertEqual(len(ts), 4)

    def test_default_format_contains_dashes(self):
        ts = generate_timestamp()
        self.assertIn("-", ts)


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestCheckPathInstance(unittest.TestCase):
    def test_posix_path_returns_true(self):
        self.assertTrue(check_path_instance(Path("/some/path")))

    def test_string_returns_false(self):
        self.assertFalse(check_path_instance("/some/path"))

    def test_none_returns_false(self):
        self.assertFalse(check_path_instance(None))

    def test_integer_returns_false(self):
        self.assertFalse(check_path_instance(42))


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestReadJsonAsDict(unittest.TestCase):
    def test_reads_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value"}, f)
            fname = f.name
        try:
            result = read_json_as_dict(fname)
            self.assertEqual(result, {"key": "value"})
        finally:
            os.unlink(fname)

    def test_missing_file_returns_empty_dict(self):
        result = read_json_as_dict("/nonexistent/path/file.json")
        self.assertEqual(result, {})

    def test_nested_structure(self):
        data = {"a": {"b": [1, 2, 3]}, "c": True}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            fname = f.name
        try:
            result = read_json_as_dict(fname)
            self.assertEqual(result, data)
        finally:
            os.unlink(fname)


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestSaveDictAsJson(unittest.TestCase):
    def test_saves_and_reads_back(self):
        data = {"x": 1, "y": "hello"}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            save_dict_as_json(fname, data)
            with open(fname) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)
        finally:
            os.unlink(fname)

    def test_none_saves_empty_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            save_dict_as_json(fname, None)
            with open(fname) as f:
                loaded = json.load(f)
            self.assertEqual(loaded, {})
        finally:
            os.unlink(fname)

    def test_path_values_converted_to_str(self):
        data = {"out": Path("/some/output")}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            fname = f.name
        try:
            save_dict_as_json(fname, data)
            with open(fname) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["out"], "/some/output")
        finally:
            os.unlink(fname)


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestFindSmartspimChannels(unittest.TestCase):
    def test_finds_matching_channel_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "Ex_488_Em_525").mkdir()
            Path(tmpdir, "Ex_639_Em_680").mkdir()
            Path(tmpdir, "unrelated_folder").mkdir()
            channels = find_smartspim_channels(tmpdir)
            self.assertEqual(sorted(channels), ["Ex_488_Em_525", "Ex_639_Em_680"])

    def test_no_matching_channels_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "unrelated").mkdir()
            channels = find_smartspim_channels(tmpdir)
            self.assertEqual(channels, [])

    def test_custom_regex(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "Ex_488_Em_525").mkdir()
            Path(tmpdir, "Ex_639_Em_680").mkdir()
            channels = find_smartspim_channels(tmpdir, channel_regex=r"Ex_488.*")
            self.assertIn("Ex_488_Em_525", channels)
            self.assertNotIn("Ex_639_Em_680", channels)

    def test_matches_files_as_well_as_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir, "Ex_561_Em_593")).touch()
            channels = find_smartspim_channels(tmpdir)
            self.assertIn("Ex_561_Em_593", channels)


@unittest.skipUnless(UTILS_AVAILABLE, "aind_smartspim_fuse.utils not importable")
class TestGetCodeOceanCpuLimit(unittest.TestCase):
    def setUp(self):
        # Clear relevant env vars before each test
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
        os.environ["CO_CPUS"] = "8"
        result = get_code_ocean_cpu_limit()
        self.assertIsInstance(result, int)
        self.assertEqual(result, 8)

    def test_co_cpus_not_a_string(self):
        os.environ["CO_CPUS"] = "16"
        result = get_code_ocean_cpu_limit()
        self.assertNotIsInstance(result, str)

    def test_aws_batch_returns_1(self):
        os.environ["AWS_BATCH_JOB_ID"] = "some-job-id"
        result = get_code_ocean_cpu_limit()
        self.assertEqual(result, 1)

    def test_co_cpus_takes_priority_over_aws_batch(self):
        os.environ["CO_CPUS"] = "4"
        os.environ["AWS_BATCH_JOB_ID"] = "some-job-id"
        result = get_code_ocean_cpu_limit()
        self.assertEqual(result, 4)


if __name__ == "__main__":
    unittest.main()
