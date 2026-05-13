"""Tests for TeraStitcher command-building functions in terastitcher_fusion.py.

These tests verify that the functions assemble the correct CLI strings and
write expected output files. They do not execute any external process.
"""

import os
import tempfile
import unittest
from pathlib import Path

try:
    from aind_smartspim_fuse.terastitcher_fusion import (
        build_parallel_command, terastitcher_import_cmd,
        terastitcher_merge_cmd)

    TERAS_AVAILABLE = True
except ImportError:
    TERAS_AVAILABLE = False


def _cpu_params(n_procs=4, additional=None, hostfile="/etc/hosts"):
    return {
        "cpu_params": {
            "number_processes": n_procs,
            "additional_params": additional or [],
            "hostfile": hostfile,
        }
    }


def _import_params():
    return {
        "vxl1": 0.748,
        "vxl2": 0.748,
        "vxl3": 2.0,
        "additional_params": [],
    }


def _merge_params(tmpdir):
    return {
        "s": str(Path(tmpdir) / "xml_merging_Ex_488_Em_525.xml"),
        "d": str(Path(tmpdir) / "output"),
        "sfmt": '"TIFF (unstitched, 3D)"',
        "dfmt": '"TIFF (tiled, 4D)"',
        "cpu_params": {
            "number_processes": 4,
            "additional_params": [],
            "hostfile": "/etc/hosts",
        },
        "width": 256,
        "height": 256,
        "depth": 256,
        "additional_params": ["fixed_tiling"],
        "ch_dir": "Ex_488_Em_525",
    }


@unittest.skipUnless(TERAS_AVAILABLE, "aind_smartspim_fuse not importable")
class TestBuildParallelCommand(unittest.TestCase):
    def test_contains_mpirun(self):
        cmd = build_parallel_command(_cpu_params(), "/path/to/tool.py")
        self.assertIn("mpirun -np", cmd)

    def test_contains_process_count(self):
        cmd = build_parallel_command(_cpu_params(n_procs=8), "/tool.py")
        self.assertIn("8", cmd)

    def test_contains_tool_path(self):
        cmd = build_parallel_command(_cpu_params(), "/some/tool.py")
        self.assertIn("python /some/tool.py", cmd)

    def test_contains_hostfile(self):
        cmd = build_parallel_command(_cpu_params(hostfile="/my/hosts"), "/tool.py")
        self.assertIn("--hostfile /my/hosts", cmd)

    def test_includes_additional_params(self):
        cmd = build_parallel_command(
            _cpu_params(additional=["overwrite_zeros"]), "/tool.py"
        )
        self.assertIn("--overwrite_zeros", cmd)

    def test_empty_additional_params_still_produces_valid_cmd(self):
        cmd = build_parallel_command(_cpu_params(additional=[]), "/tool.py")
        self.assertIn("mpirun", cmd)

    def test_process_count_is_embedded_correctly(self):
        for n in (1, 4, 16):
            with self.subTest(n=n):
                cmd = build_parallel_command(_cpu_params(n_procs=n), "/tool.py")
                self.assertIn(f"mpirun -np {n}", cmd)


@unittest.skipUnless(TERAS_AVAILABLE, "aind_smartspim_fuse not importable")
class TestTerastitcherImportCmd(unittest.TestCase):
    CHANNEL = "Ex_488_Em_525"

    def test_returns_two_element_tuple(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = terastitcher_import_cmd(
                input_path="/data/channel",
                xml_output_path=tmpdir,
                import_params=_import_params(),
                channel_name=self.CHANNEL,
            )
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)

    def test_command_is_a_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd, _ = terastitcher_import_cmd(
                input_path="/data/channel",
                xml_output_path=tmpdir,
                import_params=_import_params(),
                channel_name=self.CHANNEL,
            )
            self.assertIsInstance(cmd, str)

    def test_command_contains_import_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd, _ = terastitcher_import_cmd(
                input_path="/data/channel",
                xml_output_path=tmpdir,
                import_params=_import_params(),
                channel_name=self.CHANNEL,
            )
            self.assertIn("--import", cmd)

    def test_command_contains_input_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd, _ = terastitcher_import_cmd(
                input_path="/data/my_channel",
                xml_output_path=tmpdir,
                import_params=_import_params(),
                channel_name=self.CHANNEL,
            )
            self.assertIn("/data/my_channel", cmd)

    def test_binary_path_contains_channel_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, binary_path = terastitcher_import_cmd(
                input_path="/data/channel",
                xml_output_path=tmpdir,
                import_params=_import_params(),
                channel_name=self.CHANNEL,
            )
            self.assertIn(self.CHANNEL, binary_path)

    def test_binary_path_ends_with_bin(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, binary_path = terastitcher_import_cmd(
                input_path="/data/channel",
                xml_output_path=tmpdir,
                import_params=_import_params(),
                channel_name=self.CHANNEL,
            )
            self.assertTrue(binary_path.endswith(".bin"))

    def test_writes_json_params_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            terastitcher_import_cmd(
                input_path="/data/channel",
                xml_output_path=tmpdir,
                import_params=_import_params(),
                channel_name=self.CHANNEL,
            )
            expected = Path(tmpdir) / f"import_params_{self.CHANNEL}.json"
            self.assertTrue(expected.exists())

    def test_output_xml_path_contains_channel_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd, _ = terastitcher_import_cmd(
                input_path="/data/channel",
                xml_output_path=tmpdir,
                import_params=_import_params(),
                channel_name=self.CHANNEL,
            )
            self.assertIn(self.CHANNEL, cmd)


@unittest.skipUnless(TERAS_AVAILABLE, "aind_smartspim_fuse not importable")
class TestTerastitcherMergeCmd(unittest.TestCase):
    CHANNEL = "Ex_488_Em_525"

    def test_returns_string(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = terastitcher_merge_cmd(
                xml_output_path=Path(tmpdir),
                merge_params=_merge_params(tmpdir),
                channel_name=self.CHANNEL,
                paraconverter_path="/path/to/paraconverter.py",
            )
            self.assertIsInstance(cmd, str)

    def test_contains_mpirun(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = terastitcher_merge_cmd(
                xml_output_path=Path(tmpdir),
                merge_params=_merge_params(tmpdir),
                channel_name=self.CHANNEL,
                paraconverter_path="/path/to/paraconverter.py",
            )
            self.assertIn("mpirun", cmd)

    def test_s_flag_uses_short_form(self):
        # The function replaces --s= with -s= to comply with paraconverter
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = terastitcher_merge_cmd(
                xml_output_path=Path(tmpdir),
                merge_params=_merge_params(tmpdir),
                channel_name=self.CHANNEL,
                paraconverter_path="/path/to/paraconverter.py",
            )
            self.assertNotIn("--s=", cmd)
            self.assertIn("-s=", cmd)

    def test_d_flag_uses_short_form(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = terastitcher_merge_cmd(
                xml_output_path=Path(tmpdir),
                merge_params=_merge_params(tmpdir),
                channel_name=self.CHANNEL,
                paraconverter_path="/path/to/paraconverter.py",
            )
            self.assertNotIn("--d=", cmd)
            self.assertIn("-d=", cmd)

    def test_writes_merge_json_params_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            terastitcher_merge_cmd(
                xml_output_path=Path(tmpdir),
                merge_params=_merge_params(tmpdir),
                channel_name=self.CHANNEL,
                paraconverter_path="/path/to/paraconverter.py",
            )
            expected = Path(tmpdir) / f"merge_volume_params_{self.CHANNEL}.json"
            self.assertTrue(expected.exists())

    def test_paraconverter_path_in_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = terastitcher_merge_cmd(
                xml_output_path=Path(tmpdir),
                merge_params=_merge_params(tmpdir),
                channel_name=self.CHANNEL,
                paraconverter_path="/my/paraconverter.py",
            )
            self.assertIn("/my/paraconverter.py", cmd)


if __name__ == "__main__":
    unittest.main()
