"""Tests for CLI argument parsing (og-free, uvicorn never launched).

main() parses argv then calls _run_server; we patch _run_server to capture
the parsed Namespace without importing uvicorn or building a Backend.
"""

import unittest
from unittest import mock

from ovinfer import cli


class CliAutoloadTest(unittest.TestCase):
    def _autoload(self, argv):
        captured = {}
        with mock.patch.object(cli, "_run_server", lambda a: captured.update(a=a)):
            cli.main(argv)
        return captured["a"].autoload

    def test_autoload_flag_true(self):
        self.assertIs(self._autoload(["server", "--autoload"]), True)

    def test_no_autoload_flag_false(self):
        self.assertIs(self._autoload(["server", "--no-autoload"]), False)

    def test_autoload_absent_is_none(self):
        self.assertIsNone(self._autoload(["server"]))

    def test_device_map_flag_removed(self):
        with self.assertRaises(SystemExit):
            cli.main(["server", "--device-map", "org/a=NPU"])

    def test_preload_flag_removed(self):
        with self.assertRaises(SystemExit):
            cli.main(["server", "--preload", "org/a:llm"])


if __name__ == "__main__":
    unittest.main()
