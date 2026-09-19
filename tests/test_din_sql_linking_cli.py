import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class CliTests(unittest.TestCase):
    def cli(self):
        try:
            from scripts.rc_evaluation.din_sql_linking import cli
        except ImportError as exc:
            self.fail(f"DIN Linking CLI is missing: {exc}")
        return cli

    def test_status_and_verify_do_not_load_environment_or_run_models(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp) / "batch"
            for command, target, expected in (
                ("status", "status", {"state": "ok"}),
                ("verify", "verify_batch", {"ok": True}),
            ):
                output = io.StringIO()
                with patch.object(cli, target, return_value=expected) as called, contextlib.redirect_stdout(output):
                    cli.main([command, "--batch", str(batch)])
                called.assert_called_once_with(batch)
                self.assertEqual(json.loads(output.getvalue()), expected)

    def test_rerun_requires_explicit_group_and_ids(self):
        cli = self.cli()
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                cli.main(["rerun", "--batch", "/tmp/batch"])


if __name__ == "__main__":
    unittest.main()
