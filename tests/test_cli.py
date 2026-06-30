"""Tests for rosi.cli — CLI entry points (in-process, no heavy compute)."""

import sys
from pathlib import Path

import pytest
import yaml

from rosi.cli import cmd_generate_array, cmd_run, cmd_validate, main


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_minimal_config(tmp_path):
    """Write a minimal valid config YAML with a real mics.csv and return its path."""
    mics = tmp_path / "mics.csv"
    mics.write_text("0,0,1.5\n1,0,1.5\n0,1,1.5\n")
    cfg = {
        "sample_rate": 44100,
        "duration": 0.1,
        "speed_of_sound": 343.0,
        "rpm": 60.0,
        "mic_positions_csv": str(mics),
        "scan_grid": {"r_max": 0.5, "n_r": 4, "n_theta": 4},
        "fft_size": 128,
        "overlap": 0.5,
        "f_min": 100.0,
        "f_max": 5000.0,
        "output_image": str(tmp_path / "out.png"),
        "sources": [{"R": 0.3, "phi0": 0.0, "freq": 1000, "amplitude": 1.0, "phase": 0.0}],
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


class _Args:
    """Minimal argparse.Namespace stand-in."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── cmd_validate ──────────────────────────────────────────────────────────────

class TestCmdValidate:
    def test_valid_config_returns_0(self, tmp_path):
        cfg = _make_minimal_config(tmp_path)
        args = _Args(config=str(cfg))
        assert cmd_validate(args) == 0

    def test_bad_config_returns_1(self, tmp_path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("sample_rate: 'not_a_number'\n")
        args = _Args(config=str(cfg))
        assert cmd_validate(args) == 1

    def test_missing_config_returns_1(self, tmp_path):
        args = _Args(config=str(tmp_path / "nonexistent.yaml"))
        assert cmd_validate(args) == 1


# ── cmd_run --dry-run ─────────────────────────────────────────────────────────

class TestCmdRun:
    def test_dry_run_returns_0(self, tmp_path, capsys):
        cfg = _make_minimal_config(tmp_path)
        args = _Args(
            config=str(cfg),
            output=None,
            dry_run=True,
            no_plot=True,
            sample_rate=None,
            duration=None,
            speed_of_sound=None,
            rpm=None,
            mic_positions_csv=None,
            r_max=None,
            n_r=None,
            n_theta=None,
            fft_size=None,
            overlap=None,
            f_min=None,
            f_max=None,
            wav_file=None,
            tacho_channel=None,
        )
        assert cmd_run(args) == 0
        captured = capsys.readouterr()
        assert "Merged configuration" in captured.out
        assert "Dry run complete" in captured.out


# ── cmd_generate_array ────────────────────────────────────────────────────────

class TestCmdGenerateArray:
    def test_writes_csv(self, tmp_path):
        out = tmp_path / "arr.csv"
        args = _Args(output=str(out), N=6, R=1.5, Z=2.0, force=True)
        assert cmd_generate_array(args) == 0
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        # header + 6 data rows
        assert len(lines) == 7

    def test_returns_0(self, tmp_path):
        out = tmp_path / "arr.csv"
        args = _Args(output=str(out), N=4, R=1.0, Z=0.5, force=True)
        assert cmd_generate_array(args) == 0


# ── main() dispatch ───────────────────────────────────────────────────────────

class TestMainDispatch:
    def test_validate_subcommand(self, tmp_path):
        cfg = _make_minimal_config(tmp_path)
        test_argv = ["rosi", "validate", "--config", str(cfg)]
        orig = sys.argv
        try:
            sys.argv = test_argv
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        finally:
            sys.argv = orig

    def test_generate_array_subcommand(self, tmp_path):
        out = tmp_path / "gen.csv"
        test_argv = ["rosi", "generate-array", "-N", "8", "-R", "2.0", "-Z", "1.0", "-o", str(out), "-f"]
        orig = sys.argv
        try:
            sys.argv = test_argv
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
            assert out.exists()
        finally:
            sys.argv = orig

    def test_dry_run_subcommand(self, tmp_path):
        cfg = _make_minimal_config(tmp_path)
        test_argv = ["rosi", "run", "--config", str(cfg), "--dry-run", "--no-plot"]
        orig = sys.argv
        try:
            sys.argv = test_argv
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0
        finally:
            sys.argv = orig