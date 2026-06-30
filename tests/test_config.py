"""Tests for rosi.config — Pydantic models, YAML loading, merge overrides."""

import pytest
import yaml
from pydantic import ValidationError

from rosi.config import load_config_from_yaml, merge_config_with_overrides

# ── helpers ────────────────────────────────────────────────────────────────────


def _valid_yaml(tmp_path, overrides=None):
    """Write a minimal valid config YAML and return its path."""
    data = {
        "sample_rate": 44100,
        "duration": 1.0,
        "speed_of_sound": 343.0,
        "rpm": 60.0,
        "mic_positions_csv": str(tmp_path / "mics.csv"),
        "scan_grid": {"r_max": 1.0, "n_r": 8, "n_theta": 8},
        "fft_size": 1024,
        "overlap": 0.5,
        "f_min": 100.0,
        "f_max": 5000.0,
        "output_image": str(tmp_path / "out.png"),
        "sources": [
            {"R": 0.5, "phi0": 0.0, "freq": 3000, "amplitude": 1.0, "phase": 0.0}
        ],
    }
    if overrides:
        data.update(overrides)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


@pytest.fixture()
def mic_csv(tmp_path):
    """Create a minimal mics.csv so file-existence validators pass."""
    p = tmp_path / "mics.csv"
    p.write_text("0,0,1.5\n1,0,1.5\n0,1,1.5\n")
    return p


# ── load_config_from_yaml ─────────────────────────────────────────────────────


class TestLoadConfig:
    def test_loads_valid_yaml(self, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path)
        cfg = load_config_from_yaml(p)
        assert cfg.sample_rate == 44100
        assert cfg.rpm == 60.0
        assert len(cfg.sources) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config_from_yaml(str(tmp_path / "nope.yaml"))

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config_from_yaml(str(p))


# ── fft_size validator ────────────────────────────────────────────────────────


class TestFftSize:
    @pytest.mark.parametrize("val", [63, 100, 0, 65])
    def test_rejects_invalid(self, val, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path, {"fft_size": val})
        with pytest.raises(ValidationError):
            load_config_from_yaml(p)

    @pytest.mark.parametrize("val", [64, 128, 256, 512, 1024])
    def test_accepts_valid(self, val, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path, {"fft_size": val})
        cfg = load_config_from_yaml(p)
        assert cfg.fft_size == val


# ── overlap validator ─────────────────────────────────────────────────────────


class TestOverlap:
    @pytest.mark.parametrize("val", [-0.1, 1.0, 1.5])
    def test_rejects_out_of_range(self, val, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path, {"overlap": val})
        with pytest.raises(ValidationError):
            load_config_from_yaml(p)

    @pytest.mark.parametrize("val", [0.0, 0.5, 0.99])
    def test_accepts_valid(self, val, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path, {"overlap": val})
        cfg = load_config_from_yaml(p)
        assert cfg.overlap == val


# ── f_min / f_max ─────────────────────────────────────────────────────────────


class TestFreqRange:
    def test_f_max_must_exceed_f_min(self, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path, {"f_min": 5000, "f_max": 1000})
        with pytest.raises(ValidationError):
            load_config_from_yaml(p)

    def test_equal_raises(self, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path, {"f_min": 3000, "f_max": 3000})
        with pytest.raises(ValidationError):
            load_config_from_yaml(p)


# ── simulation mode required fields ───────────────────────────────────────────


class TestSimMode:
    def test_missing_sample_rate_raises(self, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path, {"sample_rate": None})
        with pytest.raises(ValidationError):
            load_config_from_yaml(p)

    def test_missing_sources_raises(self, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path, {"sources": []})
        with pytest.raises(ValidationError):
            load_config_from_yaml(p)


# ── wav_file / sources mutual exclusion ───────────────────────────────────────


class TestWavFileExclusion:
    def test_wav_file_with_sources_raises(self, tmp_path, mic_csv):
        wav = tmp_path / "test.wav"
        wav.write_bytes(
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        p = _valid_yaml(tmp_path, {"wav_file": str(wav)})
        with pytest.raises(ValidationError):
            load_config_from_yaml(p)


# ── file existence validators ─────────────────────────────────────────────────


class TestFileExistence:
    def test_mic_csv_missing_raises(self, tmp_path):
        p = _valid_yaml(tmp_path, {"mic_positions_csv": str(tmp_path / "nope.csv")})
        with pytest.raises(ValidationError, match="not found"):
            load_config_from_yaml(p)

    def test_output_parent_missing_raises(self, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path, {"output_image": "/nonexistent/dir/out.png"})
        with pytest.raises(ValidationError, match="does not exist"):
            load_config_from_yaml(p)


# ── merge_config_with_overrides ───────────────────────────────────────────────


class TestMergeOverrides:
    def test_nested_scan_grid_key(self, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path)
        cfg = load_config_from_yaml(p)
        merged = merge_config_with_overrides(cfg, {"r_max": 2.0, "n_r": 10})
        assert merged.scan_grid.r_max == 2.0
        assert merged.scan_grid.n_r == 10
        assert merged.scan_grid.n_theta == 8  # unchanged

    def test_none_values_ignored(self, tmp_path, mic_csv):
        p = _valid_yaml(tmp_path)
        cfg = load_config_from_yaml(p)
        merged = merge_config_with_overrides(cfg, {"r_max": None, "fft_size": 2048})
        assert merged.scan_grid.r_max == 1.0  # unchanged
        assert merged.fft_size == 2048

    def test_wav_input_dict_merge(self, tmp_path, mic_csv):
        wav = tmp_path / "test.wav"
        wav.write_bytes(
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        p = _valid_yaml(tmp_path, {"wav_input": {"path": str(wav), "tacho_channel": 2}})
        cfg = load_config_from_yaml(p)
        merged = merge_config_with_overrides(cfg, {"wav_input": {"tacho_channel": 5}})
        assert merged.wav_input.tacho_channel == 5
        assert merged.wav_input.path == str(wav)
