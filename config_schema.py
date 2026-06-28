"""
config_schema.py — Pydantic models for ROSI configuration validation.
"""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ScanGridConfig(BaseModel):
    """Scan grid configuration."""

    r_max: float = Field(gt=0, description="Outer radius of scan area [m]")
    n_r: int = Field(ge=1, description="Number of radial steps")
    n_theta: int = Field(ge=1, description="Number of angular steps")


class SourceConfig(BaseModel):
    """Individual rotating source configuration."""

    R: float = Field(ge=0, description="Radius from rotation axis [m]")
    phi0: float = Field(description="Initial angular position [radians]")
    freq: float = Field(gt=0, description="Tone frequency [Hz]")
    amplitude: float = Field(description="Source strength (arbitrary units)")
    phase: float = Field(default=0.0, description="Initial phase [radians]")


class WavInputConfig(BaseModel):
    """Configuration for loading signals from a WAV file with an optical tachometer."""

    path: str = Field(description="Path to the multi-channel WAV file")
    tacho_channel: int = Field(
        default=0,
        description="0-based index of the tachometer channel within the WAV file",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Edge-detection threshold for the tachometer signal (auto if None)",
    )

    @field_validator("path")
    @classmethod
    def wav_must_exist(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"WAV file not found: {v}")
        return v


class ROSIConfig(BaseModel):
    """Complete ROSI configuration."""

    # Signal parameters — optional when wav_input is provided (derived from WAV)
    sample_rate: Optional[int] = Field(default=None, description="Sample rate [Hz]")
    duration: Optional[float] = Field(default=None, description="Signal duration [seconds]")
    speed_of_sound: float = Field(gt=0, description="Speed of sound [m/s]")
    rpm: Optional[float] = Field(default=None, description="Rotor speed [rev/min]")

    # WAV input (mutually exclusive with simulated sources for RPM/timing)
    wav_input: Optional[WavInputConfig] = Field(
        default=None,
        description="Load signals from a WAV file; tachometer channel provides RPM",
    )

    # Microphone array
    mic_positions_csv: str = Field(description="Path to microphone positions CSV")

    # Scan grid
    scan_grid: ScanGridConfig = Field(description="Scan grid configuration")

    # Beamforming parameters
    fft_size: int = Field(description="Welch block length [samples]")
    overlap: float = Field(ge=0, lt=1, description="Welch overlap fraction [0, 1)")
    f_min: float = Field(ge=0, description="Minimum frequency [Hz]")
    f_max: float = Field(description="Maximum frequency [Hz]")

    # Output
    output_image: str = Field(
        default="rosi_result.png", description="Output PNG filename"
    )

    # Sources (optional)
    sources: list[SourceConfig] = Field(
        default_factory=list, description="Rotating sources"
    )

    @field_validator("fft_size")
    @classmethod
    def fft_size_must_be_power_of_2(cls, v: int) -> int:
        """Validate fft_size is a power of 2."""
        if v < 64:
            raise ValueError("fft_size must be >= 64")
        if (v & (v - 1)) != 0:
            raise ValueError("fft_size must be a power of 2")
        return v

    @model_validator(mode="after")
    def check_mode_and_required_fields(self) -> "ROSIConfig":
        """
        In simulation mode (no wav_input), sample_rate, duration, and rpm must
        all be specified with valid positive values.  In WAV mode they are
        derived from the WAV file and may be omitted from the config.
        """
        if self.wav_input is None:
            missing = [
                name
                for name, val in [
                    ("sample_rate", self.sample_rate),
                    ("duration", self.duration),
                    ("rpm", self.rpm),
                ]
                if val is None
            ]
            if missing:
                raise ValueError(
                    f"The following fields are required when wav_input is not set: "
                    + ", ".join(missing)
                )
            if self.sample_rate is not None and self.sample_rate <= 0:
                raise ValueError("sample_rate must be > 0")
            if self.duration is not None and self.duration <= 0:
                raise ValueError("duration must be > 0")
            if self.rpm is not None and self.rpm < 0:
                raise ValueError("rpm must be >= 0")

        if self.f_max <= self.f_min:
            raise ValueError("f_max must be > f_min")
        return self

    @field_validator("mic_positions_csv", mode="before")
    @classmethod
    def mic_csv_must_exist(cls, v: str) -> str:
        """Validate microphone CSV file exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Microphone CSV not found: {path}")
        return v

    @field_validator("output_image", mode="before")
    @classmethod
    def output_path_parent_must_exist(cls, v: str) -> str:
        """Validate output path parent directory exists."""
        path = Path(v)
        parent = path.parent if path.parent != Path() else Path(".")
        if not parent.exists():
            raise ValueError(f"Output directory does not exist: {parent}")
        return v


def load_config_from_yaml(config_path: str) -> ROSIConfig:
    """Load and validate config from YAML file."""
    from pathlib import Path

    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Config file is empty: {path}")

    return ROSIConfig(**data)


def merge_config_with_overrides(config: ROSIConfig, overrides: dict) -> ROSIConfig:
    """
    Merge CLI overrides into config.
    Converts nested keys like 'r_max' to 'scan_grid.r_max'.
    """
    config_dict = config.model_dump()

    for key, value in overrides.items():
        if value is None:
            continue

        # Handle nested scan_grid keys
        if key in ["r_max", "n_r", "n_theta"]:
            if "scan_grid" not in config_dict:
                config_dict["scan_grid"] = {}
            config_dict["scan_grid"][key] = value
        # Handle nested wav_input keys (dict merge)
        elif key == "wav_input" and isinstance(value, dict):
            if config_dict.get("wav_input") is None:
                config_dict["wav_input"] = value
            else:
                config_dict["wav_input"].update(value)
        else:
            config_dict[key] = value

    return ROSIConfig(**config_dict)
