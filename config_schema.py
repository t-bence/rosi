"""
config_schema.py — Pydantic models for ROSI configuration validation.
"""

from pathlib import Path

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


class ROSIConfig(BaseModel):
    """Complete ROSI configuration."""

    # Signal parameters — required for simulation mode, derived from WAV in measurement mode
    sample_rate: int | None = Field(default=None, gt=0, description="Sample rate [Hz]")
    duration: float | None = Field(default=None, gt=0, description="Signal duration [seconds]")
    speed_of_sound: float = Field(gt=0, description="Speed of sound [m/s]")
    rpm: float = Field(ge=0, description="Rotor speed [rev/min]")

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

    # Measurement mode: multi-channel WAV file replaces simulation
    wav_file: str | None = Field(default=None, description="Path to multi-channel WAV file")

    # Sources — required for simulation mode, unused in measurement mode
    sources: list[SourceConfig] = Field(
        default_factory=list, description="Rotating sources (simulation mode only)"
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
    def f_max_greater_than_f_min(self) -> "ROSIConfig":
        """Validate f_max > f_min."""
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be > f_min")
        return self

    @field_validator("wav_file", mode="before")
    @classmethod
    def wav_file_must_exist(cls, v: str | None) -> str | None:
        if v is None:
            return v
        path = Path(v)
        if not path.exists():
            raise ValueError(f"WAV file not found: {path}")
        return v

    @model_validator(mode="after")
    def check_mode_completeness(self) -> "ROSIConfig":
        if self.wav_file is not None:
            # Measurement mode: sources must not be set
            if self.sources:
                raise ValueError("Cannot specify both wav_file and sources")
        else:
            # Simulation mode: sample_rate, duration, and sources are required
            if self.sample_rate is None:
                raise ValueError("sample_rate is required in simulation mode (no wav_file)")
            if self.duration is None:
                raise ValueError("duration is required in simulation mode (no wav_file)")
            if not self.sources:
                raise ValueError("At least one source is required in simulation mode (no wav_file)")
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

        # Handle nested keys
        if key in ["r_max", "n_r", "n_theta"]:
            if "scan_grid" not in config_dict:
                config_dict["scan_grid"] = {}
            config_dict["scan_grid"][key] = value
        else:
            config_dict[key] = value

    return ROSIConfig(**config_dict)
