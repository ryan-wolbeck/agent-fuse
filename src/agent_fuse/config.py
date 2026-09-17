"""Configuration loading for Agent Fuse.

Defaults are deliberately conservative starting points, not scientifically
validated universal limits -- see README "Configuration" section. Users are
expected to tune thresholds for their own workloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

CONFIG_FILENAME = ".agent-fuse.yaml"


class ConfigError(Exception):
    """Raised when a config file exists but cannot be parsed/validated."""


class ResponseRateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    window_seconds: int = Field(default=600, gt=0)
    max_responses: int = Field(default=100, gt=0)


class TokenRateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    window_seconds: int = Field(default=600, gt=0)
    max_input_tokens: int = Field(default=2_000_000, gt=0)


class ContextReplayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    minimum_input_tokens: int = Field(default=100_000, gt=0)
    max_cached_ratio: float = Field(default=0.90, gt=0, le=1)


class RepeatedToolCallConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    window_seconds: int = Field(default=300, gt=0)
    max_repetitions: int = Field(default=30, gt=0)


class FuseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    response_rate: ResponseRateConfig = Field(default_factory=ResponseRateConfig)
    token_rate: TokenRateConfig = Field(default_factory=TokenRateConfig)
    context_replay: ContextReplayConfig = Field(default_factory=ContextReplayConfig)
    repeated_tool_call: RepeatedToolCallConfig = Field(default_factory=RepeatedToolCallConfig)

    def max_window_seconds(self) -> int:
        """Largest rolling window any enabled rule needs kept in memory."""
        windows = [
            self.response_rate.window_seconds if self.response_rate.enabled else 0,
            self.token_rate.window_seconds if self.token_rate.enabled else 0,
            self.repeated_tool_call.window_seconds if self.repeated_tool_call.enabled else 0,
        ]
        return max(windows) if any(windows) else 600


def default_config() -> FuseConfig:
    return FuseConfig()


def find_config_file(start: Path | None = None) -> Path | None:
    """Search cwd, then walk up to filesystem root, then user config dir."""
    cwd = start or Path.cwd()
    for directory in [cwd, *cwd.parents]:
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate

    user_config = Path.home() / ".config" / "agent-fuse" / "config.yaml"
    if user_config.is_file():
        return user_config

    return None


def load_config(path: Path | None = None) -> FuseConfig:
    """Load config from an explicit path, discovered file, or defaults.

    Raises ConfigError if a config file exists but is invalid -- Agent Fuse
    never silently falls back to defaults when the user *did* provide a
    config, since that could mask thresholds the user believes are active.
    """
    resolved = path or find_config_file()
    if resolved is None:
        return default_config()

    try:
        raw_text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read config file {resolved}: {exc}") from exc

    try:
        data: Any = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {resolved}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config file {resolved} must contain a YAML mapping at the top level")

    try:
        return FuseConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {resolved}:\n{exc}") from exc
