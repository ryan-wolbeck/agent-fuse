import pytest

from agent_fuse.config import ConfigError, default_config, load_config


def test_load_config_none_with_nothing_discoverable_returns_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("agent_fuse.config.find_config_file", lambda *a, **k: None)
    config = load_config(None)
    assert config.response_rate.max_responses == 100


def test_default_config_matches_documented_conservative_defaults() -> None:
    config = default_config()
    assert config.response_rate.enabled is True
    assert config.response_rate.window_seconds == 600
    assert config.response_rate.max_responses == 100
    assert config.token_rate.max_input_tokens == 2_000_000
    assert config.context_replay.minimum_input_tokens == 100_000
    assert config.context_replay.max_cached_ratio == 0.90
    assert config.repeated_tool_call.window_seconds == 300
    assert config.repeated_tool_call.max_repetitions == 30


def test_load_config_explicit_missing_path_is_a_hard_error(tmp_path) -> None:
    """An explicitly-specified --config path that doesn't exist is a user
    error, not a signal to silently fall back to defaults."""
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_load_config_valid_overrides(tmp_path) -> None:
    path = tmp_path / ".agent-fuse.yaml"
    path.write_text(
        """
version: 1
response_rate:
  enabled: false
repeated_tool_call:
  max_repetitions: 5
"""
    )
    config = load_config(path)
    assert config.response_rate.enabled is False
    assert config.repeated_tool_call.max_repetitions == 5
    # untouched sections keep defaults
    assert config.token_rate.max_input_tokens == 2_000_000


def test_load_config_invalid_yaml_raises(tmp_path) -> None:
    path = tmp_path / ".agent-fuse.yaml"
    path.write_text("not: [valid: yaml: at: all")
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_unknown_field_raises(tmp_path) -> None:
    path = tmp_path / ".agent-fuse.yaml"
    path.write_text("totally_unknown_field: true")
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_non_mapping_raises(tmp_path) -> None:
    path = tmp_path / ".agent-fuse.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_max_window_seconds_ignores_disabled_rules() -> None:
    config = default_config()
    config.response_rate.enabled = False
    config.token_rate.enabled = False
    config.repeated_tool_call.window_seconds = 42
    assert config.max_window_seconds() == 42
