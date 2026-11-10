"""
Torch-free configuration helpers shared by the training scripts and the
FastAPI service.

Keeping YAML loading (with ``_base_`` inheritance) free of heavy imports lets
the inference API read the *same* merged config as training without importing
PyTorch or TensorBoard.
"""

import sys
from pathlib import Path

import yaml

__all__ = ["configure_utf8_console", "load_config"]


def configure_utf8_console() -> None:
    """
    Force UTF-8 output on stdout/stderr.

    Windows consoles default to cp1252, which raises ``UnicodeEncodeError``
    for characters such as the check/cross marks used in this project's
    progress output. Calling this once per entry point keeps logging safe.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Detached/redirected streams without a text encoding.
            pass


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration with ``_base_`` inheritance support.

    If the config contains a ``_base_`` key, that base config is loaded first
    and merged recursively. Child config values override parent values.

    Args:
        config_path: Path to the YAML config file (absolute or relative to cwd)

    Returns:
        Merged configuration dictionary

    Example:
        >>> config = load_config('configs/efficientnet_b0.yaml')
        >>> print(config['data']['img_size'])  # From base.yaml
        224
        >>> print(config['model']['name'])  # From efficientnet_b0.yaml
        'efficientnet_b0'
    """
    config_path = Path(config_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    # Handle base config inheritance
    if '_base_' in config:
        base_path = config.pop('_base_')

        # Resolve base path relative to current config's directory
        if not Path(base_path).is_absolute():
            base_path = config_path.parent / base_path

        # Recursively load base config (supports chained inheritance)
        base_config = load_config(str(base_path))

        # Merge: base config first, then override with child config
        return _merge_dicts(base_config, config)

    return config


def _merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries.

    Values in 'override' take precedence over 'base'.
    Nested dicts are merged recursively; other values are replaced.
    """
    merged = base.copy()

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged
