"""Config module: schema defaults, cfg_hash, transform presets, structured configs."""

from src.config.hash import cfg_hash, EXCLUDED_KEYS
from src.config.schema import SCHEMA_VERSION, apply_schema_defaults
from src.config.structured import register_configs, ConTopoConfig
