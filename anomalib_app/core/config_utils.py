import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def get_config():

    if os.getenv("DEBUG_MODE") == "True":
        # When running in debug mode, the config file is located one level up from the current directory
        config_path = (
            Path(__file__).resolve().parent.parent.parent / "config.json"
        )
    else:
        # When running in normal mode, the config file is located two levels up from the current directory
        config_path = (
            Path(__file__).resolve().parent.parent.parent.parent / "config.json"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
