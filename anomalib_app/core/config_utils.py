import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_config():
    config_path = (
        Path(__file__).resolve().parent.parent.parent / "config.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
