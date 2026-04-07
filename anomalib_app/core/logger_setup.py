import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logger():
    # exe / python 両対応
    base_dir = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )

    log_dir = base_dir / "log"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "app.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 日次ローテーション（30日保持）
    handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )

    # ファイル名に日付を付ける
    handler.suffix = "%Y-%m-%d.log"

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    # コンソールにも出したい場合
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
