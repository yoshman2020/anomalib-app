import sys
from pathlib import Path

import streamlit.web.cli as stcli


def get_app_path():
    base_path = Path(__file__).resolve().parent

    return base_path / "anomalib_app" / "streamlit_app.py"


def streamlit_run():
    app_path = get_app_path()

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
    ]
    stcli.main()


if __name__ == "__main__":
    from anomalib_app.core.logger_setup import setup_logger

    logger = setup_logger()
    logger.info("Start the Streamlit application")
    streamlit_run()
