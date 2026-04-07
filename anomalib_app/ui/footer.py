import logging
from pathlib import Path

import streamlit as st

from anomalib_app.core.config_utils import get_config

logger = logging.getLogger(__name__)


def disp_footer():
    """Display footer information."""
    logger.debug("disp_footer called")
    st.divider()
    config = get_config()
    company_name = config.get("company_name", "Your Company Name")
    st.caption(
        f"© 2026 {company_name}",
    )
    col1, col2 = st.columns(
        2, gap="xxsmall", vertical_alignment="center", width=650
    )
    with col1:
        st.caption(
            "This application uses third-party open-source software.",
        )
    with col2:
        # LICENSE読み込み
        PROJECT_ROOT = Path(__file__).resolve()

        # 上に登って LICENSE を探す
        for parent in PROJECT_ROOT.parents:
            license_path = parent / "LICENSE"
            if license_path.exists():
                break
        else:
            raise FileNotFoundError("LICENSE not found")

        with open(license_path, "rb") as f:
            data = f.read()
        st.download_button(
            label="LICENSE",
            data=data,
            file_name="LICENSE",
            mime="text/plain",
            type="tertiary",
        )
    logger.debug("disp_footer finished")
