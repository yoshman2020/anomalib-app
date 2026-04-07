import logging

import streamlit as st

from anomalib_app.core import constants
from anomalib_app.core.config_utils import get_config
from anomalib_app.core.pipeline import main_page
from anomalib_app.ui.conditions import configure_conditions
from anomalib_app.ui.footer import disp_footer
from anomalib_app.ui.metrics_ui import disp_metrics_about_button

logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the tab_conditions configuration and the main page layout.
    It retrieves the user inputs from the tab_conditions, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    config = get_config()

    level_str = config.get("log_level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

    APP_TITLE = config.get("app_title", "AI異常検知")

    logger.info("Initializing the Streamlit application")

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":control_knobs:",
        layout="wide",
    )

    st.markdown(
        f"""
        <style>
        div[data-testid="stMainBlockContainer"] {{
            padding-top: 3em;
        }}
        .custom-header {{
            position: fixed;
            top: 0.5rem;
            left: 1rem;
            z-index: 999999;
            font-size: 1.5rem;
            font-weight: bold;
        }}
        </style>
        <div class="custom-header">{APP_TITLE}</div>
        """,
        unsafe_allow_html=True,
    )

    tab_conditions, tab_results, tab_metrics = st.tabs(
        ["検査条件", "検査結果", "パフォーマンス"]
    )

    st.session_state["tabs"] = {
        "conditions": tab_conditions,
        "results": tab_results,
        "metrics": tab_metrics,
    }

    with tab_results:
        _ = st.number_input(
            "画像表示列数",
            value=9,
            min_value=1,
            max_value=10,
            step=1,
            key="column_count",
            help="画像表示の列数を指定します。",
            width=200,
        )
        train_images_placeholder = st.empty()
        train_images_container = train_images_placeholder.container(
            key="train_images_container", height=constants.COLUMN_HEIGHT
        )
        result_images_placeholder = st.empty()
        result_images_container = result_images_placeholder.container(
            key="result_images_container", height=constants.COLUMN_HEIGHT_RESULT
        )
        download_button_placeholder = st.empty()
        download_button_container = download_button_placeholder.container(
            key="download_button_container"
        )

    with tab_metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            confutsion_matrix_placeholder = st.empty()
            confutsion_matrix_container = (
                confutsion_matrix_placeholder.container(
                    key="confutsion_matrix_container",
                    horizontal_alignment="right",
                )
            )

        with col2:
            auroc_placeholder = st.empty()
            auroc_container = auroc_placeholder.container(
                key="auroc_container", horizontal_alignment="right"
            )
        with col3:
            aupimo_placeholder = st.empty()
            aupimo_container = aupimo_placeholder.container(
                key="aupimo_container", horizontal_alignment="right"
            )

    # メトリクスのコンテナ
    metrics_containers = [
        confutsion_matrix_container,
        auroc_container,
        aupimo_container,
    ]

    logger.debug("Start configuring conditions")
    submitted = configure_conditions(tab_conditions)
    disp_metrics_about_button(tab_metrics, metrics_containers)
    logger.debug("Start rendering the main page")
    main_page(
        submitted,
        train_images_container,
        result_images_container,
        download_button_container,
        tab_metrics,
        metrics_containers,
    )
    disp_footer()
    logger.info("End of the Streamlit display function")
