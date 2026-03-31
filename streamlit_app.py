import streamlit as st

from core.pipeline import main_page
from ui.conditions import configure_conditions
from ui.footer import disp_footer
from ui.metrics_ui import disp_metrix_about_button

st.set_page_config(
    page_title="AI異常検知",
    page_icon=":control_knobs:",
    layout="wide",
)

st.markdown("# AI異常検知")

tab_conditions, tab_results, tab_metrics = st.tabs(
    ["検査条件", "検査結果", "パフォーマンス"]
)

st.session_state["tabs"] = {
    "conditions": tab_conditions,
    "results": tab_results,
    "metrics": tab_metrics,
}

with tab_results:
    column_count = st.number_input(
        "画像表示列数",
        value=6,
        min_value=1,
        max_value=10,
        step=1,
        key="column_count",
        help="画像表示の列数を指定します。",
        width=200,
    )
    train_images_placeholder = st.empty()
    result_images_placeholder = st.empty()
    download_button_placeholder = st.empty()

with tab_metrics:
    col1, col2, col3 = st.columns(3)
    with col1:
        confutsion_matrix_placeholder = st.empty()
        confutsion_matrix_container = confutsion_matrix_placeholder.container(
            horizontal_alignment="right"
        )
    with col2:
        auroc_placeholder = st.empty()
        auroc_container = auroc_placeholder.container(
            horizontal_alignment="right"
        )
    with col3:
        aupimo_placeholder = st.empty()
        aupimo_container = aupimo_placeholder.container(
            horizontal_alignment="right"
        )

# メトリクスのコンテナ
metrix_containers = [
    confutsion_matrix_container,
    auroc_container,
    aupimo_container,
]


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the tab_conditions configuration and the main page layout.
    It retrieves the user inputs from the tab_conditions, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    submitted = configure_conditions(tab_conditions)
    disp_metrix_about_button(tab_metrics, metrix_containers)
    main_page(
        submitted,
        result_images_placeholder,
        train_images_placeholder,
        download_button_placeholder,
        tab_metrics,
        metrix_containers,
    )
    disp_footer()


if __name__ == "__main__":
    main()
