from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from core import constants


def disp_train_images(
    images: list[UploadedFile], train_images_placeholder
) -> None:
    """Display the training images in a grid.

    Args:
        images (list[UploadedFile]): List of uploaded training images.
    """
    print("disp_train_images called")
    if images:
        with train_images_placeholder.container(height=constants.COLUMN_HEIGHT):
            st.header("正常画像")
            # Create a grid of images
            select_column_count = st.session_state["column_count"]
            cols = st.columns(select_column_count)
            for i, image in enumerate(images):
                cols[i % select_column_count].image(
                    image,
                    caption=f"{image.name}",
                    width="stretch",
                )
    print("disp_train_images finished")


def disp_session_images(
    result_images_placeholder,
    train_images_placeholder,
    download_button_placeholder,
):
    """
    Displays training and test images, heat maps, and inspection results from the Streamlit session state.

    This function performs the following:
    - Displays training images using `disp_train_images`.
    - Retrieves test images, heat maps, result strings, and threshold information from the Streamlit session state.
    - Shows the inspection threshold information.
    - Arranges test images, their corresponding heat maps, and result messages in a 3-column grid layout.
    - For each test image:
        - Displays the original image.
        - Displays the corresponding heat map.
        - Shows the inspection result as a success or error message, depending on the result content.

    Assumes the following keys exist in `st.session_state`:
        - "train_images": List of training images.
        - "heat_maps": List of heat map images.
        - "test_image_path": List of test image paths.
        - "str_results": List of result strings for each test image.
        - "str_threshold": String describing the threshold used for inspection.

    Requires a global `result_images_placeholder` for displaying the results section.
    """
    print("disp_session_images called")
    disp_train_images(
        st.session_state["train_images"], train_images_placeholder
    )

    test_images = st.session_state["test_images"]
    heat_maps = st.session_state["heat_maps"]
    test_image_path = st.session_state["test_image_path"]
    str_results = st.session_state["str_results"]
    str_threshold = st.session_state["str_threshold"]
    select_column_count = st.session_state["column_count"]
    group_size = 3  # 1セット3列
    # 1行あたりの結果数
    items_per_row = select_column_count // group_size

    with result_images_placeholder.container(height=300):
        st.header("検査結果")
        st.info(str_threshold)
        # Create a grid of images
        for i in range(0, len(test_images), items_per_row):
            cols = st.columns(select_column_count)
            for j in range(items_per_row):
                idx = i + j
                if idx >= len(test_images):
                    break

                image = test_images[idx]
                heat_map = heat_maps[idx]
                image_path = test_image_path[idx]
                result = str_results[idx]
                base_col = j * group_size

                cols[base_col].image(image, width="stretch")
                cols[base_col + 1].image(heat_map, width="stretch")
                cols[base_col + 2].write(image_path)
                with cols[base_col + 2]:
                    if "正常" in result:
                        st.success(result)
                    else:
                        st.error(result)

    with download_button_placeholder.container():
        # 保存ボタン
        zip_path = Path(constants.RESULT_PATH) / "result.zip"
        if zip_path.exists():
            st.download_button(
                "結果保存",
                data=zip_path.read_bytes(),
                file_name="result.zip",
                on_click="ignore",
            )

        # モデル保存ボタン
        if constants.MODEL_PATH.exists():
            st.download_button(
                "モデル保存",
                data=constants.MODEL_PATH.read_bytes(),
                file_name="model.pt",
                on_click="ignore",
            )

    st.html(
        """
        <script>
        setTimeout(() => {
            const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
            if (tabs.length > 1) {
                tabs[1].click();
            }
        }, 500);
        </script>
        """,
        unsafe_allow_javascript=True,
    )

    print("disp_session_images finished")
