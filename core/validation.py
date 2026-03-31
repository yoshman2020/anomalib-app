import streamlit as st

from core import constants


def is_input_ok(has_abnormal: bool) -> bool:
    """
    Checks if the required input images and model selections are present in the Streamlit session state.

    Parameters:
        has_abnormal (bool): Indicates if abnormal images are present (not used in this function).

    Returns:
        bool: True if all required inputs are provided; False otherwise. Displays error messages in the Streamlit UI if inputs are missing.
    """
    print("is_input_ok called")
    if st.session_state["chk_model_file"]:
        # モデルファイル使用
        if st.session_state["model_file"] is None:
            st.error("モデルを選択してください。", icon="❌")
            return False
        if (
            st.session_state["test_images"] is None
            or len(st.session_state["test_images"]) == 0
        ):
            st.error("検査画像を選択してください。", icon="❌")
            return False
        return True

    if (
        st.session_state["train_images"] is None
        or len(st.session_state["train_images"]) == 0
    ):
        st.error("正常画像を選択してください。", icon="❌")
        return False
    if (
        st.session_state["test_images"] is None
        or len(st.session_state["test_images"]) == 0
    ):
        st.error("検査画像を選択してください。", icon="❌")
        return False
    if (
        0 < len(constants.MODEL_BACKBONES[st.session_state["model_name"]])
        and st.session_state["backbone"] is None
    ):
        st.error("モデルを選択してください。", icon="❌")
        return False
    if has_abnormal and (
        st.session_state["mask_images"] is None
        or len(st.session_state["abnormal_images"])
        != len(st.session_state["mask_images"])
    ):
        st.error("異常画像とマスク画像はセットで選択してください。", icon="❌")
        return False
    print("is_input_ok finished")
    return True
