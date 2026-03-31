import pandas as pd
import streamlit as st

import core.constants as constants


def configure_conditions(tab_conditions) -> bool:
    """
    Configures the Streamlit tab_conditions UI for anomaly detection settings.

    The tab_conditions includes:
        - File uploaders for training and test images.
        - Selectboxes for choosing the anomaly detection model and its backbone.
        - Checkbox and number input for threshold configuration (with auto/manual option).
        - Number inputs for image resizing and training epochs.
        - Submit button to start the inspection process.
        - Buttons and dialogs providing information about available models and backbones.

    Returns:
        bool: True if the form is submitted ("検査開始" button pressed), otherwise False.
    """
    print("configure_conditions called")
    with tab_conditions:
        st.markdown("## :level_slider: 検査条件")

        # モデルファイル
        col_model_file_1, col_model_file_2 = st.columns([2, 8])
        with col_model_file_1:
            chk_model_file = st.checkbox(
                "モデルファイル使用",
                key="chk_model_file",
                value=False,
                help="学習済みのモデルファイルを使用します。",
            )
        with col_model_file_2:
            st.file_uploader(
                "モデル選択",
                type=["pt"],
                accept_multiple_files=False,
                key="model_file",
                help="学習済みモデルファイルを選択してください。",
                disabled=not chk_model_file,
            )

        # 画像ファイル
        col_file_1, col_file_2 = st.columns(2)
        with col_file_1:
            st.file_uploader(
                "正常画像",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                key="train_images",
                help="学習に使用する正常画像を選択してください。",
                disabled=chk_model_file,
            )
        with col_file_2:
            st.file_uploader(
                "検査画像",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                key="test_images",
                help="検査する画像を選択してください。",
            )

        # 検査手法、モデル（バックボーン）
        col_name_backbone_1, col_name_backbone_2 = st.columns(2)
        with col_name_backbone_1:
            # 検査手法
            col_model_name_1, col_model_name_2 = st.columns(
                [9, 1], vertical_alignment="bottom"
            )
            with col_model_name_1:
                model_name = st.selectbox(
                    "検査手法",
                    options=constants.MODEL_NAMES,
                    index=12,
                    key="model_name",
                    disabled=chk_model_file,
                )

            @st.dialog("検査手法について", width="large")
            def about_model_name():
                df = pd.DataFrame(constants.ABOUT_MODEL_NAMES)
                st.table(df)

            with col_model_name_2:
                if st.button("?", key="button_about_model_name"):
                    about_model_name()
        with col_name_backbone_2:
            # モデル（バックボーン）
            col_backbone_1, col_backbone_2 = st.columns(
                [9, 1], vertical_alignment="bottom"
            )
            with col_backbone_1:
                st.selectbox(
                    "モデル",
                    options=constants.MODEL_BACKBONES[model_name],
                    index=0,
                    key="backbone",
                    disabled=chk_model_file,
                )

            @st.dialog("モデルについて", width="large")
            def about_backbone():
                df = pd.DataFrame(constants.ABOUT_BACKBONE)
                st.table(df)

            with col_backbone_2:
                if st.button("?", key="button_about_backbone"):
                    about_backbone()

        # しきい値、縮小サイズ、学習サイズ
        col_sizes_1, col_sizes_2, col_sizes_3, col_sizes_4 = st.columns(
            [1, 3, 3, 3]
        )
        with col_sizes_1:
            threshold_auto = st.checkbox(
                "自動しきい値",
                key="threshold_auto",
                value=True,
                help="正常画像をもとにしきい値を推定します。",
                disabled=chk_model_file,
            )
        with col_sizes_2:
            st.number_input(
                "しきい値",
                format="%0.5f",
                key="threshold",
                disabled=chk_model_file or threshold_auto,
                help="指定したしきい値で正常／異常を判定します。",
            )
        with col_sizes_3:
            st.number_input(
                "縮小サイズ",
                value=128,
                min_value=2,
                key="image_size",
                help="学習前に画像を指定したサイズに縮小します。",
                disabled=chk_model_file,
            )
        with col_sizes_4:
            st.number_input(
                "学習回数",
                value=1,
                min_value=1,
                key="epochs",
                help="指定した回数学習します。",
                disabled=chk_model_file,
            )

        # 異常画像、マスク画像
        col_abnormal_mask_1, col_abnormal_mask_2 = st.columns(2)
        with col_abnormal_mask_1:
            st.file_uploader(
                "異常画像　※未選択可",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                key="abnormal_images",
                help="異常画像がある場合はマスク画像とセットで選択してください。未選択でも検査できます。",
                disabled=chk_model_file,
            )
        with col_abnormal_mask_2:
            st.file_uploader(
                "マスク画像　※未選択可",
                type=["png", "jpg", "jpeg", "bmp"],
                accept_multiple_files=True,
                key="mask_images",
                help="マスク画像を異常画像とセットで選択してください。未選択でも検査できます。",
                disabled=chk_model_file,
            )

        submitted = st.button(
            "検査開始", type="primary", width="content", key="submitted"
        )

    print("configure_conditions finished")
    return submitted
