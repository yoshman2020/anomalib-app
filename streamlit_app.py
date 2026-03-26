import csv
import io
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from anomalib.data import Folder, FolderDataset
from anomalib.data.dataclasses.torch.image import ImageBatch
from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from anomalib.engine.engine import Engine
from anomalib.metrics import AUPIMO, AUROC, F1AdaptiveThreshold, F1Max, F1Score
from anomalib.models import VlmAd, WinClip
from matplotlib.ticker import MaxNLocator, PercentFormatter
from PIL import Image
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile

import constants
import models

st.set_page_config(
    page_title="AI異常検知",
    page_icon=":control_knobs:",
    layout="wide",
)

st.markdown("# AI異常検知")
tab_conditions, tab_results, tab_metrics = st.tabs(
    ["検査条件", "検査結果", "パフォーマンス"]
)


def disp_session_images():
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
    disp_train_images(st.session_state["train_images"])

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

    print("disp_session_images finished")


# 画像表示カラムの高さ
COLUMN_HEIGHT = 300
COLUMN_HEIGHT_RESULT = 500
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


def configure_conditions() -> bool:
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

        chk_model_file = st.checkbox(
            "モデルファイル使用",
            key="chk_model_file",
            value=False,
            help="学習済みのモデルファイルを使用します。",
        )
        st.file_uploader(
            "モデル選択",
            type=["pt"],
            accept_multiple_files=False,
            key="model_file",
            help="学習済みモデルファイルを選択してください。",
            disabled=not chk_model_file,
        )

        st.file_uploader(
            "正常画像",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            key="train_images",
            help="学習に使用する正常画像を選択してください。",
            disabled=chk_model_file,
        )

        st.file_uploader(
            "検査画像",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            key="test_images",
            help="検査する画像を選択してください。",
        )

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

        threshold_auto = st.checkbox(
            "自動しきい値",
            key="threshold_auto",
            value=True,
            help="正常画像をもとにしきい値を推定します。",
            disabled=chk_model_file,
        )
        st.number_input(
            "しきい値",
            format="%0.5f",
            key="threshold",
            disabled=chk_model_file or threshold_auto,
            help="指定したしきい値で正常／異常を判定します。",
        )

        st.number_input(
            "縮小サイズ",
            value=128,
            min_value=2,
            key="image_size",
            help="学習前に画像を指定したサイズに縮小します。",
            disabled=chk_model_file,
        )
        st.number_input(
            "学習回数",
            value=1,
            min_value=1,
            key="epochs",
            help="指定した回数学習します。",
            disabled=chk_model_file,
        )

        st.file_uploader(
            "異常画像　※未選択可",
            type=["png", "jpg", "jpeg", "bmp"],
            accept_multiple_files=True,
            key="abnormal_images",
            help="異常画像がある場合はマスク画像とセットで選択してください。未選択でも検査できます。",
            disabled=chk_model_file,
        )

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


def disp_metrix_about_button():
    """Displays information buttons for confusion matrix, AUROC, and AUPIMO metrics."""
    with confutsion_matrix_container:

        @st.dialog("混同行列（Confusion Matrix）とは", width="large")
        def about_confution_matrix():
            st.write(
                "混同行列（Confusion Matrix）は、分類モデルの性能を視覚的に評価するための表です。<br />",
                "以下の表は、予測されたラベルと実際のラベルの組み合わせを示しています。<br />",
                unsafe_allow_html=True,
            )
            df = pd.DataFrame(
                {
                    "実際: 異常": ["TN", "FP"],
                    "実際: 正常": ["FN", "TP"],
                },
                index=["予測: 異常", "予測: 正常"],
            )
            df_t = df.T
            st.dataframe(df_t, column_config={"_index": ""}, width="content")
            st.write(
                "- TP (True Positive): 実際に異常だったものが、異常と予測された回数\n",
                "- TN (True Negative): 実際に正常だったものが、正常と予測された回数\n",
                "- FP (False Positive): 実際に正常だったものが、異常と予測された回数\n",
                "- FN (False Negative): 実際に異常だったものが、正常と予測された回数\n",
            )
            st.write(
                "この表から、モデルの性能を以下の指標で評価できます。\n",
                "- 再現率（Recall）: TP / (TP + FN)\n",
                "- 適合率（Precision）: TP / (TP + FP)\n",
                "- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)\n",
                "- F1 Max: しきい値を動かしたときに得られる最大のF1 Score\n",
                "- 正解率（Accuracy）: (TP + TN) / (TP + TN + FP + FN)\n",
            )
            st.write(
                "再現率は、異常なものをどれだけ見つけられたかの割合です。異常を一つでも取りこぼしたくない場合、この値を100%に近づけます。<br />"
                "適合率は、異常と予測されたものの中で、本当に異常だったものの割合です。異常を誤って検出してしまうことを抑える場合、この値を100%に近づけます。",
                unsafe_allow_html=True,
            )

        if st.button("?", key="button_about_confution_matrix"):
            about_confution_matrix()

    with auroc_container:

        @st.dialog("AUROCとは", width="large")
        def about_auroc():
            st.markdown(
                "AUROC（Area Under the Receiver Operating Characteristic curve）は、分類モデルの性能を評価する指標です。<br />"
                "ROC曲線は、しきい値を変化させたときの「再現率（Recall）」と「偽陽性率（False Positive Rate）」の関係を示したものです。<br />"
                "AUROCは、このROC曲線の下の面積を示しており、値が高いほどモデルの性能が良いことを意味します。<br />"
                "AUROCの値は、0.0〜1.0の範囲を取り、0.5はランダムな予測と同等の性能、1.0は完璧な予測性能を示します。<br />"
                "一般的に値が高いほど性能が良いとされますが、実用的な基準はタスクやデータによって異なります。<br />",
                unsafe_allow_html=True,
            )

        if st.button("?", key="button_about_auroc"):
            about_auroc()

    with aupimo_container:

        @st.dialog("AUPIMOとは", width="large")
        def about_aupimo():
            st.write(
                "AUPIMO（Area Under the Per-Image Overlap）は、「異常領域をどれだけ正しく重ねて検出できているか」を、しきい値全体で評価した指標です。<br />"
                "各画像ごとに、予測された異常領域と正解領域の重なり具合（PIMO）を計算し、しきい値を変えながらその値を積算して求められます。<br />"
                "特に、画像のピクセル単位での異常検知性能を評価するために用いられます。<br />"
                "AUPIMOの値は一般に0〜1の範囲を取り、高いほど異常領域を正確に捉えていることを示します。<br />",
                unsafe_allow_html=True,
            )
            st.write(
                "以下は、AUPIMOの分布や特性を把握するための関連統計情報です。"
            )
            df = pd.DataFrame(
                {
                    "指標": [
                        "nobs",
                        "minmax",
                        "mean",
                        "variance",
                        "skewness",
                        "kurtosis",
                    ],
                    "意味": [
                        "データの個数（サンプル数）",
                        "最小値と最大値のタプル (min, max)",
                        "平均値（算術平均）",
                        "分散（不偏分散）",
                        "歪度（わいど）",
                        "尖度（せんど）",
                    ],
                    "説明": [
                        "いくつデータがあるか",
                        "データの範囲の端",
                        "データの中心的な値",
                        "データのばらつきの大きさ（平均からどれくらい散らばっているか）",
                        "分布の左右の偏り具合　0：左右対称　正：右に長い（右に裾が伸びる）　負：左に長い",
                        "分布のとがり具合　0：正規分布と同じ　正：とがっている（外れ値が出やすい）　負：平たい",
                    ],
                }
            )
            st.dataframe(df, hide_index=True)

        if st.button("?", key="button_about_aupimo"):
            about_aupimo()


def disp_train_images(images: list[UploadedFile]) -> None:
    """Display the training images in a grid.

    Args:
        images (list[UploadedFile]): List of uploaded training images.
    """
    print("disp_train_images called")
    if images:
        with train_images_placeholder.container(height=COLUMN_HEIGHT):
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


def predict_to_image(pred_image: torch.Tensor):
    """
    Converts a PyTorch tensor representing an image prediction to a NumPy uint8 image array.
    The function permutes the tensor dimensions from (C, H, W) to (W, H, C), clips the values to the [0, 1] range,
    scales them to [0, 255], and converts the result to uint8 format suitable for image display or saving.
    Args:
        pred_image (torch.Tensor): The predicted image tensor with shape (C, H, W).
    Returns:
        np.ndarray: The processed image as a NumPy array with dtype uint8 and shape (W, H, C).
    """
    print("predict_to_image called")
    return (pred_image.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255).astype(
        np.uint8
    )


def disp_result_images(
    predictions: list[ImageBatch] | None, threshold: float, has_abnormal: bool
) -> None:
    """
    Displays the result images, anomaly maps, and prediction scores in a Streamlit app, and provides a downloadable ZIP file containing the results.

    Args:
        predictions (list[ImageBatch]): A list of prediction objects, each containing image data, anomaly maps, prediction scores, and image paths.
        threshold (float): The threshold value used to determine if a prediction is normal or anomalous.
        has_abnormal (bool): Indicates if any abnormal predictions exist.

    Functionality:
        - Displays original images, their corresponding anomaly heatmaps, and prediction results in a 3-column grid layout.
        - Generates a CSV file summarizing the results (file name, score, and judgment).
        - Packages the heatmap images and CSV file into a ZIP archive.
        - Provides a download button for the ZIP file in the Streamlit interface.

    Notes:
        - Assumes the existence of several helper functions and variables (e.g., get_map_min_max, get_item, predict_to_image, superimpose_anomaly_map_g, result_images_placeholder, constants.RESULT_PATH).
        - Handles both torch.Tensor and non-tensor image types.
        - Skips predictions with missing or invalid data.
    """
    print("disp_result_images called")
    if predictions is None:
        print("No predictions to display.")
        return

    init_results()

    images = st.session_state["test_images"]

    map_min, map_max, map_ptp = get_map_min_max(predictions)
    with result_images_placeholder.container(height=COLUMN_HEIGHT_RESULT):
        st.header("検査結果")
        str_threshold = f"しきい値: {threshold:.5f}"
        st.info(str_threshold)
        st.session_state["str_threshold"] = str_threshold

        # CSV用のメモリバッファを用意
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        now = datetime.now()
        csv_writer.writerow([f"{now:%Y/%m/%d %H:%M:%S}"])
        csv_writer.writerow(
            [
                "検査手法",
                st.session_state["model_name"],
                "モデル",
                st.session_state["backbone"],
                "しきい値",
                threshold,
            ]
        )
        csv_writer.writerow(["ファイル名", "スコア", "判定"])

        # 画像とcsvファイルをzipに
        zip_path = Path(constants.RESULT_PATH) / "result.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        select_column_count = st.session_state["column_count"]
        group_size = 3  # 1セット3列
        # 1行あたりの結果数
        items_per_row = select_column_count // group_size
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for i in range(0, len(predictions), items_per_row):
                cols = st.columns(select_column_count)
                for j in range(items_per_row):
                    idx = i + j
                    if idx >= len(predictions):
                        break

                    prediction = predictions[idx]
                    image = images[idx]

                    base_col = j * group_size

                    cols[base_col].image(image, caption="", width="stretch")

                    image_path = (
                        Path(prediction.image_path[0]).name
                        if prediction.image_path
                        else "unknown"
                    )

                    # 縦横比
                    pil_image = Image.open(image)
                    height_per_width = pil_image.height / pil_image.width

                    anomaly_map = get_item(prediction, "anomaly_map")
                    if anomaly_map is not None:
                        anomaly_map = anomaly_map.cpu().numpy().squeeze()  # type: ignore
                        # anomaly_mapに合わせてリサイズ
                        resized_image = pil_image.resize(anomaly_map.shape[:2])  # type: ignore
                        np_image = np.array(resized_image)
                        # anomaly_mapを画像に重ねる
                        heat_map = superimpose_anomaly_map_g(
                            anomaly_map=anomaly_map,
                            image=np_image,
                            map_min=map_min,
                            map_ptp=map_ptp,
                        )
                        pil_heat_map = Image.fromarray(heat_map)
                        resized_heat_map = pil_heat_map.resize(
                            (
                                pil_heat_map.width,
                                int(pil_heat_map.width * height_per_width),
                            )
                        )
                        cols[base_col + 1].image(
                            resized_heat_map, caption="", width="stretch"
                        )
                        st.session_state["heat_maps"].append(resized_heat_map)

                        # zipに書き込み
                        # メモリ上に画像を保存
                        img_bytes = io.BytesIO()
                        resized_heat_map.save(
                            img_bytes,
                            format="JPEG",
                        )
                        img_bytes.seek(0)

                        # ZIPに画像を書き込み
                        zipf.writestr(
                            "result_" + Path(image_path).stem + ".jpg",
                            img_bytes.read(),
                        )

                    cols[base_col + 2].write(image_path)
                    st.session_state["test_image_path"].append(image_path)

                    pred_score = get_item(prediction, "pred_score")
                    pred_score = pred_score if pred_score is not None else 0.0

                    if pred_score <= threshold:
                        judge = "正常"
                    else:
                        judge = "異常"

                    str_results = f"score:{pred_score:.5f} [{judge}]"
                    st.session_state["str_results"].append(str_results)

                    with cols[base_col + 2]:
                        if judge == "正常":
                            st.success(str_results)
                        else:
                            st.error(str_results)

                    # CSVにファイル名を追加
                    csv_writer.writerow([image_path, pred_score, judge])

            # CSVをZIPに追加
            csv_buffer.seek(0)
            zipf.writestr(
                "result.csv", csv_buffer.getvalue().encode("utf-8-sig")
            )

    with download_button_placeholder.container():
        if zip_path.exists():
            # 保存ボタン
            st.download_button(
                "結果保存",
                data=zip_path.read_bytes(),
                file_name="result.zip",
                on_click="ignore",
            )

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
    print("disp_result_images finished")


def disp_metrics() -> None:
    """
    Calculates and displays the AUROC metric for a list of predictions in a Streamlit app.

    Args:
        predictions (list[ImageBatch]): A list of prediction objects, each containing image data, anomaly maps, and ground truth masks.

    Functionality:
        - Extracts anomaly maps and ground truth masks from the predictions.
        - Computes the AUROC metric using the extracted data.
        - Displays the AUROC score in the Streamlit interface.

    Notes:
        - Assumes that each prediction object has 'anomaly_map' and 'ground_truth_mask' attributes.
        - Handles cases where predictions are None or empty.
    """

    print("disp_metrics called")

    with tab_metrics:
        if st.session_state["chk_model_file"]:
            print("No predictions to calculate AUROC.")
            st.info(
                "モデルファイルを使用した場合は、パフォーマンスは表示されません。"
            )
            return

        if not st.session_state["abnormal_images"]:
            st.info(
                "異常画像が選択されていないため、パフォーマンスは表示されません。"
            )
            return

        predictions = st.session_state["train_predictions"]

        if predictions is None or len(predictions) == 0:
            st.info("指定した条件では、パフォーマンスは表示されません。")
            return
        predictions = cast(list[ImageBatch], predictions)

    with confutsion_matrix_container:
        try:

            # 混同行列
            all_preds = []
            all_labels = []

            for batch in predictions:
                # それぞれ Tensor を想定
                preds = batch.pred_label
                labels = batch.gt_label

                # GPU対策 + listに追加
                all_preds.append(preds.detach().cpu())  # type: ignore
                all_labels.append(labels.detach().cpu())  # type: ignore

            # 連結
            y_pred = torch.cat(all_preds).numpy()
            y_true = torch.cat(all_labels).numpy()

            # confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            cm_display = ConfusionMatrixDisplay(cm).plot()
            st.pyplot(cm_display.figure_)

            recall = recall_score(y_true, y_pred)
            # st.write(f"再現率（Recall）: {recall:.5f}")
            precision = precision_score(y_true, y_pred)
            # st.write(f"適合率（Precision）: {precision:.5f}")

            f1 = F1Score(fields=["pred_score", "gt_label"])
            f1_max = F1Max(fields=["pred_score", "gt_label"])
            for batch in predictions:
                f1.update(batch)
                f1_max.update(batch)
            f1_score = f1.compute()
            f1_max_score = f1_max.compute()

            accuracy = accuracy_score(y_true, y_pred)
            st.write(f"正解率（Accuracy）: {accuracy:.5f}")

            df_score = pd.DataFrame(
                {
                    "再現率（Recall）": [f"{recall:.5f}"],
                    "適合率（Precision）": [f"{precision:.5f}"],
                    "F1 Score": [f"{f1_score:.5f}"],
                    "F1 Max": [f"{f1_max_score:.5f}"],
                },
                index=["値"],
            )
            df_score = df_score.T
            st.table(df_score)

        except Exception as e:
            print(e)
            print(type(e))
            st.info("指定した条件では、混同行列は表示されません。")

    with auroc_container:
        try:
            # Calculate AUROC
            auroc = AUROC(fields=["pred_score", "gt_label"])
            for batch in predictions:
                auroc.update(batch)

            figure, _ = auroc.generate_figure()
            st.pyplot(figure)

        except Exception as e:
            print(e)
            print(type(e))
            st.info("指定した条件では、AUROCは表示されません。")

    with aupimo_container:

        try:
            # Calculate AUPIMO
            aupimo = AUPIMO(
                return_average=False,
            )
            for batch in predictions:
                aupimo.update(batch)
            # `pimo_result` has the PIMO curves of each image
            # `aupimo_result` has the AUPIMO values
            #     i.e. their Area Under the Curve (AUC)
            pimo_result, aupimo_result = aupimo.compute()
            # ignore removing the `nan`s
            isnan = torch.isnan(aupimo_result.aupimos)  # type: ignore

            # the `nan`s are the normal images; they do not
            # have a score because recall is not defined for them
            descriptive = stats.describe(aupimo_result.aupimos[~isnan])  # type: ignore
            df_descriptive = pd.DataFrame([descriptive], columns=descriptive._fields)  # type: ignore
            df_descriptive["minmax"] = df_descriptive["minmax"].apply(
                lambda x: f"{x[0]},{x[1]}"
            )
            df_descriptive = df_descriptive.T
            df_descriptive.columns = ["value"]
            df_descriptive["value"] = df_descriptive["value"].astype(str)
            st.dataframe(df_descriptive)
            fig, ax = plt.subplots()
            ax.hist(
                aupimo_result.aupimos.numpy(),  # type: ignore
                bins=np.linspace(0, 1, 11),  # type: ignore
                edgecolor="black",
            )
            ax.set_ylabel("Count (number of images)")
            ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
            ax.set_xlim(0, 1)
            ax.set_xlabel("AUPIMO [%]")
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.grid()
            ax.set_title("AUPIMO distribution")
            st.pyplot(fig)  # noqa: B018, RUF100
        except Exception as e:
            print(e)
            st.info("指定した条件では、AUPIMOは表示されません。")

    print("disp_metrics finished")


def init_results():
    """
    Initializes session state variables for storing test images, heat maps, image paths, result strings, and threshold value in a Streamlit application.
    """
    print("init_results called")
    st.session_state["heat_maps"] = []
    st.session_state["test_image_path"] = []
    st.session_state["str_results"] = []
    st.session_state["str_threshold"] = ""
    print("init_results finished")


def save_images(
    train_images: list[UploadedFile],
    test_images: list[UploadedFile],
    abnormal_images: list[UploadedFile],
    mask_images: list[UploadedFile],
):
    """
    Saves uploaded images to designated dataset and result directories, organizing them based on the presence of abnormal images.

    This function performs the following steps:
    1. Deletes existing images in the dataset and result paths.
    2. Determines if abnormal images are present in the session state.
    3. Saves training images to the appropriate 'train' directory (either directly or under 'normal' subdirectory).
    4. Saves test images to the appropriate 'test' directory (either directly or under 'normal' subdirectory), naming them with a zero-padded index and their original name.

    Args:
        train_images (list[UploadedFile]): List of uploaded training images.
        test_images (list[UploadedFile]): List of uploaded test images.
        abnormal_images (list[UploadedFile]): List of uploaded abnormal images.
        mask_images (list[UploadedFile]): List of uploaded mask images.

    Note:
        The function relies on global constants for dataset and result paths, and session state for abnormal image detection.
    """
    print("save_images called")
    # 1. Delete DATASET_PATH images
    dataset_path = Path(constants.DATASET_PATH)
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    result_path = Path(constants.RESULT_PATH)
    if result_path.exists():
        shutil.rmtree(result_path)

    # 異常画像有り
    has_abnormal = st.session_state["abnormal_images"] is not None and 0 < len(
        st.session_state["abnormal_images"]
    )

    # 2. Save train_images to DATASET_PATH/train
    train_dir = dataset_path / "train" / "normal"
    save_images_to_dir(train_images, train_dir)

    # 3. Save test_images to DATASET_PATH/test
    test_dir = dataset_path / "test"
    save_images_to_dir(test_images, test_dir)

    if has_abnormal:
        abnormal_dir = dataset_path / "train" / "abnormal"
        save_images_to_dir(abnormal_images, abnormal_dir)
        mask_dir = dataset_path / "train" / "mask"
        save_images_to_dir(mask_images, mask_dir)

    print("save_images finished")


def save_images_to_dir(images: list[UploadedFile], directory: Path):
    """
    Saves a list of image-like objects to the specified directory.

    Each image is saved with a filename composed of its index and its original name,
    with directory structure flattened and separated by underscores.

    Args:
        images (list[UploadedFile]): An iterable of image-like objects, each having a 'name' attribute and a 'getvalue()' method.
        directory (Path): The target directory to save images. Created if it does not exist.

    Raises:
        OSError: If the image cannot be written to disk.
    """
    print("save_images_to_dir called")
    directory.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        image_name = "_".join(Path(image.name).parts)
        with open(directory / f"{i:04}.{image_name}", "wb") as f:
            f.write(image.getvalue())
    print("save_images_to_dir finished")


def get_item(prediction, key):
    """
    Retrieve the value associated with a given key from a prediction object, handling various data types.
    Parameters:
        prediction (object): The object containing the attribute to retrieve.
        key (str): The attribute name to access within the prediction object.
    Returns:
        The first element or scalar value associated with the specified key, depending on the type:
            - For lists or tuples: returns the first element if available, else None.
            - For torch.Tensor: returns None if empty, the scalar value if single element, or the first element otherwise.
            - For numpy.ndarray: returns None if empty, the scalar value if single element, or the first element otherwise.
            - For other types (int, float, etc.): returns the value directly.
        Returns None if the attribute does not exist.
    """
    print(f"get_item called for key: {key}")
    if not hasattr(prediction, key):
        return None

    val = getattr(prediction, key)

    # list or tuple
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) > 0 else None

    # PyTorch Tensor
    if isinstance(val, torch.Tensor):
        if val.numel() == 0:
            return None
        if val.numel() == 1:
            return val.item()
        return val[0] if val.dim() > 0 else val.item()

    # NumPy ndarray
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return None
        if val.size == 1:
            return val.item()
        return val.flat[0]  # flat iteratorで最初の要素

    print("get_item finished")
    # それ以外（int, float, etc.）
    return val


def get_map_min_max(predictions) -> tuple[float, float, float]:
    """
    Calculates the minimum, maximum, and range (peak-to-peak) values across anomaly maps in a list of predictions.
    Args:
        predictions (Iterable): A list or iterable of prediction objects, each containing an 'anomaly_map' attribute.
            Each 'anomaly_map' is expected to be a tensor-like object where the first element (index 0) can be
            converted to a NumPy array via `.cpu().numpy()`.
    Returns:
        tuple[float, float, float]: A tuple containing:
            - map_min (float): The minimum value found across all anomaly maps.
            - map_max (float): The maximum value found across all anomaly maps.
            - map_ptp (float): The range (max - min) of the anomaly map values.
    """
    print("get_map_min_max called")
    if (
        predictions is None
        or len(predictions) == 0
        or predictions[0].anomaly_map is None
        or len(predictions[0].anomaly_map) == 0
    ):
        return 0.0, 1.0, 1.0
    map_min = min(
        prediction.anomaly_map[0].min().cpu().numpy()
        for prediction in predictions
    )
    map_max = max(
        prediction.anomaly_map[0].max().cpu().numpy()
        for prediction in predictions
    )
    map_ptp = map_max - map_min
    print("get_map_min_max finished")
    return map_min, map_max, map_ptp


def superimpose_anomaly_map_g(
    anomaly_map: np.ndarray,
    image: np.ndarray,
    map_min: float,
    map_ptp: float,
    alpha: float = 0.4,
    gamma: int = 0,
) -> np.ndarray:
    """Superimpose anomaly map on image.

    Args:
        anomaly_map (np.ndarray): Anomaly map.
        image (np.ndarray): Image.
        alpha (float): Alpha value for superimposition.
        gamma (int): Gamma value for superimposition.
        map_min (float): Minimum value of the anomaly map.
        map_ptp (float): Range of the anomaly map.

    Returns:
        np.ndarray: Superimposed image.
    """
    print("superimpose_anomaly_map_g called")
    assert anomaly_map.shape == image.shape[:2], (
        f"Anomaly map shape {anomaly_map.shape} does not match image shape "
        f"{image.shape[:2]}."
    )
    nomalized_map = (((anomaly_map - map_min) / map_ptp) * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(nomalized_map, cv2.COLORMAP_JET)
    rgb_color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    height, width = rgb_color_map.shape[:2]
    image = cv2.resize(image, (width, height))
    print("superimpose_anomaly_map_g finished")
    return cv2.addWeighted(rgb_color_map, alpha, image, (1 - alpha), gamma)


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


def main_page(submitted: bool) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
    """
    print("main_page called with submitted =", submitted)
    if submitted:
        start = time.time()
        with st.spinner("処理中", show_time=True):

            # 異常画像有り（モデルファイルを使用しない）
            has_abnormal = (
                not st.session_state["chk_model_file"]
                and st.session_state["abnormal_images"] is not None
                and 0 < len(st.session_state["abnormal_images"])
            )

            # 入力チェック
            if not is_input_ok(has_abnormal):
                init_results()
                return

            try:

                # アップロードしたファイルをフォルダに保存
                save_images(
                    st.session_state["train_images"],
                    st.session_state["test_images"],
                    st.session_state["abnormal_images"],
                    st.session_state["mask_images"],
                )

                # テスト用
                folder_dataset_test = FolderDataset(
                    name="folder_dataset_test",
                    normal_dir=Path(constants.DATASET_PATH) / "test",
                    # splitをtrainにすることでラベルを「正常」としてnormal_dirのデータを読み込み
                    split="train",
                )

                train_predictions = None
                st.session_state["train_predictions"] = None

                if st.session_state["chk_model_file"]:
                    # モデルファイルを使用する場合
                    # 学習済みモデルを読み込む
                    model_file = st.session_state["model_file"]
                    print(f"Loading model from {model_file}")
                    loaded_model = torch.load(model_file, weights_only=False)
                    print(loaded_model)
                    model = loaded_model["model"]
                    engine = Engine()
                    predictions = engine.predict(
                        model=model, dataset=folder_dataset_test
                    )
                    threshold = loaded_model["threshold"]

                else:
                    # モデルファイルを使用しない場合
                    # Load the selected model
                    model = models.get_model(
                        model_name=st.session_state["model_name"],
                        backbone=st.session_state["backbone"],
                        image_size=st.session_state["image_size"],
                        max_epochs=st.session_state["epochs"],
                    )

                    if has_abnormal:
                        # 異常画像が選択されている場合
                        datamodule = Folder(
                            name="custom",
                            root=constants.DATASET_PATH,
                            # abnormal_dirとnormal_test_dirのデータがvalになる
                            normal_dir=Path("train") / "normal",
                            abnormal_dir=Path("train") / "abnormal",
                            mask_dir=Path("train") / "mask",
                            normal_test_dir=Path("train") / "normal",
                            val_split_mode=ValSplitMode.SAME_AS_TEST,
                            train_batch_size=constants.BATCH_SIZE,
                            eval_batch_size=constants.BATCH_SIZE,
                            num_workers=0,
                        )
                    else:
                        # 異常画像が選択されていない場合
                        datamodule = Folder(
                            name="custom",
                            root=constants.DATASET_PATH,
                            normal_dir=Path("train") / "normal",
                            normal_test_dir="test",
                            # test_split_mode=TestSplitMode.SYNTHETIC,
                            # val_split_mode=ValSplitMode.SYNTHETIC,
                            val_split_mode=ValSplitMode.FROM_TRAIN,
                            # 検証データを入れるとスコアが1か0になるため、検証はしない
                            # ratioを0にするとデフォルト値で分割されるため、非常に小さい値を設定
                            val_split_ratio=0.0001,
                            train_batch_size=constants.BATCH_SIZE,
                            eval_batch_size=constants.BATCH_SIZE,
                            num_workers=1,
                        )
                    datamodule.setup()

                    # ----- 学習 -----
                    engine = Engine(
                        # callbacks=callbacks,
                        max_epochs=st.session_state["epochs"],
                        accelerator="auto",
                        devices=1,
                    )

                    try:
                        engine.fit(model=model, datamodule=datamodule)
                    except Exception as e:
                        print(e)
                        # retry
                        engine.fit(model=model, datamodule=datamodule)

                    # モデル保存
                    engine.export(
                        model=model,
                        export_type=ExportType.TORCH,
                        export_root=Path(constants.RESULT_PATH),
                    )

                    if has_abnormal:
                        # 異常画像が選択されている場合
                        # パフォーマンスとしきい値用に全件predict
                        train_predictions = engine.predict(
                            model=model,
                            dataloaders=datamodule.val_dataloader(),
                        )
                        st.session_state["train_predictions"] = (
                            train_predictions
                        )
                    elif (
                        not st.session_state["chk_model_file"]
                        and st.session_state["threshold_auto"]
                    ):
                        # モデルファイルを使用せずにしきい値自動の場合
                        # しきい値用にtrainをpredict
                        train_predictions = engine.predict(
                            model=model,
                            dataloaders=datamodule.train_dataloader(),
                        )
                        st.session_state["train_predictions"] = (
                            train_predictions
                        )

                    # しきい値
                    if st.session_state["threshold_auto"]:
                        # しきい値自動
                        if (
                            has_abnormal
                            and train_predictions is not None
                            and 0 < len(train_predictions)
                        ):
                            # 異常画像が選択されている場合
                            try:
                                adaptiveThreshold = F1AdaptiveThreshold(
                                    fields=["pred_score", "gt_label"]
                                )
                                for batch in train_predictions:  # type: ignore
                                    adaptiveThreshold.update(batch)  # type: ignore
                                threshold = adaptiveThreshold.compute()
                            except Exception as e:
                                print(f"exception: {e}")
                                train_scores = [
                                    get_item(prediction, "pred_score")
                                    for prediction in (train_predictions or [])
                                ]
                                # 閾値（99.7％）
                                threshold = np.mean(train_scores) + 3 * np.std(train_scores)  # type: ignore
                                # # 四分位範囲×1.5の場合
                                # threshold = np.percentile(train_scores, 75) + 1.5 * (
                                #     np.percentile(train_scores, 75) - np.percentile(train_scores, 25)
                                # )
                            print(f"threshold: {threshold}")
                        else:
                            # 異常画像が選択されていない場合
                            train_scores = [
                                get_item(prediction, "pred_score")
                                for prediction in (train_predictions or [])
                            ]
                            # 閾値（99.7％）
                            threshold = np.mean(train_scores) + 3 * np.std(train_scores)  # type: ignore
                    else:
                        # 入力したしきい値
                        threshold = st.session_state["threshold"]

                    # しきい値をモデルファイルに保存
                    exported_model = torch.load(
                        constants.MODEL_PATH, weights_only=False
                    )
                    exported_model["threshold"] = threshold
                    torch.save(exported_model, constants.MODEL_PATH)

                    # 予想
                    if isinstance(model, WinClip) or isinstance(model, VlmAd):
                        # VLM-AD、WinClipはFolderDatasetに対応していないためFolderで再度読み込み
                        datamodule_test = Folder(
                            name="custom_test",
                            root=constants.DATASET_PATH,
                            normal_dir=Path("test"),
                            val_split_mode=ValSplitMode.SAME_AS_TEST,
                            val_split_ratio=0.0001,
                            normal_test_dir="test",
                            train_batch_size=constants.BATCH_SIZE,
                            eval_batch_size=constants.BATCH_SIZE,
                            num_workers=0,
                        )
                        datamodule_test.setup()
                        predictions = engine.predict(
                            model=model, datamodule=datamodule_test
                        )
                    else:
                        predictions = engine.predict(
                            model=model, dataset=folder_dataset_test
                        )

                # 結果描画
                disp_train_images(st.session_state["train_images"])
                disp_result_images(predictions, threshold=threshold, has_abnormal=has_abnormal)  # type: ignore
                disp_metrics()

                end = time.time()
                elapsed_time = end - start
                minutes = int(elapsed_time // 60)
                seconds = elapsed_time % 60
                st.success(
                    f"処理完了 ({minutes} minutes, {seconds:.1f} seconds)",
                    icon="✔️",
                )
                print(f"elapsed_time: {elapsed_time}")
            except Exception as e:
                print(e)
                end = time.time()
                elapsed_time = end - start
                minutes = int(elapsed_time // 60)
                seconds = elapsed_time % 60
                st.error(
                    f"Encountered an error: {e}\n({minutes} minutes, {seconds:.1f} seconds)",
                    icon="❌",
                )
                init_results()

    elif "str_results" in st.session_state and 0 < len(
        st.session_state["str_results"]
    ):
        # 処理済み
        disp_session_images()
    else:
        pass

    print("main_page finished")


def disp_footer():
    """Display footer information."""
    print("disp_footer called")
    st.divider()
    st.caption(
        "© 2026 Your Company Name",
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
        with open("LICENSE", "rb") as f:
            data = f.read()
        st.download_button(
            label="LICENSE",
            data=data,
            file_name="LICENSE",
            mime="text/plain",
            type="tertiary",
        )
    print("disp_footer finished")


def main():
    """
    Main function to run the Streamlit application.

    This function initializes the tab_conditions configuration and the main page layout.
    It retrieves the user inputs from the tab_conditions, and passes them to the main page function.
    The main page function then generates images based on these inputs.
    """
    submitted = configure_conditions()
    disp_metrix_about_button()
    main_page(submitted)
    disp_footer()


if __name__ == "__main__":
    main()
