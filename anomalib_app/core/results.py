import copy
import csv
import io
import logging
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from anomalib_app.core import constants
from anomalib_app.core.image_utils import superimpose_anomaly_map_g
from anomalib_app.core.list_utils import get_item

logger = logging.getLogger(__name__)


def init_results():
    """
    Initializes session state variables for storing test images, heat maps, image paths, result strings, and threshold value in a Streamlit application.
    """
    logger.debug("init_results called")
    st.session_state["heat_maps"] = []
    st.session_state["test_image_path"] = []
    st.session_state["str_results"] = []
    st.session_state["str_threshold"] = ""
    st.session_state["train_images_used"] = []
    st.session_state["test_images_used"] = []
    logger.debug("init_results finished")


def create_result_data(predictions, threshold):
    """
    Compute heat maps, result strings, file paths, and ZIP/CSV for predictions.
    Stores everything in st.session_state for later display.

    Args:
        predictions (list[ImageBatch]): List of prediction objects.
        threshold (float): Threshold for judging normal vs anomalous.
    """
    logger.debug("create_result_data called")
    if predictions is None:
        return

    # 初期化
    init_results()
    st.session_state["str_threshold"] = f"しきい値: {threshold:.5f}"
    if not st.session_state["chk_model_file"]:
        st.session_state["train_images_used"] = copy.deepcopy(
            st.session_state["train_images"]
        )
    st.session_state["test_images_used"] = copy.deepcopy(
        st.session_state["test_images"]
    )

    images = st.session_state["test_images_used"]

    # ヒートマップ用の最小・最大値取得
    map_min, map_max, map_ptp = get_map_min_max(predictions)

    # CSV用メモリバッファ
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    now = datetime.now()
    csv_writer.writerow([f"{now:%Y/%m/%d %H:%M:%S}"])
    csv_writer.writerow(
        [
            "検査手法",
            st.session_state.get("model_name", ""),
            "モデル",
            st.session_state.get("backbone", ""),
            "しきい値",
            threshold,
        ]
    )
    csv_writer.writerow(["ファイル名", "スコア", "判定"])

    # ZIPファイル作成
    zip_path = Path(constants.RESULT_PATH) / "result.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    select_column_count = st.session_state.get("column_count", 3)
    group_size = 3
    items_per_row = select_column_count // group_size

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for i in range(0, len(predictions), items_per_row):
            for j in range(items_per_row):
                idx = i + j
                if idx >= len(predictions):
                    break

                prediction = predictions[idx]
                image = images[idx]
                # base_col = j * group_size

                # オリジナル画像パス
                image_path = (
                    Path(prediction.image_path[0]).name
                    if prediction.image_path
                    else "unknown"
                )
                st.session_state["test_image_path"].append(image_path)

                # 予測スコアと判定
                pred_score = get_item(prediction, "pred_score") or 0.0
                judge = "異常" if pred_score > threshold else "正常"
                str_results = f"score:{pred_score:.5f} [{judge}]"
                st.session_state["str_results"].append(str_results)

                # ヒートマップ生成
                anomaly_map = get_item(prediction, "anomaly_map")
                if anomaly_map is not None:
                    anomaly_map = anomaly_map.cpu().numpy().squeeze()  # type: ignore
                    pil_image = Image.open(image)
                    height_per_width = pil_image.height / pil_image.width
                    resized_image = pil_image.resize(anomaly_map.shape[:2])  # type: ignore
                    np_image = np.array(resized_image)
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
                    st.session_state["heat_maps"].append(resized_heat_map)

                    # ZIPに保存
                    img_bytes = io.BytesIO()
                    resized_heat_map.save(img_bytes, format="JPEG")
                    img_bytes.seek(0)
                    zipf.writestr(
                        "result_" + Path(image_path).stem + ".jpg",
                        img_bytes.read(),
                    )

                # CSVに記録
                csv_writer.writerow([image_path, pred_score, judge])

        # CSVをZIPに追加
        csv_buffer.seek(0)
        zipf.writestr("result.csv", csv_buffer.getvalue().encode("utf-8-sig"))

    # ZIPのパスも保存しておく
    st.session_state["zip_path"] = zip_path

    logger.debug("create_result_data finished")


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
    logger.debug("get_map_min_max called")
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
    logger.debug("get_map_min_max finished")
    return map_min, map_max, map_ptp
