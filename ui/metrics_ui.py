from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from anomalib.data import ImageBatch
from anomalib.metrics import AUPIMO, AUROC, F1Max, F1Score
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


def disp_metrix_about_button(tab, metrix_containers):
    """Displays information buttons for confusion matrix, AUROC, and AUPIMO metrics."""
    with metrix_containers[0]:

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

    with metrix_containers[1]:

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

    with metrix_containers[2]:

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

@st.cache_data
def calc_all_metrics(_predictions: list[ImageBatch]) -> dict:
    """Calculate all metrics for the given predictions.

    Args:
        _predictions (list[ImageBatch]): _description_

    Returns:
        dict: A dictionary containing the calculated metrics.
    """

    fig_cm = None
    accuracy = None
    df_score = None
    fig_auroc = None
    fig_hist = None
    df_desc = None
    haserrors = [False, False, False]

    # ===== 混同行列 =====
    try:
        all_preds = []
        all_labels = []

        for batch in _predictions:
            # それぞれ Tensor を想定 GPU対策 + listに追加
            all_preds.append(batch.pred_label.detach().cpu()) # type: ignore
            all_labels.append(batch.gt_label.detach().cpu()) # type: ignore

        # 連結
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()

        # ===== confusion matrix =====
        cm = confusion_matrix(y_true, y_pred)
        fig_cm, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)

        # ===== scores =====
        recall = recall_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        f1 = F1Score(fields=["pred_score", "gt_label"])
        f1_max = F1Max(fields=["pred_score", "gt_label"])
        for batch in _predictions:
            f1.update(batch)
            f1_max.update(batch)
        f1_score = f1.compute()
        f1_max_score = f1_max.compute()

        df_score = pd.DataFrame(
            {
                "正解率（Accuracy）": [f"{accuracy:.5f}"],
                "再現率（Recall）": [f"{recall:.5f}"],
                "適合率（Precision）": [f"{precision:.5f}"],
                "F1 Score": [f"{f1_score:.5f}"],
                "F1 Max": [f"{f1_max_score:.5f}"],
            },
            index=["値"],
        ).T
    except Exception as e:
        print(e)
        print(type(e))
        haserrors[0] = True

    # ===== AUROC =====
    try:
        # Calculate AUROC
        auroc = AUROC(fields=["pred_score", "gt_label"])
        for batch in _predictions:
            auroc.update(batch)

        fig_auroc, _ = auroc.generate_figure()

    except Exception as e:
        print(e)
        print(type(e))
        haserrors[1] = True

    # ===== AUPIMO =====
    try:

        aupimo = AUPIMO(return_average=False)
        for batch in _predictions:
            aupimo.update(batch)

        _, aupimo_result = aupimo.compute()
        isnan = torch.isnan(aupimo_result.aupimos) # type: ignore

        descriptive = stats.describe(aupimo_result.aupimos[~isnan]) # type: ignore

        df_desc = pd.DataFrame([descriptive], columns=descriptive._fields)
        df_desc["minmax"] = df_desc["minmax"].apply(lambda x: f"{x[0]},{x[1]}")
        df_desc = df_desc.T
        df_desc.columns = ["value"]
        df_desc["value"] = df_desc["value"].astype(str)

        fig_hist, ax = plt.subplots()
        ax.hist(
            aupimo_result.aupimos.numpy(), # type: ignore
            bins=np.linspace(0, 1, 11), # type: ignore
            edgecolor="black",
        )
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUPIMO [%]")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid()
    except Exception as e:
        print(e)
        print(type(e))
        haserrors[2] = True

    return {
        "fig_cm": fig_cm,
        "accuracy": accuracy,
        "df_score": df_score,
        "fig_auroc": fig_auroc,
        "fig_aupimo": fig_hist,
        "df_aupimo": df_desc,
        "haserrors": haserrors,
    }

def run_metrics_if_needed():
    """Run metrics calculation if needed."""
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = None

    predictions = st.session_state.get("train_predictions")

    # ===== 自動リセット =====
    if "prev_predictions" not in st.session_state:
        st.session_state["prev_predictions"] = None

    if st.session_state["prev_predictions"] is not predictions:
        st.session_state["metrics"] = None
        st.session_state["prev_predictions"] = predictions

    if predictions and st.session_state["metrics"] is None:
        st.session_state["metrics"] = calc_all_metrics(predictions)

def disp_metrics(tab_metrics, metrix_containers) -> None:
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

        run_metrics_if_needed()
        result = st.session_state.get("metrics")

        if result is None:
            st.info("指定した条件では、パフォーマンスは表示されません。")
            return

    with metrix_containers[0]:
        if result["haserrors"][0]:
            st.info("指定した条件では、混同行列は表示されません。")
        else:
            st.pyplot(result["fig_cm"])
            st.table(result["df_score"])

    with metrix_containers[1]:
        if result["haserrors"][1]:
            st.info("指定した条件では、AUROCは表示されません。")
        else:
            st.pyplot(result["fig_auroc"])

    with metrix_containers[2]:
        if result["haserrors"][2]:
            st.info("指定した条件では、AUPIMOは表示されません。")
        else:
            st.pyplot(result["fig_aupimo"])
            st.dataframe(result["df_aupimo"])

    print("disp_metrics finished")
