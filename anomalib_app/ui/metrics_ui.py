import logging

import pandas as pd
import streamlit as st

from anomalib_app.core.metrics import run_metrics_if_needed

logger = logging.getLogger(__name__)


def disp_metrics_about_button(tab, metrics_containers):
    """Displays information buttons for confusion matrix, AUROC, and AUPIMO metrics."""
    with metrics_containers[0]:

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
                "- 正解率（Accuracy）: (TP + TN) / (TP + TN + FP + FN)\n",
                "- 再現率（Recall）: TP / (TP + FN)\n",
                "- 適合率（Precision）: TP / (TP + FP)\n",
                "- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)\n",
                "- F1 Max: しきい値を動かしたときに得られる最大のF1 Score\n",
            )
            st.write(
                "再現率は、異常なものをどれだけ見つけられたかの割合です。異常を一つでも取りこぼしたくない場合、この値を100%に近づけます。<br />"
                "適合率は、異常と予測されたものの中で、本当に異常だったものの割合です。異常を誤って検出してしまうことを抑える場合、この値を100%に近づけます。",
                unsafe_allow_html=True,
            )

        if st.button(
            "?",
            key="button_about_confution_matrix",
            help="混同行列（Confusion Matrix）とは",
        ):
            about_confution_matrix()

    with metrics_containers[1]:

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

        if st.button("?", key="button_about_auroc", help="AUROCとは"):
            about_auroc()

    with metrics_containers[2]:

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

        if st.button("?", key="button_about_aupimo", help="AUPIMOとは"):
            about_aupimo()


def disp_metrics(tab_metrics, metrics_containers) -> None:
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

    logger.debug("disp_metrics called")

    with tab_metrics:
        if st.session_state["chk_model_file"]:
            logger.debug("No predictions to calculate AUROC.")
            st.info(
                "モデルファイルを使用した場合は、パフォーマンスは表示されません。"
            )
            return

        if not st.session_state["abnormal_images"]:
            st.info(
                "異常画像が選択されていないため、パフォーマンスは表示されません。"
            )
            return

        if not st.session_state["chk_disp_metrics"]:
            st.info("指定した条件では、パフォーマンスは表示されません。")
            return

        run_metrics_if_needed()
        result = st.session_state.get("metrics")

        if result is None:
            st.info("指定した条件では、パフォーマンスは表示されません。")
            return

    with metrics_containers[0]:
        if (
            not st.session_state["chk_disp_confusion_matrix"]
            or result["haserrors"][0]
        ):
            st.info("指定した条件では、混同行列は表示されません。")
        else:
            st.pyplot(result["fig_cm"])
            st.table(result["df_score"])

    with metrics_containers[1]:
        if not st.session_state["chk_disp_auroc"] or result["haserrors"][1]:
            st.info("指定した条件では、AUROCは表示されません。")
        else:
            st.pyplot(result["fig_auroc"])

    with metrics_containers[2]:
        if not st.session_state["chk_disp_aupimo"] or result["haserrors"][2]:
            st.info("指定した条件では、AUPIMOは表示されません。")
        else:
            st.pyplot(result["fig_aupimo"])
            st.dataframe(result["df_aupimo"])

    logger.debug("disp_metrics finished")
