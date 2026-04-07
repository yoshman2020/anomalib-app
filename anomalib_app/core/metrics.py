import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from anomalib.data import ImageBatch
from anomalib.metrics import AUPIMO, AUROC, F1Max, F1Score
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


def calc_all_metrics(predictions: list[ImageBatch]) -> dict:
    """Calculate all metrics for the given predictions.

    Args:
        predictions (list[ImageBatch]): predictions to calculate metrics for.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    logger.debug("calc_all_metrics called")

    if not predictions or not st.session_state["chk_disp_metrics"]:
        return {
            "fig_cm": None,
            "accuracy": None,
            "df_score": None,
            "fig_auroc": None,
            "fig_aupimo": None,
            "df_aupimo": None,
            "haserrors": [True, True, True],
        }

    fig_cm = None
    accuracy = None
    df_score = None
    fig_auroc = None
    fig_hist = None
    df_desc = None
    haserrors = [False, False, False]

    # ===== 混同行列 =====
    fig_cm, accuracy, df_score = calc_accuracy(predictions, haserrors)

    # ===== AUROC =====
    fig_auroc = calc_auroc(predictions, haserrors)

    # ===== AUPIMO =====
    fig_hist, df_desc = calc_aupimo(predictions, haserrors)

    logger.debug("calc_all_metrics finished")

    return {
        "fig_cm": fig_cm,
        "accuracy": accuracy,
        "df_score": df_score,
        "fig_auroc": fig_auroc,
        "fig_aupimo": fig_hist,
        "df_aupimo": df_desc,
        "haserrors": haserrors,
    }


def calc_accuracy(
    predictions: list[ImageBatch], haserrors: list[bool]
) -> tuple[
    Figure | None,
    float | np.float16 | np.float32 | np.float64 | None,
    pd.DataFrame | None,
]:
    """Calculate accuracy and confusion matrix for the given predictions.

    Args:
        predictions (list[ImageBatch]): predictions to calculate metrics for.
        haserrors (list[bool]): list to track any errors that occur during metric calculation.

    Returns:
        tuple[ Figure | None, float | np.float16 | np.float32 | np.float64 | None, pd.DataFrame | None, ]: _description_
    """
    if not st.session_state["chk_disp_confusion_matrix"]:
        return None, None, None
    fig_cm = None
    accuracy = None
    df_score = None
    try:
        all_preds = []
        all_labels = []

        for batch in predictions:
            # それぞれ Tensor を想定 GPU対策 + listに追加
            all_preds.append(batch.pred_label.detach().cpu())  # type: ignore
            all_labels.append(batch.gt_label.detach().cpu())  # type: ignore

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
        for batch in predictions:
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
        logger.exception("Error calculating accuracy and confusion matrix: %s", e)
        haserrors[0] = True
    return fig_cm, accuracy, df_score


def calc_auroc(
    predictions: list[ImageBatch], haserrors: list[bool]
) -> Figure | None:
    """Calculate AUROC for the given predictions.

    Args:
        predictions (list[ImageBatch]): predictions to calculate metrics for.
        haserrors (list[bool]): list to track any errors that occur during metric calculation.

    Returns:
        Figure | None: AUROC figure if calculation is successful, None otherwise.
    """
    if not st.session_state["chk_disp_auroc"]:
        return None
    fig_auroc = None
    try:
        # Calculate AUROC
        auroc = AUROC(fields=["pred_score", "gt_label"])
        for batch in predictions:
            auroc.update(batch)

        fig_auroc, _ = auroc.generate_figure()

    except Exception as e:
        logger.exception("Error calculating AUROC: %s", e)
        haserrors[1] = True
    return fig_auroc


def calc_aupimo(
    predictions: list[ImageBatch], haserrors: list[bool]
) -> tuple[Figure | None, pd.DataFrame | None]:
    """Calculate AUPIMO for the given predictions.

    Args:
        predictions (list[ImageBatch]): predictions to calculate metrics for.
        haserrors (list[bool]): list to track any errors that occur during metric calculation.

    Returns:
        tuple[Figure | None, pd.DataFrame | None]: AUPIMO figure and descriptive statistics if calculation is successful, None otherwise.
    """
    if not st.session_state["chk_disp_aupimo"]:
        return None, None
    fig_hist = None
    df_desc = None
    try:
        aupimo = AUPIMO(return_average=False)
        for batch in predictions:
            aupimo.update(batch)

        _, aupimo_result = aupimo.compute()
        isnan = torch.isnan(aupimo_result.aupimos)  # type: ignore

        descriptive = stats.describe(aupimo_result.aupimos[~isnan])  # type: ignore

        df_desc = pd.DataFrame([descriptive], columns=descriptive._fields)
        df_desc["minmax"] = df_desc["minmax"].apply(lambda x: f"{x[0]},{x[1]}")
        df_desc = df_desc.T
        df_desc.columns = ["value"]
        df_desc["value"] = df_desc["value"].astype(str)

        fig_hist, ax = plt.subplots()
        ax.hist(
            aupimo_result.aupimos.numpy(),  # type: ignore
            bins=np.linspace(0, 1, 11),  # type: ignore
            edgecolor="black",
        )
        ax.set_ylabel("Count")
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUPIMO [%]")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid()
    except Exception as e:
        logger.exception("Error calculating AUPIMO: %s", e)
        haserrors[2] = True
    return fig_hist, df_desc


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
