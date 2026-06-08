import copy
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from anomalib.data import Folder, Folder3D, FolderDataset
from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from anomalib.engine.engine import Engine
from anomalib.metrics import F1AdaptiveThreshold
from anomalib.models import CFM, AnomalyVFM, VlmAd, WinClip
from streamlit.delta_generator import DeltaGenerator
from streamlit.elements.lib.mutable_tab_container import TabContainer

import anomalib_app.core.constants as constants
import anomalib_app.core.models as models
from anomalib_app.core.io_utils import save_images
from anomalib_app.core.list_utils import get_item
from anomalib_app.core.results import create_result_data, init_results
from anomalib_app.core.validation import is_input_ok
from anomalib_app.ui.display import disp_session_images, show_result_tab
from anomalib_app.ui.metrics_ui import disp_metrics

logger = logging.getLogger(__name__)


def main_page(
    submitted: bool,
    train_images_container: DeltaGenerator,
    result_images_container: DeltaGenerator,
    download_button_container: DeltaGenerator,
    tab_metrics: TabContainer,
    metrics_containers: list[DeltaGenerator],
) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
    """
    logger.info(f"main_page called with submitted = {submitted}")

    if not submitted:
        if "str_results" in st.session_state and 0 < len(
            st.session_state["str_results"]
        ):
            # 処理済み
            disp_session_images(
                train_images_container,
                result_images_container,
                download_button_container,
            )
            disp_metrics(tab_metrics, metrics_containers)
        return

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

            logger.debug(f"creating dataset root={constants.DATASET_PATH}")
            # テスト用
            folder_dataset_test = FolderDataset(
                name="folder_dataset_test",
                root=constants.DATASET_PATH,
                normal_dir="test",
                # splitをtrainにすることでラベルを「正常」としてnormal_dirのデータを読み込み
                split="train",
            )
            logger.debug("folder_dataset_test created")

            train_predictions = None
            st.session_state["train_predictions"] = None
            predictions = None
            threshold = 0
            batch_size = st.session_state["batch_size"]
            num_workers = st.session_state["num_workers"]

            if st.session_state["chk_model_file"]:
                logger.debug(
                    "Using model file:", st.session_state["model_file"]
                )
                # モデルファイルを使用する場合
                # 学習済みモデルを読み込む
                model_file = st.session_state["model_file"]
                logger.debug(f"Loading model from {model_file}")
                loaded_model = torch.load(model_file, weights_only=False)
                logger.debug(loaded_model)
                model = loaded_model["model"]
                engine = Engine()
                predictions = engine.predict(
                    model=model, dataset=folder_dataset_test
                )
                if st.session_state["threshold_auto"]:
                    logger.debug(
                        "Calculating threshold automatically from model file..."
                    )
                    # モデルファイルからしきい値を取得
                    threshold = loaded_model["threshold"]
                else:
                    # 入力したしきい値
                    threshold = st.session_state["threshold"]

                # モデルファイルをコピー
                exported_model = copy.deepcopy(loaded_model)
                # しきい値をモデルファイルに保存
                exported_model["threshold"] = threshold
                Path(constants.MODEL_PATH).parent.mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(exported_model, constants.MODEL_PATH)
                logger.debug(
                    f"Threshold saved to model file: {constants.MODEL_PATH}"
                )

            else:
                logger.debug(
                    "No model file provided. Training a new model from scratch."
                )
                # モデルファイルを使用しない場合
                # Load the selected model
                model = models.get_model(
                    model_name=st.session_state["model_name"],
                    backbone=st.session_state["backbone"],
                    image_size=st.session_state["image_size"],
                    max_epochs=st.session_state["epochs"],
                    batch_size=batch_size,
                )

                if has_abnormal:
                    logger.debug(
                        f"Abnormal images provided. Using them for training. root={constants.DATASET_PATH}"
                    )
                    # 異常画像が選択されている場合
                    if isinstance(model, CFM):
                        # CFMの場合は3Dモデル使用
                        datamodule = Folder3D(
                            name="custom",
                            root=constants.DATASET_PATH,
                            # abnormal_dirとnormal_test_dirのデータがvalになる
                            normal_dir=Path("train") / "normal",
                            abnormal_dir=Path("train") / "abnormal",
                            mask_dir=Path("train") / "mask",
                            # TODO 3Dモデル
                            # normal_depth_dir=Path("depth") / "normal",
                            # abnormal_depth_dir=Path("depth") / "abnormal",
                            # normal_test_depth_dir=Path("depth") / "test",
                            normal_test_dir=Path("train") / "normal",
                            val_split_mode=ValSplitMode.SAME_AS_TEST,
                            train_batch_size=batch_size,
                            eval_batch_size=batch_size,
                            num_workers=num_workers,
                        )
                    else:
                        datamodule = Folder(
                            name="custom",
                            root=constants.DATASET_PATH,
                            # abnormal_dirとnormal_test_dirのデータがvalになる
                            normal_dir=Path("train") / "normal",
                            abnormal_dir=Path("train") / "abnormal",
                            mask_dir=Path("train") / "mask",
                            normal_test_dir=Path("train") / "normal",
                            val_split_mode=ValSplitMode.SAME_AS_TEST,
                            train_batch_size=batch_size,
                            eval_batch_size=batch_size,
                            num_workers=num_workers,
                        )
                else:
                    logger.debug(
                        f"No abnormal images provided. Training with only normal images. root={constants.DATASET_PATH}"
                    )
                    # 異常画像が選択されていない場合
                    if isinstance(model, CFM):
                        datamodule = Folder3D(
                            name="custom",
                            root=constants.DATASET_PATH,
                            normal_dir=Path("train") / "normal",
                            # TODO 3Dモデル
                            # normal_depth_dir=Path("depth") / "normal",
                            # abnormal_depth_dir=Path("depth") / "abnormal",
                            # normal_test_depth_dir=Path("depth") / "test",
                            normal_test_dir="test",
                            # test_split_mode=TestSplitMode.SYNTHETIC,
                            # val_split_mode=ValSplitMode.SYNTHETIC,
                            val_split_mode=ValSplitMode.FROM_TRAIN,
                            # 検証データを入れるとスコアが1か0になるため、検証はしない
                            # ratioを0にするとデフォルト値で分割されるため、非常に小さい値を設定
                            val_split_ratio=0.0001,
                            train_batch_size=batch_size,
                            eval_batch_size=batch_size,
                            num_workers=num_workers,
                        )
                    else:
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
                            train_batch_size=batch_size,
                            eval_batch_size=batch_size,
                            num_workers=num_workers,
                        )
                datamodule.setup()

                logger.debug("Training model...")
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
                    logger.exception("Error during training: %s", e)
                    logger.info("Retrying training...")
                    # retry
                    engine.fit(model=model, datamodule=datamodule)
                logger.debug("Training completed.")

                # モデル保存
                engine.export(
                    model=model,
                    export_type=ExportType.TORCH,
                    export_root=constants.RESULT_PATH,
                )
                logger.debug(f"Model exported to: {constants.RESULT_PATH}")

                if has_abnormal:
                    logger.debug(
                        "Predicting on training data for threshold calculation..."
                    )
                    # 異常画像が選択されている場合で、自動しきい値またはパフォーマンス表示がチェックされている場合
                    # パフォーマンスとしきい値用に全件predict
                    if (
                        st.session_state["threshold_auto"]
                        or st.session_state["chk_disp_metrics"]
                    ):
                        if isinstance(model, AnomalyVFM):
                            # AnomalyVFMの場合はそのままdatamoduleを渡す
                            train_predictions = engine.predict(
                                model=model, datamodule=datamodule
                            )
                        else:
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
                    logger.debug(
                        "Predicting on training data for threshold calculation (no abnormal images)..."
                    )
                    # モデルファイルを使用せずにしきい値自動の場合
                    # しきい値用にtrainをpredict
                    train_predictions = engine.predict(
                        model=model,
                        dataloaders=datamodule.train_dataloader(),
                    )
                    st.session_state["train_predictions"] = train_predictions

                # しきい値
                if st.session_state["threshold_auto"]:
                    logger.debug("Calculating threshold automatically...")
                    # しきい値自動
                    if (
                        has_abnormal
                        and train_predictions is not None
                        and 0 < len(train_predictions)
                    ):
                        logger.debug(
                            "Calculating threshold using F1AdaptiveThreshold..."
                        )
                        # 異常画像が選択されている場合
                        try:
                            adaptiveThreshold = F1AdaptiveThreshold(
                                fields=["pred_score", "gt_label"]
                            )
                            for batch in train_predictions:  # type: ignore
                                adaptiveThreshold.update(batch)  # type: ignore
                            threshold = adaptiveThreshold.compute()
                        except Exception as e:
                            logger.exception(
                                "Exception from F1AdaptiveThreshold: %s", e
                            )
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
                        logger.debug(
                            f"threshold from F1AdaptiveThreshold: {threshold}"
                        )
                    else:
                        logger.debug(
                            "Calculating threshold using mean + 3*std (no abnormal images)..."
                        )
                        # 異常画像が選択されていない場合
                        train_scores = [
                            get_item(prediction, "pred_score")
                            for prediction in (train_predictions or [])
                        ]
                        # 閾値（99.7％）
                        threshold = np.mean(train_scores) + 3 * np.std(train_scores)  # type: ignore
                        logger.debug(
                            f"threshold from mean + 3*std: {threshold}"
                        )
                else:
                    # 入力したしきい値
                    threshold = st.session_state["threshold"]
                    logger.debug(f"threshold from user input: {threshold}")

                # しきい値をモデルファイルに保存
                if Path.exists(constants.MODEL_PATH):
                    try:
                        exported_model = torch.load(
                            constants.MODEL_PATH, weights_only=False
                        )
                        exported_model["threshold"] = threshold
                        torch.save(exported_model, constants.MODEL_PATH)
                        logger.debug(
                            f"Threshold saved to model file: {constants.MODEL_PATH}"
                        )
                    except Exception as e:
                        logger.exception(
                            "Error saving threshold to model file: %s", e
                        )

                # 予想
                if (
                    isinstance(model, AnomalyVFM)
                    or isinstance(model, WinClip)
                    or isinstance(model, VlmAd)
                ):
                    logger.debug(
                        f"Model is AnomalyVFM or WinClip or VlmAd, using Folder dataset for prediction... root={constants.DATASET_PATH}"
                    )
                    # AnomalyVFM、VLM-AD、WinClipはFolderDatasetに対応していないためFolderで再度読み込み
                    datamodule_test = Folder(
                        name="custom_test",
                        root=constants.DATASET_PATH,
                        normal_dir=Path("test"),
                        val_split_mode=ValSplitMode.SAME_AS_TEST,
                        val_split_ratio=0.0001,
                        normal_test_dir="test",
                        train_batch_size=batch_size,
                        eval_batch_size=batch_size,
                        num_workers=num_workers,
                    )
                    datamodule_test.setup()
                    predictions = engine.predict(
                        model=model, datamodule=datamodule_test
                    )
                else:
                    logger.debug("Predicting on test dataset...")
                    predictions = engine.predict(
                        model=model, dataset=folder_dataset_test
                    )

            # 結果描画
            create_result_data(predictions, threshold)
            disp_session_images(
                train_images_container,
                result_images_container,
                download_button_container,
            )
            show_result_tab()
            disp_metrics(tab_metrics, metrics_containers)

            end = time.time()
            elapsed_time = end - start
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            st.success(
                f"処理完了 ({minutes} minutes, {seconds:.1f} seconds)",
                icon="✔️",
            )
            logger.debug(f"elapsed_time: {elapsed_time}")
        except Exception as e:
            logger.exception("Error occurred during processing: %s", e)
            end = time.time()
            elapsed_time = end - start
            minutes = int(elapsed_time // 60)
            seconds = elapsed_time % 60
            st.error(
                f"Encountered an error: {e}\n({minutes} minutes, {seconds:.1f} seconds)",
                icon="❌",
            )
            init_results()
