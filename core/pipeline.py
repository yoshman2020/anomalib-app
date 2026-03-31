import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from anomalib.data import Folder, FolderDataset
from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from anomalib.engine.engine import Engine
from anomalib.metrics import F1AdaptiveThreshold
from anomalib.models import VlmAd, WinClip

import core.constants as constants
import core.models as models
from core.io_utils import save_images
from core.list_utils import get_item
from core.results import create_result_data_cached, init_results
from core.validation import is_input_ok
from ui.display import disp_session_images
from ui.metrics_ui import disp_metrics


def main_page(
    submitted: bool,
    result_images_placeholder,
    train_images_placeholder,
    download_button_placeholder,
    tab_metrics,
    metrix_containers,
) -> None:
    """Main page layout and logic for generating images.

    Args:
        submitted (bool): Flag indicating whether the form has been submitted.
    """
    print("main_page called with submitted =", submitted)

    if not submitted:
        if "str_results" in st.session_state and 0 < len(
            st.session_state["str_results"]
        ):
            # 処理済み
            disp_session_images(
                result_images_placeholder,
                train_images_placeholder,
                download_button_placeholder,
            )
            disp_metrics(tab_metrics, metrix_containers)
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

            # テスト用
            folder_dataset_test = FolderDataset(
                name="folder_dataset_test",
                normal_dir=Path(constants.DATASET_PATH) / "test",
                # splitをtrainにすることでラベルを「正常」としてnormal_dirのデータを読み込み
                split="train",
            )

            train_predictions = None
            st.session_state["train_predictions"] = None
            predictions = None
            threshold = 0

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
                    st.session_state["train_predictions"] = train_predictions
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
                    st.session_state["train_predictions"] = train_predictions

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
            create_result_data_cached(predictions, threshold)
            disp_session_images(
                result_images_placeholder,
                train_images_placeholder,
                download_button_placeholder,
            )
            disp_metrics(tab_metrics, metrix_containers)

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
