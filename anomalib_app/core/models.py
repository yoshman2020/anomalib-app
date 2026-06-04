#!/usr/bin/env python3
import logging
from typing import Any

import timm
from anomalib import PrecisionType
from anomalib.metrics import Evaluator
from anomalib.models import (
    AnomalyVFM,
    AnomalyDINO,
    Cfa,
    CFM,
    Cflow,
    Csflow,
    Dfkde,
    Dfm,
    Dinomaly,
    Draem,
    Dsr,
    EfficientAd,
    InpFormer,
    Fastflow,
    Fre,
    Ganomaly,
    Glass,
    GeneralAD,
    L2BT,
    Padim,
    Patchcore,
    ReverseDistillation,
    Stfpm,
    Supersimplenet,
    Uflow,
    VlmAd,
    WinClip,
)
from anomalib.models.components import AnomalibModule
from anomalib.models.components.classification import FeatureScalingMethod
from anomalib.models.image.efficient_ad.torch_model import EfficientAdModelSize
from anomalib.models.image.reverse_distillation.anomaly_map import (
    AnomalyMapGenerationMode,
)
from anomalib.models.image.vlm_ad.utils import ModelName
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer
from torchvision.transforms.v2 import Compose, Resize

logger = logging.getLogger(__name__)


class FreEx(Fre):
    """
    FreEx extends the Fre model with additional configuration for training epochs.
    Args:
        max_epochs (int): Maximum number of training epochs. Defaults to 220.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        layer: str = "layer3",
        pre_trained: bool = True,
        pooling_kernel_size: int = 2,
        input_dim: int = 65536,
        latent_dim: int = 220,
        pre_processor: PreProcessor | bool = True,
        post_processor: PostProcessor | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
        max_epochs: int = 220,
    ) -> None:
        super().__init__(
            backbone,
            layer,
            pre_trained,
            pooling_kernel_size,
            input_dim,
            latent_dim,
            pre_processor,
            post_processor,
            evaluator,
            visualizer,
        )
        self.max_epochs = max_epochs

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Return FRE-specific trainer arguments.

        Returns:
            dict[str, Any]: Dictionary of trainer arguments:
                - ``gradient_clip_val``: ``0``
                - ``max_epochs``: ``220``
                - ``num_sanity_val_steps``: ``0``
        """
        super_trainer_arguments = super().trainer_arguments
        super_trainer_arguments["max_epochs"] = self.max_epochs
        return super_trainer_arguments


def get_model(
    model_name: str,
    backbone: str,
    image_size: int = 256,
    max_epochs: int = 1,
    batch_size: int = 1,
) -> AnomalibModule:
    """
    Creates and returns an anomaly detection model based on the specified model name and backbone.
    Args:
        model_name (str): The name of the anomaly detection model to instantiate. Supported values include
            "CFA", "C-Flow", "CS-Flow", "DFKDE", "DFM", "DRAEM", "DSR", "Efficient AD", "FastFlow", "FRE",
            "GANomaly", "PaDiM", "PatchCore", "Reverse Distillation", "STFPM", "SuperSimpleNet", "U-Flow",
            "VLM-AD", "WinCLIP".
        backbone (str): The name of the backbone neural network architecture to use for feature extraction.
        image_size (int, optional): The size to which input images will be resized. Defaults to 256.
        max_epochs (int, optional): The maximum number of training epochs. Defaults to 1.
        batch_size (int, optional): The batch size for training. Defaults to 1.
    Returns:
        AnomalibModule: An instance of the specified anomaly detection model, configured with the given backbone
        and preprocessing settings.
    Raises:
        ValueError: If an unknown model name is provided.
    """

    if backbone is None or backbone in [
        "",
        "mcait",
        "dinov2_vit_small_14",
        "vit_base_patch8_224.dino",
        "dinov2reg_vit_base_14",
        "vit_large_patch14_dinov2.lvd142m",
    ]:
        layers = []
    else:
        feature_model = timm.create_model(backbone, features_only=True)
        layers = feature_model.feature_info.module_name()  # type: ignore
        logger.debug(layers)

    # 前処理の設定
    transform = Compose([Resize((image_size, image_size))])
    pre_processor = PreProcessor(transform=transform)

    post_processor = True
    evaluator = True

    # Visualizerは使用しない
    visualizer = False

    # ----- モデル構築 -----
    if model_name == "AnomalyVFM":
        model = AnomalyVFM(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            precision=PrecisionType.FLOAT32,
        )
    elif model_name == "AnomalyDINO":
        model = AnomalyDINO(
            num_neighbours=1,
            encoder_name=backbone,
            masking=False,
            coreset_subsampling=False,
            sampling_ratio=0.1,
            precision=PrecisionType.FLOAT32,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
    elif model_name == "CFA":
        model = Cfa(
            backbone=backbone,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            gamma_c=1,
            gamma_d=1,
            num_nearest_neighbors=3,
            num_hard_negative_features=3,
            radius=1e-5,
        )
        # max_epochs = 30
        # callbacks = [EarlyStopping(patience=5, monitor="pixel_AUROC", mode="max")]
    elif model_name == "CFM":
        model = CFM(
            lr=0.0001,
            rgb_backbone=backbone,
            group_size=128,
            num_group=1024,
            pointmae_weights=None,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
    elif model_name == "C-Flow":
        model = Cflow(
            backbone=backbone,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            layers=layers[-3:],
            fiber_batch_size=batch_size,
            pre_trained=True,
            decoder="freia-cflow",
            condition_vector=128,
            coupling_blocks=8,
            clamp_alpha=1.9,
            permute_soft=False,
            lr=0.0001,
        )
        # max_epochs = 50
        # callbacks = [EarlyStopping(patience=5, monitor="pixel_AUROC", mode="max")]
    elif model_name == "CS-Flow":
        model = Csflow(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            cross_conv_hidden_channels=1024,
            n_coupling_blocks=4,
            clamp=3,
            num_channels=3,
        )
        # max_epochs = 240
    elif model_name == "DFKDE":
        model = Dfkde(
            backbone=backbone,
            layers=[layers[-1]],
            # pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            pre_trained=True,
            # n_pca_components=16,
            n_pca_components=1,
            feature_scaling_method=FeatureScalingMethod.SCALE,
            max_training_points=40000,
        )
        # max_epochs = 1
        # callbacks = [EarlyStopping(monitor="pixel_AUROC", mode="max")]
    elif model_name == "DFM":
        model = Dfm(
            backbone=backbone,
            layer=layers[-2],
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            pre_trained=True,
            pooling_kernel_size=4,
            pca_level=0.97,
            score_type="fre",
        )
        # max_epochs = 1
    elif model_name == "Dinomaly":
        model = Dinomaly(
            encoder_name=backbone,
            bottleneck_dropout=0.2,
            decoder_depth=8,
            target_layers=None,
            fuse_layer_encoder=None,
            fuse_layer_decoder=None,
            remove_class_token=False,
            use_context_recentering=False,
            precision=PrecisionType.FLOAT32,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
    elif model_name == "DRAEM":
        model = Draem(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            enable_sspcab=False,
            sspcab_lambda=0.1,
            # anomaly_source_path=None,
            beta=(0.1, 1.0),
        )
        # max_epochs = 700
        # callbacks = [EarlyStopping(patience=20, monitor="pixel_AUROC", mode="max")]
    elif model_name == "DSR":
        model = Dsr(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            latent_anomaly_strength=0.2,
            upsampling_train_ratio=0.7,
        )
        # max_epochs = 700
    elif model_name == "Efficient AD":
        model = EfficientAd(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            imagenet_dir="./datasets/imagenette",
            teacher_out_channels=384,
            model_size=EfficientAdModelSize.S,
            lr=0.0001,
            weight_decay=0.00001,
            padding=False,
            pad_maps=True,
        )
        # max_epochs = 1000
    elif model_name == "InpFormer":
        model = InpFormer(
            encoder_name=backbone,
            target_layers=None,
            fuse_layer_encoder=None,
            fuse_layer_decoder=None,
            remove_class_token=True,
            inp_num=6,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
    elif model_name == "FastFlow":
        model = Fastflow(
            backbone=backbone,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            pre_trained=True,
            flow_steps=8,
            conv3x3_only=False,
            hidden_ratio=1.0,
        )
        # max_epochs = 500
        # callbacks = [EarlyStopping(monitor="pixel_AUROC", mode="max")]
    elif model_name == "FRE":
        model = FreEx(
            backbone=backbone,
            layer=layers[-2],
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            input_dim=image_size * image_size,
            latent_dim=image_size,
            pre_trained=True,
            pooling_kernel_size=2,
            max_epochs=max_epochs,
        )
        # max_epochs = 1220
    elif model_name == "GANomaly":
        model = Ganomaly(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            batch_size=32,
            n_features=64,
            latent_vec_size=100,
            extra_layers=0,
            add_final_conv_layer=True,
            wadv=1,
            wcon=50,
            wenc=1,
            lr=0.0002,
            beta1=0.5,
            beta2=0.999,
        )
        # max_epochs = 100
        # callbacks = [EarlyStopping(monitor="image_AUROC", mode="max")]
    elif model_name == "Glass":
        model = Glass(
            input_shape=(288, 288),
            anomaly_source_path=None,
            backbone=backbone,
            pretrain_embed_dim=1536,
            target_embed_dim=1536,
            patchsize=3,
            patchstride=1,
            pre_trained=True,
            layers=None,
            pre_projection=1,
            discriminator_layers=2,
            discriminator_hidden=1024,
            learning_rate=0.0001,
            step=20,
            svd=0,
            gaussian_noise_std=0.015,
            radius_quantile=0.75,
            focal_loss_quantile_threshold=0.5,
            mining=True,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
    elif model_name == "GeneralAD":
        model = GeneralAD(
            backbone=backbone,
            layers=(24,),
            hidden_dim=2048,
            lr=0.0005,
            lr_decay_factor=0.2,
            weight_decay=0.00001,
            epochs=160,
            noise_std=0.25,
            dsc_layers=1,
            dsc_heads=4,
            dsc_dropout=0.1,
            num_fake_patches=-1,
            fake_feature_type="random",
            top_k=10,
            pre_trained=True,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
    elif model_name == "L2BT":
        model = L2BT(
            lr=0.0001,
            layers=(7, 11),
            blur_w_l=5,
            blur_w_u=7,
            blur_pad_l=2,
            blur_pad_u=3,
            blur_repeats_l=5,
            blur_repeats_u=3,
            topk_ratio=0.001,
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
    elif model_name == "PaDiM":
        model = Padim(
            backbone=backbone,
            layers=layers[-4:-1],
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            n_features=100,
            pre_trained=True,
        )
        # max_epochs = 1
    elif model_name == "PatchCore":
        model = Patchcore(
            backbone=backbone,
            layers=layers[-3:-1],
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            pre_trained=True,
            coreset_sampling_ratio=0.1,
            num_neighbors=9,
        )
        # max_epochs = 1
    elif model_name == "Reverse Distillation":
        model = ReverseDistillation(
            backbone=backbone,
            layers=layers[-4:-1],
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            anomaly_map_mode=AnomalyMapGenerationMode.ADD,
            pre_trained=True,
        )
        # max_epochs = 200
        # callbacks = [EarlyStopping(monitor="pixel_AUROC", mode="max")]
    elif model_name == "STFPM":
        model = Stfpm(
            backbone=backbone,
            layers=layers[-4:-1],
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        # max_epochs = 100
        # callbacks = [EarlyStopping(patience=5, monitor="pixel_AUROC", mode="max")]
    elif model_name == "SuperSimpleNet":
        model = Supersimplenet(
            backbone=backbone,
            layers=layers[-3:-1],
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            perlin_threshold=0.2,
            supervised=False,
        )
        # max_epochs = 1
    elif model_name == "U-Flow":
        model = Uflow(
            backbone=backbone,
            # pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            flow_steps=4,
            affine_clamp=2.0,
            affine_subnet_channels_ratio=1.0,
            permute_soft=False,
        )
        # max_epochs = 200
        # callbacks = [EarlyStopping(patience=20, monitor="pixel_AUROC", mode="max")]
    elif model_name == "VLM-AD":
        model = VlmAd(
            model=ModelName.LLAMA_OLLAMA,
            api_key=None,
            k_shot=0,
        )
        # max_epochs = 1
    elif model_name == "WinCLIP":
        model = WinClip(
            # pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
            class_name="transistor",
            k_shot=0,
            scales=(2, 3),
            few_shot_source=None,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
