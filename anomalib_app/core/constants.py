# import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_base_path() -> Path:
    """EXE実行時とpy実行時で、リソースファイルのパスの基準が異なるため、両方に対応する関数"""
    return Path(__file__).parent.parent.parent


BASE_PATH = get_base_path()
# 選択した画像の保存フォルダ
DATASET_PATH = BASE_PATH / "datasets" / "uploaded"
# 検査結果フォルダ
RESULT_PATH = BASE_PATH / "results"
# モデルファイル
MODEL_PATH = BASE_PATH / RESULT_PATH / "weights" / "torch" / "model.pt"
# バッチサイズ
BATCH_SIZE = 1

# 画像表示カラムの高さ
COLUMN_HEIGHT = 300
COLUMN_HEIGHT_RESULT = 350

# 検査手法リスト
MODEL_NAMES = [
    # # 0 # △推論時間長め # 視覚基盤モデルを活用した教師なし異常検知モデル(Anomaly Vision Foundation Model)
    # # backboneなし
    "AnomalyVFM",
    # 1 # △推論時間長め # DINOv2特徴表現を利用した異常検知モデル(Anomaly Detection with DINO)
    # "dinov2_vit_small_14"
    "AnomalyDINO",
    # 2 # △一般に高速（データ規模依存） # 観測変数の因子構造を検証する統計的手法(Continuous Flow Analysis)
    # SUPPORTED_BACKBONES = ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")
    "CFA",
    # TODO 3Dモデル
    # # 3 # △アプローチ・条件で異なる # Vision Transformer特徴とフローマッチングを組み合わせた異常検知モデル(Conditional Flow Matching)
    # # "vit_base_patch8_224.dino"
    # "CFM",
    # 4 # △アプローチ・条件で異なる # 連続確率を用いて離散生成モデルを拡張するフローマッチング手法(Conditional Flow Matching)
    "C-Flow",
    # 5 # △アプローチ・条件で異なる # 連続状態離散フローマッチングモデル(Continuous-State Flow Matching)
    # backboneなし
    "CS-Flow",
    # 6 # △大規模データ場合は処理時間が伸びる # 高次元データの外れ値検知向けのカーネル密度推定モデル(Distribution-Free Kernel Density Estimation)
    "DFKDE",
    # 7 # ×時間がかかる # 生成的フローマッチング(Deep Feature Matching)型モデル(Deep Flow Matching)
    "DFM",
    # 8 #△推論時間長め # DINO特徴空間における異常領域検出モデル(DINO-based Anomaly Detection)
    # backboneなし
    "Dinomaly",
    # 9 # ×時間がかかる # 深層自己監督と再構成で異常検知を行うモデル(Dual Reconstruction AutoEncoder-based Model)
    # backboneなし
    "DRAEM",
    # 10 # ◯比較的高速 # 正規化フローを用いて外れ値を検知するモデル(Deep Subspace Reconstruction)
    # backboneなし
    "DSR",
    # 11 # ◎非常に高速 # 計算効率重視の異常検知アルゴリズム(Efficient Anomaly Detection)
    # backboneなし
    "Efficient AD",
    # # 12 # △推論時間長め # 補完学習を利用したTransformerベース異常検知モデル(Inpainting Transformer)
    # # "dinov2reg_vit_base_14"
    "INP-Former",
    # 13 # ◯GPU推論で高速 # 高速流ベース生成による異常検知法
    # SUPPORTED_BACKBONES = ("cait_m48_448", "deit_base_distilled_patch16_384", "resnet18", "wide_resnet50_2")
    "FastFlow",
    # 14 # ◎高速 # 再構成誤差に基づく異常検知ネットワーク(Feature Reconstruction Error)
    "FRE",
    # 15 # ×時間がかかる # 生成対向ネットワーク(GAN)による再構成誤差を活用する異常検知モデル(Generative Adversarial Network Anomaly Detection)
    # backboneなし
    "GANomaly",
    # 16 # ◯高速 # ガラス状欠陥検査向けに設計された異常検知モデル(Glass Surface Anomaly Detection)
    # "wide_resnet50_2"
    "Glass",
    # 17 # ×時間がかかる # 汎用視覚基盤モデルを利用した異常検知モデル(General Anomaly Detection)
    # "vit_large_patch14_dinov2.lvd142m"
    "GeneralAD",
    # # 18 # △推論時間長め # Local-to-Global双方向Transformerによる異常検知モデル(Local-to-Bidirectional Transformer)
    # # backboneなし
    "L2BT",
    # 19 # △構造や実装でばらつきあり # 多変量分布で特徴空間の異常を検出するモデル(Patch Distribution Modeling)
    "PaDiM",
    # 20 # ◎非常に高速 # 高次元特徴空間におけるパッチベースの異常検知モデル
    "PatchCore",
    # 21 # ◯高速 # 教師あり逆蒸留法を用いた異常検知モデル
    "Reverse Distillation",
    # 22 # △構造や実装でばらつきあり # 教師なし空間的注意機構(フローベースパッチマッチング)を用いた異常検知モデル(Student-Teacher Feature Pyramid Matching)
    "STFPM",
    # 23 # ◎非常に高速 # シンプルで効果的な異常検知ニューラルネットワーク
    "SuperSimpleNet",
    # 24 # △アプローチ・条件で異なる # 光フロー(U-Net構造)を用いた異常検知モデル(U-Net-based Flow)
    # AVAILABLE_EXTRACTORS = ["mcait", "resnet18", "wide_resnet50_2"]
    "U-Flow",
    # 25 # △推論時間長め # 大規模視覚言語モデルを用いた異常検知モデル(Vision-Language Model for Anomaly Detection)
    # backboneなし
    "VLM-AD",
    # 26 # △推論時間長め # ウィンドウ注意機構を用いたCLIPベースの異常検知モデル(Windowed CLIP)
    # backboneなし
    "WinCLIP",
]

# モデル（バックボーン）リスト
BACKBONES = [
    # ResNet 一般的なモデル
    # 0 # ◎非常に高速
    "resnet18",
    # 1 # ◯高速
    "resnet50",
    # 2 # △やや遅い
    "resnet101",
    # 3 # ×遅め
    "resnet152",
    # Wide ResNet より広い層を持つモデル
    # 4 # ◯高速
    "wide_resnet50_2",
    # 5 # △やや遅い
    "wide_resnet101_2",
    # EfficientNet 軽量で高性能なモデル
    # 6 # ◎非常に高速
    "efficientnet_b0",
    # 7 # ◯高速
    "efficientnet_b3",
    # 8 # △やや遅め
    "efficientnet_b4",
    # 9 # △やや遅め
    "efficientnet_b5",
    # Vision Transformer (ViT) 画像のパッチを特徴量として取り込むモデル
    # 10 # △やや遅い
    "vit_base_patch16_224",
    # 11 # ×遅め
    "vit_large_patch16_224",
    # Swin Transformer 局所的な注意機構を持つモデル
    # 13 # △やや遅い
    "swin_base_patch4_window7_224",
    # 14 # ×遅め
    "swin_large_patch4_window7_224",
    # DenseNet 高密度な接続を持つモデル
    # 15 # ◯高速
    "densenet121",
    # 16 # ◯高速
    "densenet169",
    # 17 # △やや遅い
    "densenet201",
    # RegNet 効率的なアーキテクチャを持つモデル
    # 18 # ◎非常に高速
    "regnetx_002",
    # 19 # ◯高速
    "regnetx_004",
    # 20 # ◎非常に高速
    "regnety_032",
    # MobileNet 軽量なモデル
    # 21 # ◎非常に高速
    "mobilenetv2_100",
    # 22 # ◎非常に高速
    "mobilenetv3_large_100",
    # VGG シンプルで広く使われるモデル
    # 23 # △やや遅い
    "vgg19_bn",
    # ViT派生の大規模モデルで、大規模データセット向き(Class-Attention in Image Transformers)
    # 24 # ×時間がかかる
    "cait_m48_448",
    # 蒸留で軽量・効率化したViTモデル(Data-efficient Image Transformer Base Distilled Patch)
    # 25 # ×時間がかかる
    "deit_base_distilled_patch16_384",
    # マルチヘッド層と階層的注意機構で視覚タスクに高精度をもたらすVision Transformerモデル
    # 26 # △やや遅い
    "mcait",
    # DINOv2 自己教師あり学習による高性能な視覚特徴抽出モデル
    # 27 # ◯高速
    "dinov2_vit_small_14",
    # DINOv2 Register Tokenを導入した高精度視覚特徴抽出モデル
    # 28 # △やや遅い
    "dinov2reg_vit_base_14",
    # DINO 自己教師あり学習によるVision Transformerモデル
    # 29 # △やや遅い
    "vit_base_patch8_224.dino",
    # DINOv2 Large 大規模自己教師ありVision Transformerモデル
    # 30 # ×時間がかかる
    "vit_large_patch14_dinov2.lvd142m",
]

# 検査手法とモデル（バッグボーン）の対応
MODEL_BACKBONES = {
    "AnomalyVFM": [],
    "AnomalyDINO": ["dinov2_vit_small_14"],
    "CFA": ["vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5"],
    "CFM": ["vit_base_patch8_224.dino"],
    "C-Flow": BACKBONES,
    "CS-Flow": [],
    "DFKDE": BACKBONES,
    "DFM": BACKBONES,
    "Dinomaly": ["dinov2reg_vit_base_14"],
    "DRAEM": [],
    "DSR": [],
    "Efficient AD": [],
    "INP-Former": ["dinov2reg_vit_base_14"],
    "FastFlow": [
        "cait_m48_448",
        "deit_base_distilled_patch16_384",
        "resnet18",
        "wide_resnet50_2",
    ],
    "FRE": BACKBONES,
    "GANomaly": [],
    "Glass": ["wide_resnet50_2"],
    "GeneralAD": ["vit_large_patch14_dinov2.lvd142m"],
    "L2BT": [],
    "PaDiM": BACKBONES,
    "PatchCore": BACKBONES,
    "Reverse Distillation": BACKBONES,
    "STFPM": BACKBONES,
    "SuperSimpleNet": BACKBONES,
    "U-Flow": ["mcait", "resnet18", "wide_resnet50_2"],
    "VLM-AD": [],
    "WinCLIP": [],
}

# 検査手法について
ABOUT_MODEL_NAMES = {
    "名前": [
        "AnomalyVFM",
        "AnomalyDINO",
        "CFA",
        "CFM",
        "C-Flow",
        "CS-Flow",
        "DFKDE",
        "DFM",
        "Dinomaly",
        "DRAEM",
        "DSR",
        "Efficient AD",
        "INP-Former",
        "FastFlow",
        "FRE",
        "GANomaly",
        "Glass",
        "GeneralAD",
        "L2BT",
        "PaDiM",
        "PatchCore",
        "Reverse Distillation",
        "STFPM",
        "SuperSimpleNet",
        "U-Flow",
        "VLM-AD",
        "WinCLIP",
    ],
    "スピード": [
        "△推論時間長め",
        "△推論時間長め",
        "△一般に高速（データ規模依存）",
        "△アプローチ・条件で異なる",
        "△アプローチ・条件で異なる",
        "△アプローチ・条件で異なる",
        "△大規模データ場合は処理時間が伸びる",
        "×時間がかかる",
        "△推論時間長め",
        "×時間がかかる",
        "◯比較的高速",
        "◎非常に高速",
        "△推論時間長め",
        "◯GPU推論で高速",
        "◎高速",
        "×時間がかかる",
        "◯高速",
        "×時間がかかる",
        "△推論時間長め",
        "△構造や実装でばらつきあり",
        "◎非常に高速",
        "◯高速",
        "△構造や実装でばらつきあり",
        "◎非常に高速",
        "△アプローチ・条件で異なる",
        "△推論時間長め",
        "△推論時間長め",
    ],
    "説明": [
        "視覚基盤モデルを活用した教師なし異常検知モデル(Anomaly Vision Foundation Model)",
        "DINOv2特徴表現を利用した異常検知モデル(Anomaly Detection with DINO)",
        "観測変数の因子構造を検証する統計的手法(Continuous Flow Analysis)",
        "Vision Transformer特徴とフローマッチングを組み合わせた異常検知モデル(Conditional Flow Matching)",
        "連続確率を用いて離散生成モデルを拡張するフローマッチング手法(Conditional Flow Matching)",
        "連続状態離散フローマッチングモデル(Continuous-State Flow Matching)",
        "高次元データの外れ値検知向けのカーネル密度推定モデル(Distribution-Free Kernel Density Estimation)",
        "生成的フローマッチング(Deep Feature Matching)型モデル(Deep Flow Matching)",
        "DINO特徴空間における異常領域検出モデル(DINO-based Anomaly Detection)",
        "深層自己監督と再構成で異常検知を行うモデル(Dual Reconstruction AutoEncoder-based Model)",
        "正規化フローを用いて外れ値を検知するモデル(Deep Subspace Reconstruction)",
        "計算効率重視の異常検知アルゴリズム(Efficient Anomaly Detection)",
        "補完学習を利用したTransformerベース異常検知モデル(Inpainting Transformer)",
        "高速流ベース生成による異常検知法",
        "再構成誤差に基づく異常検知ネットワーク(Feature Reconstruction Error)",
        "生成対向ネットワーク(GAN)による再構成誤差を活用する異常検知モデル(Generative Adversarial Network Anomaly Detection)",
        "ガラス状欠陥検査向けに設計された異常検知モデル(Glass Surface Anomaly Detection)",
        "汎用視覚基盤モデルを利用した異常検知モデル(General Anomaly Detection)",
        "Local-to-Global双方向Transformerによる異常検知モデル(Local-to-Bidirectional Transformer)",
        "多変量分布で特徴空間の異常を検出するモデル(Patch Distribution Modeling)",
        "高次元特徴空間におけるパッチベースの異常検知モデル",
        "教師あり逆蒸留法を用いた異常検知モデル",
        "教師なし空間的注意機構(フローベースパッチマッチング)を用いた異常検知モデル(Student-Teacher Feature Pyramid Matching)",
        "シンプルで効果的な異常検知ニューラルネットワーク",
        "光フロー(U-Net構造)を用いた異常検知モデル(U-Net-based Flow)",
        "大規模視覚言語モデルを用いた異常検知モデル(Vision-Language Model for Anomaly Detection)",
        "ウィンドウ注意機構を用いたCLIPベースの異常検知モデル(Windowed CLIP)",
    ],
}

# モデル（バックボーン）について
ABOUT_BACKBONE = {
    "名前": [
        "resnet18",
        "resnet50",
        "resnet101",
        "resnet152",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "efficientnet_b0",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "vit_base_patch16_224",
        "vit_large_patch16_224",
        "swin_base_patch4_window7_224",
        "swin_large_patch4_window7_224",
        "densenet121",
        "densenet169",
        "densenet201",
        "regnetx_002",
        "regnetx_004",
        "regnety_032",
        "mobilenetv2_100",
        "mobilenetv3_large_100",
        "vgg19_bn",
        "cait_m48_448",
        "deit_base_distilled_patch16_384",
        "mcait",
        "dinov2_vit_small_14",
        "dinov2reg_vit_base_14",
        "vit_base_patch8_224.dino",
        "vit_large_patch14_dinov2.lvd142m",
    ],
    "スピード": [
        "◎非常に高速",
        "◯高速",
        "△やや遅い",
        "×遅め",
        "◯高速",
        "△やや遅い",
        "◎非常に高速",
        "◯高速",
        "△やや遅め",
        "△やや遅め",
        "△やや遅い",
        "×遅め",
        "△やや遅い",
        "×遅め",
        "◯高速",
        "◯高速",
        "△やや遅い",
        "◎非常に高速",
        "◯高速",
        "◎非常に高速",
        "◎非常に高速",
        "◎非常に高速",
        "△やや遅い",
        "×時間がかかる",
        "×時間がかかる",
        "△やや遅い",
        "◯高速",
        "△やや遅い",
        "△やや遅い",
        "×時間がかかる",
    ],
    "説明": [
        "ResNet 一般的なモデル",
        "",
        "",
        "",
        "Wide ResNet より広い層を持つモデル",
        "",
        "EfficientNet 軽量で高性能なモデル",
        "",
        "",
        "",
        "Vision Transformer (ViT) 画像のパッチを特徴量として取り込むモデル",
        "",
        "Swin Transformer 局所的な注意機構を持つモデル",
        "",
        "DenseNet 高密度な接続を持つモデル",
        "",
        "",
        "RegNet 効率的なアーキテクチャを持つモデル",
        "",
        "",
        "MobileNet 軽量なモデル",
        "",
        "VGG シンプルで広く使われるモデル",
        "ViT派生の大規模モデルで、大規模データセット向き(Class-Attention in Image Transformers)",
        "蒸留で軽量・効率化したViTモデル(Data-efficient Image Transformer Base Distilled Patch)",
        "マルチヘッド層と階層的注意機構で視覚タスクに高精度をもたらすVision Transformerモデル",
        "DINOv2 自己教師あり学習による高性能な視覚特徴抽出モデル",
        "DINOv2 Register Tokenを導入した高精度視覚特徴抽出モデル",
        "DINO 自己教師あり学習によるVision Transformerモデル",
        "DINOv2 Large 大規模自己教師ありVision Transformerモデル",
    ],
}
