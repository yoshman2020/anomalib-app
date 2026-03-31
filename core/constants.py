from pathlib import Path

# 選択した画像の保存フォルダ
DATASET_PATH = "./datasets/uploaded"
# 検査結果フォルダ
RESULT_PATH = "./results"
# モデルファイル
MODEL_PATH = Path(RESULT_PATH) / "weights" / "torch" / "model.pt"
# バッチサイズ
BATCH_SIZE = 1

# 画像表示カラムの高さ
COLUMN_HEIGHT = 300
COLUMN_HEIGHT_RESULT = 500

# 検査手法リスト
MODEL_NAMES = [
    # 0 # △一般に高速（データ規模依存） # 観測変数の因子構造を検証する統計的手法(Continuous Flow Analysis)
    # SUPPORTED_BACKBONES = ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")
    "CFA",
    # 1 # △アプローチ・条件で異なる # 連続確率を用いて離散生成モデルを拡張するフローマッチング手法(Conditional Flow Matching)
    "C-Flow",
    # 2 # △アプローチ・条件で異なる # 連続状態離散フローマッチングモデル(Continuous-State Flow Matching)
    # backboneなし
    "CS-Flow",
    # 3 # △大規模データ場合は処理時間が伸びる # 高次元データの外れ値検知向けのカーネル密度推定モデル(Distribution-Free Kernel Density Estimation)
    "DFKDE",
    # 4 # ×時間がかかる # 生成的フローマッチング(Deep Feature Matching)型モデル(Deep Flow Matching)
    "DFM",
    # 5 # ×時間がかかる # 深層自己監督と再構成で異常検知を行うモデル(Dual Reconstruction AutoEncoder-based Model)
    # backboneなし
    "DRAEM",
    # 6 # ◯比較的高速 # 正規化フローを用いて外れ値を検知するモデル(Deep Subspace Reconstruction)
    # backboneなし
    "DSR",
    # 7 # ◎非常に高速 # 計算効率重視の異常検知アルゴリズム(Efficient Anomaly Detection)
    # backboneなし
    "Efficient AD",
    # 8 # ◯GPU推論で高速 # 高速流ベース生成による異常検知法
    # SUPPORTED_BACKBONES = ("cait_m48_448", "deit_base_distilled_patch16_384", "resnet18", "wide_resnet50_2")
    "FastFlow",
    # 9 # ◎高速 # 再構成誤差に基づく異常検知ネットワーク(Feature Reconstruction Error)
    "FRE",
    # 10 # ×時間がかかる # 生成対向ネットワーク(GAN)による再構成誤差を活用する異常検知モデル(Generative Adversarial Network Anomaly Detection)
    # backboneなし
    "GANomaly",
    # 11 # △構造や実装でばらつきあり # 多変量分布で特徴空間の異常を検出するモデル(Patch Distribution Modeling)
    "PaDiM",
    # 12 # ◎非常に高速 # 高次元特徴空間におけるパッチベースの異常検知モデル
    "PatchCore",
    # 13 # ◯高速 # 教師あり逆蒸留法を用いた異常検知モデル
    "Reverse Distillation",
    # 14 # △構造や実装でばらつきあり # 教師なし空間的注意機構(フローベースパッチマッチング)を用いた異常検知モデル(Student-Teacher Feature Pyramid Matching)
    "STFPM",
    # 15 # ◎非常に高速 # シンプルで効果的な異常検知ニューラルネットワーク
    "SuperSimpleNet",
    # 16 # △アプローチ・条件で異なる # 光フロー(U-Net構造)を用いた異常検知モデル(U-Net-based Flow)
    # AVAILABLE_EXTRACTORS = ["mcait", "resnet18", "wide_resnet50_2"]
    "U-Flow",
    # 17 # △推論時間長め # 大規模視覚言語モデルを用いた異常検知モデル(Vision-Language Model for Anomaly Detection)
    # backboneなし
    "VLM-AD",
    # 18 # △推論時間長め # ウィンドウ注意機構を用いたCLIPベースの異常検知モデル(Windowed CLIP)
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
]

# 検査手法とモデル（バッグボーン）の対応
MODEL_BACKBONES = {
    "CFA": ["vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5"],
    "C-Flow": BACKBONES,
    "CS-Flow": [],
    "DFKDE": BACKBONES,
    "DFM": BACKBONES,
    "DRAEM": [],
    "DSR": [],
    "Efficient AD": [],
    "FastFlow": [
        "cait_m48_448",
        "deit_base_distilled_patch16_384",
        "resnet18",
        "wide_resnet50_2",
    ],
    "FRE": BACKBONES,
    "GANomaly": [],
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
        "CFA",
        "C-Flow",
        "CS-Flow",
        "DFKDE",
        "DFM",
        "DRAEM",
        "DSR",
        "Efficient AD",
        "FastFlow",
        "FRE",
        "GANomaly",
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
        "△一般に高速（データ規模依存）",
        "△アプローチ・条件で異なる",
        "△アプローチ・条件で異なる",
        "△大規模データ場合は処理時間が伸びる",
        "×時間がかかる",
        "×時間がかかる",
        "◯比較的高速",
        "◎非常に高速",
        "◯GPU推論で高速",
        "◎高速",
        "×時間がかかる",
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
        "観測変数の因子構造を検証する統計的手法(Continuous Flow Analysis)",
        "連続確率を用いて離散生成モデルを拡張するフローマッチング手法(Conditional Flow Matching)",
        "連続状態離散フローマッチングモデル(Continuous-State Flow Matching)",
        "高次元データの外れ値検知向けのカーネル密度推定モデル(Distribution-Free Kernel Density Estimation)",
        "生成的フローマッチング(Deep Feature Matching)型モデル(Deep Flow Matching)",
        "深層自己監督と再構成で異常検知を行うモデル(Dual Reconstruction AutoEncoder-based Model)",
        "正規化フローを用いて外れ値を検知するモデル(Deep Subspace Reconstruction)",
        "計算効率重視の異常検知アルゴリズム(Efficient Anomaly Detection)",
        "高速流ベース生成による異常検知法",
        "再構成誤差に基づく異常検知ネットワーク(Feature Reconstruction Error)",
        "生成対向ネットワーク(GAN)による再構成誤差を活用する異常検知モデル(Generative Adversarial Network Anomaly Detection)",
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
    ],
}
