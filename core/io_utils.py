import shutil
from pathlib import Path

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from core import constants


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
