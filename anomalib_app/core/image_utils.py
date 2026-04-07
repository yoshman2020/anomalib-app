import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def superimpose_anomaly_map_g(
    anomaly_map: np.ndarray,
    image: np.ndarray,
    map_min: float,
    map_ptp: float,
    alpha: float = 0.4,
    gamma: int = 0,
) -> np.ndarray:
    """Superimpose anomaly map on image.

    Args:
        anomaly_map (np.ndarray): Anomaly map.
        image (np.ndarray): Image.
        alpha (float): Alpha value for superimposition.
        gamma (int): Gamma value for superimposition.
        map_min (float): Minimum value of the anomaly map.
        map_ptp (float): Range of the anomaly map.

    Returns:
        np.ndarray: Superimposed image.
    """
    logger.debug("superimpose_anomaly_map_g called")
    assert anomaly_map.shape == image.shape[:2], (
        f"Anomaly map shape {anomaly_map.shape} does not match image shape "
        f"{image.shape[:2]}."
    )
    nomalized_map = (((anomaly_map - map_min) / map_ptp) * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(nomalized_map, cv2.COLORMAP_JET)
    rgb_color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    height, width = rgb_color_map.shape[:2]
    image = cv2.resize(image, (width, height))
    logger.debug("superimpose_anomaly_map_g finished")
    return cv2.addWeighted(rgb_color_map, alpha, image, (1 - alpha), gamma)
