import numpy as np
import torch


def get_item(prediction, key):
    """
    Retrieve the value associated with a given key from a prediction object, handling various data types.
    Parameters:
        prediction (object): The object containing the attribute to retrieve.
        key (str): The attribute name to access within the prediction object.
    Returns:
        The first element or scalar value associated with the specified key, depending on the type:
            - For lists or tuples: returns the first element if available, else None.
            - For torch.Tensor: returns None if empty, the scalar value if single element, or the first element otherwise.
            - For numpy.ndarray: returns None if empty, the scalar value if single element, or the first element otherwise.
            - For other types (int, float, etc.): returns the value directly.
        Returns None if the attribute does not exist.
    """
    print(f"get_item called for key: {key}")
    if not hasattr(prediction, key):
        return None

    val = getattr(prediction, key)

    # list or tuple
    if isinstance(val, (list, tuple)):
        return val[0] if len(val) > 0 else None

    # PyTorch Tensor
    if isinstance(val, torch.Tensor):
        if val.numel() == 0:
            return None
        if val.numel() == 1:
            return val.item()
        return val[0] if val.dim() > 0 else val.item()

    # NumPy ndarray
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return None
        if val.size == 1:
            return val.item()
        return val.flat[0]  # flat iteratorで最初の要素

    print("get_item finished")
    # それ以外（int, float, etc.）
    return val
