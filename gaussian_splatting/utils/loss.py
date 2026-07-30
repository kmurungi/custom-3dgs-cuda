import torch as t




def calculate_loss(gt_img: t.Tensor, r_img: t.Tensor) -> float:
    """
    Calculates loss based on ground truth image and rendered img

    Args: 
        gt_img: ground truth (H, W) tensor
        r_img: rendered (H, W) tensor

    Return: 
        loss
    """
