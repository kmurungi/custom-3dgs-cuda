import torch as t


def calculate_loss(gt_img: t.Tensor, r_img: t.Tensor) -> t.Tensor:
    """
    L1 loss between ground-truth and rendered images.

    Args:
        gt_img: ground truth (H, W, 3) tensor in [0, 1]
        r_img: rendered (H, W, 3) tensor

    Returns:
        scalar loss tensor
    """
    if r_img.shape != gt_img.shape:
        raise ValueError(
            f"Shape mismatch: gt {tuple(gt_img.shape)} vs rendered {tuple(r_img.shape)}"
        )



        
    return t.mean(t.abs(gt_img - r_img))
