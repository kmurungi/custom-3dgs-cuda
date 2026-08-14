import torch as t


def _spatial_gradients(img: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Channel-last (H, W, C) finite differences along width and height."""
    dx = img[:, 1:, :] - img[:, :-1, :]
    dy = img[1:, :, :] - img[:-1, :, :]
    return dx, dy


def calculate_loss(
    gt_img: t.Tensor,  # (H, W, 3)
    r_albedo: t.Tensor,  # (H, W, 3)
    r_shading: t.Tensor,  # (H, W, 1) or (H, W, 3)
    r_img: t.Tensor = None,  # Optional (H, W, 3). If None, compute as r_albedo * r_shading
    lambda_albedo: float = 0.1,
    lambda_shading: float = 0.05,
    lambda_chrom: float = 0.02,
) -> t.Tensor:
    """
    Reconstruction + unsupervised Intrinsic Image Decomposition losses.

    Combines L1 reconstruction with image-guided albedo smoothness,
    shading smoothness, and optional chromaticity regularization when
    shading is RGB.

    Args:
        gt_img: ground truth (H, W, 3) in [0, 1]
        r_albedo: rendered albedo (H, W, 3)
        r_shading: rendered shading (H, W, 1) or (H, W, 3)
        r_img: optional reconstructed image; defaults to albedo * shading
        lambda_albedo: weight for albedo smoothness
        lambda_shading: weight for shading smoothness
        lambda_chrom: weight for chromaticity regularization

    Returns:
        scalar total loss
    """
    if r_img is None:
        r_img = r_albedo * r_shading

    if r_img.shape != gt_img.shape:
        raise ValueError(
            f"Shape mismatch: gt {tuple(gt_img.shape)} vs rendered {tuple(r_img.shape)}"
        )

    # L_recon: L1 reconstruction
    l_recon = t.mean(t.abs(gt_img - r_img))

    # L_albedo: image-guided albedo smoothness
    albedo_dx, albedo_dy = _spatial_gradients(r_albedo)
    gt_dx, gt_dy = _spatial_gradients(gt_img)
    weight_x = t.exp(-10.0 * t.abs(gt_dx).mean(dim=-1, keepdim=True))
    weight_y = t.exp(-10.0 * t.abs(gt_dy).mean(dim=-1, keepdim=True))
    l_albedo = t.mean(weight_x * t.abs(albedo_dx)) + t.mean(weight_y * t.abs(albedo_dy))

    # L_shading: L2 squared gradient penalty
    shading_dx, shading_dy = _spatial_gradients(r_shading)
    l_shading = t.mean(shading_dx ** 2) + t.mean(shading_dy ** 2)

    # L_chrom: encourage monochromatic shading when RGB
    if r_shading.shape[-1] == 3:
        l_chrom = t.mean(t.var(r_shading, dim=-1))
    else:
        l_chrom = t.tensor(0.0, device=gt_img.device, dtype=gt_img.dtype)

    return (
        l_recon
        + lambda_albedo * l_albedo
        + lambda_shading * l_shading
        + lambda_chrom * l_chrom
    )
