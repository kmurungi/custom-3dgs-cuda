import torch as t
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

LAMBDA_DSSIM = 0.2


def _to_nchw(img_hwc: t.Tensor) -> t.Tensor:
    """HWC float image in [0, 1] → NCHW for torchmetrics."""
    return img_hwc.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)


def calculate_loss(
    gt_img: t.Tensor,
    rendered_img: t.Tensor,
    lambda_dssim: float = LAMBDA_DSSIM,
) -> t.Tensor:
    """Classic 3DGS reconstruction loss: (1-λ) L1 + λ (1-SSIM).

    Args:
        gt_img: ground truth (H, W, 3) in [0, 1]
        rendered_img: rendered RGB (H, W, 3)
        lambda_dssim: weight on the D-SSIM term (default 0.2)

    Returns:
        scalar loss
    """
    if rendered_img.shape != gt_img.shape:
        raise ValueError(
            f"Shape mismatch: gt {tuple(gt_img.shape)} vs rendered {tuple(rendered_img.shape)}"
        )
    l1 = t.mean(t.abs(gt_img - rendered_img))
    ssim_val = ssim(_to_nchw(rendered_img), _to_nchw(gt_img), data_range=1.0)
    return (1.0 - lambda_dssim) * l1 + lambda_dssim * (1.0 - ssim_val)


def _spatial_gradients(img: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Channel-last (H, W, C) finite differences along width and height."""
    dx = img[:, 1:, :] - img[:, :-1, :]
    dy = img[1:, :, :] - img[:-1, :, :]
    return dx, dy


def calculate_iid_loss(
    gt_img: t.Tensor,  # (H, W, 3)
    r_albedo: t.Tensor,  # (H, W, 3)
    r_shading: t.Tensor,  # (H, W, 1) or (H, W, 3)
    r_img: t.Tensor | None = None,
    lambda_albedo: float = 0.1,
    lambda_shading: float = 0.05,
    lambda_chrom: float = 0.02,
) -> t.Tensor:
    """Optional IID regularizer when separate albedo/shading maps are available.

    Combines L1 reconstruction with image-guided albedo smoothness,
    shading smoothness, and optional chromaticity regularization.
    Not used by the default trainer — requires rendered albedo/shading buffers.
    """
    if r_img is None:
        r_img = r_albedo * r_shading

    if r_img.shape != gt_img.shape:
        raise ValueError(
            f"Shape mismatch: gt {tuple(gt_img.shape)} vs rendered {tuple(r_img.shape)}"
        )

    l_recon = t.mean(t.abs(gt_img - r_img))

    albedo_dx, albedo_dy = _spatial_gradients(r_albedo)
    gt_dx, gt_dy = _spatial_gradients(gt_img)
    weight_x = t.exp(-10.0 * t.abs(gt_dx).mean(dim=-1, keepdim=True))
    weight_y = t.exp(-10.0 * t.abs(gt_dy).mean(dim=-1, keepdim=True))
    l_albedo = t.mean(weight_x * t.abs(albedo_dx)) + t.mean(weight_y * t.abs(albedo_dy))

    shading_dx, shading_dy = _spatial_gradients(r_shading)
    l_shading = t.mean(shading_dx ** 2) + t.mean(shading_dy ** 2)

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
