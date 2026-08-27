import torch as t
from gaussian_splatting.scene.gaussian import gaussian

try:
    import custom_rasterizer_cuda as crc
except ImportError:
    crc = None


class RasterizeFunction(t.autograd.Function):
    """
    Full rasterization pipeline. Returns rendered images.
    """

    @staticmethod
    def forward(ctx, mu, q, s, alpha, albedo, k_j, cam_rotation, cam_translation, fx, fy, cx, cy, height, width):
        if crc is None:
            raise ImportError(
                "custom_rasterizer_cuda is not installed. "
                "Build it with: pip install -e ."
            )

        num_gaussians = mu.shape[0]
        mu = mu.contiguous()
        q = q.contiguous()
        s = s.contiguous()
        alpha = alpha.contiguous()
        albedo = albedo.contiguous()
        k_j = k_j.contiguous()
        cam_rotation = cam_rotation.contiguous()
        cam_translation = cam_translation.contiguous()

        fx_f, fy_f = float(fx), float(fy)
        cx_f, cy_f = float(cx), float(cy)
        img_h, img_w = int(height), int(width)

        cam_position = -t.matmul(cam_rotation.T, cam_translation)
        cam_position = cam_position.reshape(3).contiguous()

        mean_2d, cov_2d, depths = crc.project(
            num_gaussians, mu, q, s, cam_rotation, cam_translation, fx_f, fy_f, cx_f, cy_f
        )
        colors = crc.SH(num_gaussians, cam_position, mu, albedo, k_j)
        rendered_img, sorted_ids, tile_ranges, final_T, n_contrib = crc.rasterize(
            num_gaussians, depths, mean_2d, cov_2d, colors, alpha, img_h, img_w
        )

        ctx.save_for_backward(
            mean_2d, cov_2d, colors, alpha,
            sorted_ids, tile_ranges, final_T, n_contrib,
            mu, q, s, albedo, k_j,
            cam_rotation, cam_translation, cam_position
        )
        ctx.num_gaussians = num_gaussians
        ctx.height = img_h
        ctx.width = img_w
        ctx.fx = fx_f
        ctx.fy = fy_f
        ctx.cx = cx_f
        ctx.cy = cy_f

        return rendered_img

    @staticmethod
    def backward(ctx, grad_output):
        (
            mean_2d, cov_2d, colors, alpha,
            sorted_ids, tile_ranges, final_T, n_contrib,
            mu, q, s, albedo, k_j,
            cam_rotation, cam_translation, cam_position,
        ) = ctx.saved_tensors

        mean_2d_grad, cov_2d_grad, colors_grad, alpha_grad = crc.backwards_rasterization(
            grad_output.contiguous(),
            mean_2d, cov_2d, colors, alpha,
            sorted_ids, tile_ranges, final_T, n_contrib,
            ctx.num_gaussians, ctx.height, ctx.width,
        )
        albedo_grad, k_j_grad = crc.backwards_SH(
            colors_grad, mu, albedo, k_j, cam_position,
        )
        mu_grad, q_grad, s_grad = crc.backwards_projection(
            mean_2d_grad, cov_2d_grad,
            mu, q, s, cam_rotation, cam_translation,
            ctx.fx, ctx.fy, ctx.cx, ctx.cy,
        )
        
        return (
            mu_grad,  # mu
            q_grad,  # q
            s_grad,  # s
            alpha_grad,  # alpha
            albedo_grad,  # albedo
            k_j_grad,  # k_j
            None,  # cam_rotation
            None,  # cam_translation
            None,  # fx
            None,  # fy
            None,  # cx
            None,  # cy
            None,  # height
            None,  # width
        )


def rasterize(gaussians: gaussian, image: dict) -> t.Tensor:
    """Python entry point used by the training loop."""
    return RasterizeFunction.apply(
        gaussians.mu,
        gaussians.q,
        gaussians.s,
        gaussians.alpha,
        gaussians.A,
        gaussians.k_j,
        image["R"],
        image["T"],
        image["fx"],
        image["fy"],
        image["cx"],
        image["cy"],
        image["height"],
        image["width"],
    )
