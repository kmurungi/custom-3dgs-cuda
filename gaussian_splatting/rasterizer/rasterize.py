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

        mean_2d, cov_2d = crc.project(
            num_gaussians, mu, q, s, cam_rotation, cam_translation, fx_f, fy_f, cx_f, cy_f
        )
        colors = crc.SH(num_gaussians, cam_position, mu, albedo, k_j)
        rendered_img = crc.rasterize(
            num_gaussians, mean_2d, cov_2d, colors, alpha, img_h, img_w
        )

        ctx.save_for_backward(mu, q, s, alpha, albedo, k_j)
        return rendered_img

    @staticmethod
    def backward(ctx, grad_output):
        # Placeholder until CUDA backward kernels are wired.
        mu, q, s, alpha, albedo, k_j = ctx.saved_tensors
        return (
            None,  # mu
            None,  # q
            None,  # s
            None,  # alpha
            None,  # albedo
            None,  # k_j
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
