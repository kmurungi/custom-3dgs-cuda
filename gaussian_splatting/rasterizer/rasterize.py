import torch as t
from gaussian_splatting.scene.gaussian import gaussian 
from gaussian_splatting.scene.camera import camera
import custom_rasterize_cuda as crc 


class rasterizeFunction(t.autograd.Function): 
    """
    Full Rasterization Pipeline, returns rendered images
    """
    
    @staticmethod 
    def forward(ctx, gaussians: gaussian, image: dict): 

        num_gaussians = gaussians.N 
        mu = gaussians.mu 
        q = gaussians.q 
        s = gaussians.s 
        alpha = gaussians.alpha 
        Albedo = gaussians.A 
        k_j = gaussians.k_j

        

        # Ensure all input tensors are contiguous in VRAM for CUDA raw pointer casting
        mu = mu.contiguous()
        q = q.contiguous()
        s = s.contiguous()
        alpha = alpha.contiguous()
        Albedo = Albedo.contiguous()
        k_j = k_j.contiguous()

        # retrieve camera intrinsic/extrinsic info
        cam_rotation = image['R'].contiguous()
        cam_translation = image['T'].contiguous()
        fx, fy = float(image['fx']), float(image['fy'])
        cx, cy = image['cx'], image['cy']
        img_h, img_w = int(image['height']), int(image['width'])

        # Compute world-space camera position 
        cam_position = -t.matmul(cam_rotation.T, cam_translation)
        cam_position = cam_position.reshape(3).contiguous()
       

        mean_2d, cov_2d = crc.projection(num_gaussians, mu, q, s, cam_rotation, cam_translation, fx, fy, cx, cy) # Projection
        colors = crc.SH(num_gaussians, cam_position, mu, Albedo, k_j) # Spherical Harmonics

        rendered_img = crc.rasterize(mean_2d, cov_2d, colors)
        ctx.save_for_backward(mu, q, s, Albedo, alpha, k_j)

        return rendered_img

    @staticmethod
    def backward(ctx, rendered_img): 

        mu, q, s, alpha, Albedo, k_j = ctx.saved_tensor

        return mu, q, s, alpha, Albedo, k_j, None
