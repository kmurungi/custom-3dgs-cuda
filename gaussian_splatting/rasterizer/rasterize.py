import torch as t
from gaussian_splatting.scene.gaussian import gaussian 
from gaussian_splatting.scene.camera import camera
import custom_rasterize_cuda as crc 


class rasterizeFunction(t.autograd.Function): 
    """
    Full Rasterization Pipeline, returns rendered images
    """
    
    @staticmethod 
    def forward(ctx, mu, q, s, alpha, Albedo, k_j, image): 
        
        proj = crc.projection(mu, q, s, image) # Projection
        sh = crc.SH( alpha, Albedo, k_j, image) # Spherical Harmonics

        rendered_img = crc.rasterize(proj, sh)
        ctx.save_for_backward(mu, q, s, Albedo, alpha, k_j)

        return rendered_img

    @staticmethod
    def backward(ctx, rendered_img): 

        mu, q, s, alpha, Albedo, k_j = ctx.saved_tensor

        return mu, q, s, alpha, Albedo, k_j, None
