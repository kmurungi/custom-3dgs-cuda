#include <torch/extension.h>
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

__global__ void tile_based_rasterization(
    const torch::Tensor mean_2d, // N x 2
    const torch::Tensor cov_2d,  // N x 2 x 2
    const torch::Tensor colors,  // N x 3 
    const torch::Tensor alpha // N x 1 
    int height, 
    int width, 
    float2* rendered_img_ptr // H x W

){ 
    



}


torch::Tensor launch_rasterization(
    int num_gaussians, // N 
    const torch::Tensor mean_2d, // N x 2
    const torch::Tensor cov_2d,  // N x 2 x 2
    const torch::Tensor colors,  // N x 3 
    const torch::Tensor alpha, // N x 1 
    int height, 
    int width
){
    // cuda tensor checks
    TORCH_CHECK(mean_2d.is_cuda(), "2D means must be a CUDA tensor");
    TORCH_CHECK(cov_2d.is_cuda(), "2D Cov must be a CUDA tensor");
    TORCH_CHECK(colors.is_cuda(), "colors must be a CUDA tensor");
    TORCH_CHECK(alpha.is_cuda(), "Alpha must be a CUDA tensor");

    // contiguous checks
    TORCH_CHECK(mean_2d.is_contiguous(), "2D means must be contiguous in memory");
    TORCH_CHECK(cov_2d.is_contiguous(), "2D Cov must be contiguous in memory");
    TORCH_CHECK(colors.is_contiguous(), "colors must be contiguous in memory");
    TORCH_CHECK(alpha.is_contiguous(), "Alpha must be contiguous in memory");

    // extract float pointer from tensor
    const float2* mean_2d_ptr = reinterpret_cast<const float2*>(mean_2d.data_ptr<float>());
    const float* cov_2d_ptr = cov_2d.data_ptr<float>();
    const float3* colors_ptr = reinterpret_cast<const float3*>(colors.data_ptr<float>());
    const float* alpha_ptr = alpha.data_ptr<float>();

    // tile based parallelization: 16 x 16 pixel tiles
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // rendered img 
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mean_2d.device());
    torch::Tensor rendered_img = torch::zeros({height, width, 3}, options);
    float3* rendered_img_ptr  = reinterpret_cast<float3*>(rendered_img.data_ptr<float>());

    tile_based_rasterization<<<numBlocks, threadsPerBlock>>>(
        num_gaussians, // N
        mean_2d_ptr, // N x 2
        cov_2d_ptr,  // N x 2 x 2
        colors_ptr,  // N x 3 
        alpha_ptr,  // N x 1 
        height, width, 
        rendered_img_ptr // Height x Width
    ); 
    return rendered_img; 
}