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
    const torch::Tensor mean_2d, // N x 2
    const torch::Tensor cov_2d,  // N x 2 x 2
    const torch::Tensor colors,  // N x 3 
    const torch::Tensor alpha // N x 1 
    int height, 
    int width
){


    // tile based parallelization
    int threads = 256; //16 x 16 tiles 1 tile per thread
    int blocks = ; 

    // rendered img 
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mean_2d.device());
    torch::Tensor rendered_img = torch::zeros({height, width}, options);
    float2* rendered_img_ptr  = reinterpret_cast<float2*>(rendered_img.data_ptr<float>());


    tile_based_rasterization<<<blocks, threadsPerBlock>>>tile_based_rasterization(
        mean_2d, // N x 2
        cov_2d,  // N x 2 x 2
        colors,  // N x 3 
        alpha,  // N x 1 
        height, width, 
        rendered_img_ptr // Height x Width
    ); 
    return rendered_img; 
}