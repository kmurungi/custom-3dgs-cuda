#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>


__global__ void backward_rasterization(

){


}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> launch_backward_rasterization(
    const int num_gaussians, 
    const float3* grad_output, 
    

){
    // 1 thread per gaussian
    int numThreadsPerBlock = 256; 
    int numBlocks = (num_gaussians + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // memory allocation for outputs 
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mu3d.device());
    torch::Tensor dl_mean2d = torch::zeros({total_gaussians, 2}, options); 
    torch::Tensor dl_cov2d = torch::zeros({total_gaussians, 3}, options);
    torch::Tensor dl_colors = torch::zeros({total_gaussians, 3}, options);
    torch::Tensor dl_alpha = torch::zeros({total_gaussians, 1}, options);
    float 2* dl_mean2d_ptr = reinterpret_cast<float2*>(dl_mean2d.data_ptr<float>());
    float 3* dl_cov2d_ptr = reinterpret_cast<float3*>(dl_cov2d.data_ptr<float>());
    float 3* dl_colors_ptr = reinterpret_cast<float3*>(dl_colors.data_ptr<float>());
    float* dl_alpha_ptr = reinterpret_cast<float*>(dl_alpha.data_ptr<float>());


    backward_rasterization<<<numBlocks, numThreadsPerBlock>>>(
        num_gaussians, 
        grad_output, 


        //output locations
        dl_mean2d_ptr, 
        dl_cov2d_ptr, 
        dl_colors_ptr
        dl_alpha_ptr
    ); 


    return (
        dl_mean2d_ptr, 
        dl_cov2d_ptr, 
        dl_colors_ptr
        dl_alpha_ptr
    );
}

