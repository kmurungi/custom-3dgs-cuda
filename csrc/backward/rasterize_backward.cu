#include "rasterize_backward.h"
#include <cuda_runtime.h>


__global__ void backward_rasterization(
    int num_gaussians,
    int height,
    int width,
    const float3* __restrict__ grad_output,
    const float2* __restrict__ mean_2d,
    const float3* __restrict__ cov_2d,
    const float3* __restrict__ colors,
    const float* __restrict__ alpha,
    const int* __restrict__ sorted_ids,
    const int2* __restrict__ tile_ranges,
    const float* __restrict__ final_T,
    const int* __restrict__ n_contrib,
    float2* __restrict__ dl_mean2d,
    float3* __restrict__ dl_cov2d,
    float3* __restrict__ dl_colors,
    float* __restrict__ dl_alpha
){

    

}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_backward_rasterization(
    const torch::Tensor grad_output,
    const torch::Tensor mean_2d,
    const torch::Tensor cov_2d,
    const torch::Tensor colors,
    const torch::Tensor alpha,
    const torch::Tensor sorted_ids,
    const torch::Tensor tile_ranges,
    const torch::Tensor final_T,
    const torch::Tensor n_contrib,
    int num_gaussians,
    int height,
    int width
){
    int numThreadsPerBlock = 256;
    int numBlocks = (num_gaussians + numThreadsPerBlock - 1) / numThreadsPerBlock;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mean_2d.device());
    torch::Tensor dl_mean2d = torch::zeros({num_gaussians, 2}, options);
    torch::Tensor dl_cov2d = torch::zeros({num_gaussians, 3}, options);
    torch::Tensor dl_colors = torch::zeros({num_gaussians, 3}, options);
    torch::Tensor dl_alpha = torch::zeros({num_gaussians, 1}, options);

    backward_rasterization<<<numBlocks, numThreadsPerBlock>>>(
        num_gaussians,
        height,
        width,
        reinterpret_cast<const float3*>(grad_output.data_ptr<float>()),
        reinterpret_cast<const float2*>(mean_2d.data_ptr<float>()),
        reinterpret_cast<const float3*>(cov_2d.data_ptr<float>()),
        reinterpret_cast<const float3*>(colors.data_ptr<float>()),
        alpha.data_ptr<float>(),
        sorted_ids.data_ptr<int>(),
        reinterpret_cast<const int2*>(tile_ranges.data_ptr<int>()),
        final_T.data_ptr<float>(),
        n_contrib.data_ptr<int>(),
        reinterpret_cast<float2*>(dl_mean2d.data_ptr<float>()),
        reinterpret_cast<float3*>(dl_cov2d.data_ptr<float>()),
        reinterpret_cast<float3*>(dl_colors.data_ptr<float>()),
        dl_alpha.data_ptr<float>()
    );

    return std::make_tuple(dl_mean2d, dl_cov2d, dl_colors, dl_alpha);
}
