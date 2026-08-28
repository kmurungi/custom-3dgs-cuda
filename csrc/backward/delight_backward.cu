#include "delight_backward.h"
#include <cuda_runtime.h>


__global__ void backward_SH(
    int num_gaussians,
    float3 camera_position,
    const float3* __restrict__ colors_grad,
    const float3* __restrict__ mu,
    const float3* __restrict__ albedo,
    const float* __restrict__ k_j,
    float3* __restrict__ albedo_grad,
    float* __restrict__ k_j_grad
){


}

std::tuple<torch::Tensor, torch::Tensor>
launch_backward_SH(
    const torch::Tensor colors_grad,
    const torch::Tensor mu,
    const torch::Tensor albedo,
    const torch::Tensor k_j,
    const torch::Tensor camera_position
){
    int num_gaussians = mu.size(0);
    int numThreadsPerBlock = 256;
    int numBlocks = (num_gaussians + numThreadsPerBlock - 1) / numThreadsPerBlock;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mu.device());
    torch::Tensor albedo_grad = torch::zeros({num_gaussians, 3}, options);
    torch::Tensor k_j_grad = torch::zeros({num_gaussians, 15, 3}, options);

    const float* cam_ptr = camera_position.data_ptr<float>();
    float3 camera_position_ = make_float3(cam_ptr[0], cam_ptr[1], cam_ptr[2]);

    backward_SH<<<numBlocks, numThreadsPerBlock>>>(
        num_gaussians,
        camera_position_,
        reinterpret_cast<const float3*>(colors_grad.data_ptr<float>()),
        reinterpret_cast<const float3*>(mu.data_ptr<float>()),
        reinterpret_cast<const float3*>(albedo.data_ptr<float>()),
        k_j.data_ptr<float>(),
        reinterpret_cast<float3*>(albedo_grad.data_ptr<float>()),
        k_j_grad.data_ptr<float>()
    );

    return std::make_tuple(albedo_grad, k_j_grad);
}
