#include "projection_backward.h"
#include <cuda_runtime.h>


__global__ void backward_projection(
    int num_gaussians,
    const float2* __restrict__ mean_2d_grad,
    const float3* __restrict__ cov_2d_grad,
    const float3* __restrict__ mu,
    const float4* __restrict__ q,
    const float3* __restrict__ s,
    const float3* __restrict__ camera_rotation,
    const float3* __restrict__ camera_translation,
    float fx, float fy, float cx, float cy,
    float3* __restrict__ mu_grad,
    float4* __restrict__ q_grad,
    float3* __restrict__ s_grad
){


}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launch_backward_projection(
    const torch::Tensor mean_2d_grad,
    const torch::Tensor cov_2d_grad,
    const torch::Tensor mu,
    const torch::Tensor q,
    const torch::Tensor s,
    const torch::Tensor camera_rotation,
    const torch::Tensor camera_translation,
    float fx, float fy, float cx, float cy
){
    int num_gaussians = mu.size(0);
    int numThreadsPerBlock = 256;
    int numBlocks = (num_gaussians + numThreadsPerBlock - 1) / numThreadsPerBlock;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mu.device());
    torch::Tensor mu_grad = torch::zeros({num_gaussians, 3}, options);
    torch::Tensor q_grad = torch::zeros({num_gaussians, 4}, options);
    torch::Tensor s_grad = torch::zeros({num_gaussians, 3}, options);

    backward_projection<<<numBlocks, numThreadsPerBlock>>>(
        num_gaussians,
        reinterpret_cast<const float2*>(mean_2d_grad.data_ptr<float>()),
        reinterpret_cast<const float3*>(cov_2d_grad.data_ptr<float>()),
        reinterpret_cast<const float3*>(mu.data_ptr<float>()),
        reinterpret_cast<const float4*>(q.data_ptr<float>()),
        reinterpret_cast<const float3*>(s.data_ptr<float>()),
        reinterpret_cast<const float3*>(camera_rotation.data_ptr<float>()),
        reinterpret_cast<const float3*>(camera_translation.data_ptr<float>()),
        fx, fy, cx, cy,
        reinterpret_cast<float3*>(mu_grad.data_ptr<float>()),
        reinterpret_cast<float4*>(q_grad.data_ptr<float>()),
        reinterpret_cast<float3*>(s_grad.data_ptr<float>())
    );

    return std::make_tuple(mu_grad, q_grad, s_grad);
}
