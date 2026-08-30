#include "projection_backward.h"
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>


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

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    // Mean path (mu_grad)
    glm::vec3 mu_world(mu[idx].x, mu[idx].y, mu[idx].z);
    glm::mat3 R_cam(
        camera_rotation[0].x, camera_rotation[1].x, camera_rotation[2].x,
        camera_rotation[0].y, camera_rotation[1].y, camera_rotation[2].y,
        camera_rotation[0].z, camera_rotation[1].z, camera_rotation[2].z
    );
    glm::vec3 t_cam(camera_translation[0].x, camera_translation[0].y, camera_translation[0].z);
    glm::vec3 mu_cam = R_cam * mu_world + t_cam;
    if (mu_cam.z <= 0.2f) return;
    float z = mu_cam.z;
    float z2 = z * z;
    float dx = mean_2d_grad[idx].x;
    float dy = mean_2d_grad[idx].y;
    glm::vec3 dmu_cam(
        (fx / z) * dx,
        (fy / z) * dy,
        (-fx * mu_cam.x / z2) * dx + (-fy * mu_cam.y / z2) * dy
    );
    glm::vec3 dmu_world = glm::transpose(R_cam) * dmu_cam;
    mu_grad[idx] = make_float3(dmu_world.x, dmu_world.y, dmu_world.z); 
    
    // Cov Path (q_grad, s_grad)
    

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
