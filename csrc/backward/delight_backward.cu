#include "delight_backward.h"
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

constexpr float SH_C0 = 0.28209479177387814f;

constexpr float SH_C1 = 0.4886025119029199f;

constexpr float SH_C2_0 = 1.0925484305920792f;
constexpr float SH_C2_1 = -1.0925484305920792f;
constexpr float SH_C2_2 = 0.31539156525252005f;
constexpr float SH_C2_3 = -1.0925484305920792f;
constexpr float SH_C2_4 = 0.5462742152960396f;

constexpr float SH_C3_0 = -0.5900435899266435f;
constexpr float SH_C3_1 = 2.890611442640554f;
constexpr float SH_C3_2 = -0.4570457994644658f;
constexpr float SH_C3_3 = 0.3731763325901154f;
constexpr float SH_C3_4 = -0.4570457994644658f;
constexpr float SH_C3_5 = 1.445305721320277f;
constexpr float SH_C3_6 = -0.5900435899266435f;

// Evaluates the 15 higher-order basis polynomials given normalized direction (x, y, z)
__device__ inline void compute_sh_basis_15(float x, float y, float z, float* Y) {
    // Degree 1 (Indices 0..2)
    Y[0] = -SH_C1 * y;
    Y[1] =  SH_C1 * z;
    Y[2] = -SH_C1 * x;

    // Degree 2 (Indices 3..7)
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    Y[3] = SH_C2_0 * xy;
    Y[4] = SH_C2_1 * yz;
    Y[5] = SH_C2_2 * (2.0f * zz - xx - yy);
    Y[6] = SH_C2_3 * xz;
    Y[7] = SH_C2_4 * (xx - yy);

    // Degree 3 (Indices 8..14)
    Y[8]  = SH_C3_0 * y * (3.0f * xx - yy);
    Y[9]  = SH_C3_1 * xy * z;
    Y[10] = SH_C3_2 * y * (4.0f * zz - xx - yy);
    Y[11] = SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    Y[12] = SH_C3_4 * x * (4.0f * zz - xx - yy);
    Y[13] = SH_C3_5 * z * (xx - yy);
    Y[14] = SH_C3_6 * x * (xx - 3.0f * yy);
}

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= num_gaussians) return; 

    // unpack
    glm::vec3 dl_dcolors( 
        colors_grad[idx].x, 
        colors_grad[idx].y, 
        colors_grad[idx].z
    ); 
    glm::vec3 mu_world(
        mu[idx].x, 
        mu[idx].y, 
        mu[idx].z
    ); 
    glm::vec3 alb(
        albedo[idx].x, 
        albedo[idx].y, 
        albedo[idx].z
    ); 

    // recompute forward 
    glm::vec3 camera_world(camera_position.x, camera_position.y, camera_position.z);
    glm::vec3 direction = glm::normalize(mu_world - camera_world);

    float Y[15];
    compute_sh_basis_15(direction.x, direction.y, direction.z, Y);

    const float* i_coeff_ptr = k_j + (idx * 45);
    glm::vec3 illumination(0.0f);
    for (int k = 0; k < 15; ++k) {
        illumination.x += Y[k] * i_coeff_ptr[k * 3 + 0];
        illumination.y += Y[k] * i_coeff_ptr[k * 3 + 1];
        illumination.z += Y[k] * i_coeff_ptr[k * 3 + 2];
    }

    glm::vec3 base_albedo = alb * SH_C0;
    glm::vec3 final_rgb = base_albedo * illumination;

    // clamped gradient- zero channels that hit the [0,1] clamp in forward
    glm::vec3 g(
        (final_rgb.x < 0.0f || final_rgb.x > 1.0f) ? 0.0f : dl_dcolors.x,
        (final_rgb.y < 0.0f || final_rgb.y > 1.0f) ? 0.0f : dl_dcolors.y,
        (final_rgb.z < 0.0f || final_rgb.z > 1.0f) ? 0.0f : dl_dcolors.z
    );

    // color = (albedo * SH_C0) * illumination  =>  dL/dalbedo = g * illumination * SH_C0
    albedo_grad[idx] = make_float3(
        g.x * illumination.x * SH_C0,
        g.y * illumination.y * SH_C0,
        g.z * illumination.z * SH_C0
    );

    // dL/dk_j[k] = g * base_albedo * Y[k]
    float* kj_grad_ptr = k_j_grad + (idx * 45);
    for (int k = 0; k < 15; ++k) {
        kj_grad_ptr[k * 3 + 0] = g.x * base_albedo.x * Y[k];
        kj_grad_ptr[k * 3 + 1] = g.y * base_albedo.y * Y[k];
        kj_grad_ptr[k * 3 + 2] = g.z * base_albedo.z * Y[k];
    }

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
