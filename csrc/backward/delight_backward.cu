#include "delight_backward.h"
#include <cuda_runtime.h>
#include "cuda_check.h"
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
    Y[0] = -SH_C1 * y;
    Y[1] =  SH_C1 * z;
    Y[2] = -SH_C1 * x;

    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    Y[3] = SH_C2_0 * xy;
    Y[4] = SH_C2_1 * yz;
    Y[5] = SH_C2_2 * (2.0f * zz - xx - yy);
    Y[6] = SH_C2_3 * xz;
    Y[7] = SH_C2_4 * (xx - yy);

    Y[8]  = SH_C3_0 * y * (3.0f * xx - yy);
    Y[9]  = SH_C3_1 * xy * z;
    Y[10] = SH_C3_2 * y * (4.0f * zz - xx - yy);
    Y[11] = SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    Y[12] = SH_C3_4 * x * (4.0f * zz - xx - yy);
    Y[13] = SH_C3_5 * z * (xx - yy);
    Y[14] = SH_C3_6 * x * (xx - 3.0f * yy);
}

// Accumulate dL/ddir from dL/dY[k] * ∂Y[k]/∂dir
__device__ inline void accumulate_dY_ddir(
    float x, float y, float z,
    const float* __restrict__ dL_dY,
    glm::vec3& dL_ddir
){
    float xx = x * x, yy = y * y, zz = z * z;

    // Degree 1
    dL_ddir.y += dL_dY[0] * (-SH_C1);
    dL_ddir.z += dL_dY[1] * ( SH_C1);
    dL_ddir.x += dL_dY[2] * (-SH_C1);

    // Degree 2
    dL_ddir.x += dL_dY[3] * (SH_C2_0 * y);
    dL_ddir.y += dL_dY[3] * (SH_C2_0 * x);

    dL_ddir.y += dL_dY[4] * (SH_C2_1 * z);
    dL_ddir.z += dL_dY[4] * (SH_C2_1 * y);

    dL_ddir.x += dL_dY[5] * (-2.0f * SH_C2_2 * x);
    dL_ddir.y += dL_dY[5] * (-2.0f * SH_C2_2 * y);
    dL_ddir.z += dL_dY[5] * ( 4.0f * SH_C2_2 * z);

    dL_ddir.x += dL_dY[6] * (SH_C2_3 * z);
    dL_ddir.z += dL_dY[6] * (SH_C2_3 * x);

    dL_ddir.x += dL_dY[7] * ( 2.0f * SH_C2_4 * x);
    dL_ddir.y += dL_dY[7] * (-2.0f * SH_C2_4 * y);

    // Degree 3
    dL_ddir.x += dL_dY[8] * (SH_C3_0 * y * 6.0f * x);
    dL_ddir.y += dL_dY[8] * (SH_C3_0 * (3.0f * xx - 3.0f * yy));

    dL_ddir.x += dL_dY[9] * (SH_C3_1 * y * z);
    dL_ddir.y += dL_dY[9] * (SH_C3_1 * x * z);
    dL_ddir.z += dL_dY[9] * (SH_C3_1 * x * y);

    dL_ddir.x += dL_dY[10] * (SH_C3_2 * y * (-2.0f * x));
    dL_ddir.y += dL_dY[10] * (SH_C3_2 * (4.0f * zz - xx - 3.0f * yy));
    dL_ddir.z += dL_dY[10] * (SH_C3_2 * y * 8.0f * z);

    dL_ddir.x += dL_dY[11] * (SH_C3_3 * z * (-6.0f * x));
    dL_ddir.y += dL_dY[11] * (SH_C3_3 * z * (-6.0f * y));
    dL_ddir.z += dL_dY[11] * (SH_C3_3 * (6.0f * zz - 3.0f * xx - 3.0f * yy));

    dL_ddir.x += dL_dY[12] * (SH_C3_4 * (4.0f * zz - 3.0f * xx - yy));
    dL_ddir.y += dL_dY[12] * (SH_C3_4 * x * (-2.0f * y));
    dL_ddir.z += dL_dY[12] * (SH_C3_4 * x * 8.0f * z);

    dL_ddir.x += dL_dY[13] * (SH_C3_5 * z * 2.0f * x);
    dL_ddir.y += dL_dY[13] * (SH_C3_5 * z * (-2.0f * y));
    dL_ddir.z += dL_dY[13] * (SH_C3_5 * (xx - yy));

    dL_ddir.x += dL_dY[14] * (SH_C3_6 * (3.0f * xx - 3.0f * yy));
    dL_ddir.y += dL_dY[14] * (SH_C3_6 * x * (-6.0f * y));
}

__global__ void backward_SH(
    int num_gaussians,
    float3 camera_position,
    const float3* __restrict__ colors_grad,
    const float3* __restrict__ mu,
    const float3* __restrict__ albedo,
    const float* __restrict__ k_j,
    float3* __restrict__ albedo_grad,
    float* __restrict__ k_j_grad,
    float3* __restrict__ mu_grad
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_gaussians) return;

    glm::vec3 dl_dcolors(
        colors_grad[idx].x,
        colors_grad[idx].y,
        colors_grad[idx].z
    );
    glm::vec3 mu_world(mu[idx].x, mu[idx].y, mu[idx].z);
    glm::vec3 alb(albedo[idx].x, albedo[idx].y, albedo[idx].z);

    glm::vec3 camera_world(camera_position.x, camera_position.y, camera_position.z);
    glm::vec3 v = mu_world - camera_world;
    float inv_len = rsqrtf(glm::dot(v, v) + 1e-8f);
    glm::vec3 direction = v * inv_len;

    float Y[15];
    compute_sh_basis_15(direction.x, direction.y, direction.z, Y);

    const float* i_coeff_ptr = k_j + (idx * 45);

    // Match forward: RGB = 0.5 + SH_C0 * f_dc + sum_k Y[k] * k_j[k]
    glm::vec3 final_rgb = 0.5f + SH_C0 * alb;
    for (int k = 0; k < 15; ++k) {
        final_rgb.x += Y[k] * i_coeff_ptr[k * 3 + 0];
        final_rgb.y += Y[k] * i_coeff_ptr[k * 3 + 1];
        final_rgb.z += Y[k] * i_coeff_ptr[k * 3 + 2];
    }

    glm::vec3 g(
        (final_rgb.x < 0.0f || final_rgb.x > 1.0f) ? 0.0f : dl_dcolors.x,
        (final_rgb.y < 0.0f || final_rgb.y > 1.0f) ? 0.0f : dl_dcolors.y,
        (final_rgb.z < 0.0f || final_rgb.z > 1.0f) ? 0.0f : dl_dcolors.z
    );

    albedo_grad[idx] = make_float3(g.x * SH_C0, g.y * SH_C0, g.z * SH_C0);

    float dL_dY[15];
    float* kj_grad_ptr = k_j_grad + (idx * 45);
    for (int k = 0; k < 15; ++k) {
        kj_grad_ptr[k * 3 + 0] = g.x * Y[k];
        kj_grad_ptr[k * 3 + 1] = g.y * Y[k];
        kj_grad_ptr[k * 3 + 2] = g.z * Y[k];
        dL_dY[k] =
            g.x * i_coeff_ptr[k * 3 + 0] +
            g.y * i_coeff_ptr[k * 3 + 1] +
            g.z * i_coeff_ptr[k * 3 + 2];
    }

    glm::vec3 dL_ddir(0.0f);
    accumulate_dY_ddir(direction.x, direction.y, direction.z, dL_dY, dL_ddir);

    // d(normalize)/dv = (I - d d^T) / ||v||
    float d_dot = glm::dot(direction, dL_ddir);
    glm::vec3 dL_dv = inv_len * (dL_ddir - direction * d_dot);
    mu_grad[idx] = make_float3(dL_dv.x, dL_dv.y, dL_dv.z);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launch_backward_SH(
    const torch::Tensor colors_grad,
    const torch::Tensor mu,
    const torch::Tensor albedo,
    const torch::Tensor k_j,
    const torch::Tensor camera_position
){
    CHECK_INPUT_FP32(colors_grad);
    CHECK_INPUT_FP32(mu);
    CHECK_INPUT_FP32(albedo);
    CHECK_INPUT_FP32(k_j);
    CHECK_INPUT_FP32(camera_position);
    TORCH_CHECK(camera_position.numel() >= 3, "camera_position must have 3 elements");

    int num_gaussians = mu.size(0);
    const int threads = 256;
    const int blocks = (num_gaussians + threads - 1) / threads;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mu.device());
    torch::Tensor albedo_grad = torch::zeros({num_gaussians, 3}, options);
    torch::Tensor k_j_grad = torch::zeros({num_gaussians, 15, 3}, options);
    torch::Tensor mu_grad = torch::zeros({num_gaussians, 3}, options);

    auto cam_cpu = camera_position.cpu().contiguous();
    const float* cam_ptr = cam_cpu.data_ptr<float>();
    float3 camera_world = make_float3(cam_ptr[0], cam_ptr[1], cam_ptr[2]);

    backward_SH<<<blocks, threads>>>(
        num_gaussians,
        camera_world,
        reinterpret_cast<const float3*>(colors_grad.data_ptr<float>()),
        reinterpret_cast<const float3*>(mu.data_ptr<float>()),
        reinterpret_cast<const float3*>(albedo.data_ptr<float>()),
        k_j.data_ptr<float>(),
        reinterpret_cast<float3*>(albedo_grad.data_ptr<float>()),
        k_j_grad.data_ptr<float>(),
        reinterpret_cast<float3*>(mu_grad.data_ptr<float>())
    );
    CUDA_CHECK(cudaGetLastError());

    return std::make_tuple(albedo_grad, k_j_grad, mu_grad);
}
