#include <torch/extension.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>


// Pre-computed SH Normalization Constants
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

__global__ void spherical_harmonics_kernel(
    int total_gaussians, 
    float3 camera_position,
    const float3* __restrict__ mu_3d,  // N x 3 
    const float3* __restrict__ albedo_coeff, // N x 3
    const float* __restrict__ illumination_coeff, // Tensor of shape (N, 15, 3) flattened
    float3* __restrict__ colors_ptr // N x 3 
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx >= total_gaussians) return; 

    glm::vec3 camera_world(camera_position.x, camera_position.y, camera_position.z); 
    glm::vec3 mu_world(mu_3d[idx].x, mu_3d[idx].y, mu_3d[idx].z); 
    glm::vec3 a_coeff(albedo_coeff[idx].x, albedo_coeff[idx].y, albedo_coeff[idx].z); 
    const float* i_coeff_ptr = illumination_coeff + (idx * 45);
    
    glm::vec3 dir = mu_world - camera_world; 
    glm::vec3 direction = glm::normalize(dir);

    // Inria 3DGS DC: RGB = 0.5 + SH_C0 * f_dc (+ higher-order bands)
    float Y[15];
    compute_sh_basis_15(direction.x, direction.y, direction.z, Y);

    glm::vec3 final_rgb = 0.5f + SH_C0 * a_coeff;
    for (int k = 0; k < 15; ++k) {
        final_rgb.x += Y[k] * i_coeff_ptr[k * 3 + 0];
        final_rgb.y += Y[k] * i_coeff_ptr[k * 3 + 1];
        final_rgb.z += Y[k] * i_coeff_ptr[k * 3 + 2];
    }

    colors_ptr[idx] = make_float3(
        fminf(fmaxf(final_rgb.x, 0.0f), 1.0f),
        fminf(fmaxf(final_rgb.y, 0.0f), 1.0f),
        fminf(fmaxf(final_rgb.z, 0.0f), 1.0f)
    );

}

torch::Tensor launch_spherical_harmonics_kernel(
    int total_gaussians, 
    const torch::Tensor camera_position, // 3 x 1
    const torch::Tensor mu_world, // N x 3 
    const torch::Tensor albedo_coeff, // N x 3 
    const torch::Tensor illumination_coeff // (N, 15, 3) flattened to 1 contiguous array 
){ 
    CHECK_INPUT_FP32(camera_position);
    CHECK_INPUT_FP32(mu_world);
    CHECK_INPUT_FP32(albedo_coeff);
    CHECK_INPUT_FP32(illumination_coeff);
    TORCH_CHECK(camera_position.numel() >= 3, "camera_position must have 3 elements");

    auto cam_cpu = camera_position.cpu().contiguous();
    const float* cam_ptr = cam_cpu.data_ptr<float>();
    float3 camera_world = make_float3(cam_ptr[0], cam_ptr[1], cam_ptr[2]);
    const float3* mu_world_ptr = reinterpret_cast<const float3*>(mu_world.data_ptr<float>()); 
    const float3* albedo_coeff_ptr = reinterpret_cast<const float3*>(albedo_coeff.data_ptr<float>()); 
    const float* illumination_coeff_ptr = reinterpret_cast<const float*>(illumination_coeff.data_ptr<float>()); 

    // output
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mu_world.device());
    torch::Tensor colors = torch::zeros({total_gaussians, 3}, options); // N x 3 
    float3* colors_ptr = reinterpret_cast<float3*>(colors.data_ptr<float>()); 

    const int threads = 256;
    const int blocks = (total_gaussians + threads - 1) / threads;

    spherical_harmonics_kernel<<<blocks, threads>>>(
        total_gaussians, camera_world,
        mu_world_ptr, albedo_coeff_ptr, illumination_coeff_ptr,
        colors_ptr
    );
    CUDA_CHECK(cudaGetLastError()); 

    return colors; 
}