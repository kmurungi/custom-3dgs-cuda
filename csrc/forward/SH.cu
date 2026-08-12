#include <torch/extension.h>
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>




void __global__ spherical_harmonics_kernel(
    int total_gaussians, 
    const float3* camera_position, // 3 x 1 vector 
    const float3* mu_world,  // N x 3 
    const float3* albedo_coeff, // N x 3
    const float3* illumination_coeff, // Tensor of shape (N, 15, 3)
    const float3* colors_ptr // N x 3 
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx > total_gaussians) return; 

    




}



torch::Tensor launch_spherical_harmonics_kernel(
    int total_gaussians, 
    const torch::Tensor camera_position, // 3 x 1
    const torch::Tensor mu_world, // N x 3 
    const torch::Tensor albedo_coeff, // N x 3 
    const torch::Tensor illumination_coeff // (N, 15, 3)
){ 
    /*
    TENSOR CHECKS
    */
    TORCH_CHECK(camera_position.is_cuda(), "camera_position must be a CUDA tensor");
    TORCH_CHECK(mu_world.is_cuda(), "mu_world must be a CUDA tensor");
    TORCH_CHECK(albedo_coeff.is_cuda(), "albedo_coeff must be a CUDA tensor");
    TORCH_CHECK(illumination_coeff.is_cuda(), "illumination_coeff must be a CUDA tensor");

    TORCH_CHECK(camera_position.is_contiguous(), "camera_position must be contiguous in memory");
    TORCH_CHECK(mu_world.is_contiguous(), "mu_world must be contiguous in memory");
    TORCH_CHECK(albedo_coeff.is_contiguous(), "albedo_coeff must be contiguous in memory");
    TORCH_CHECK(illumination_coeff.is_contiguous(), "illumination_coeff must be contiguous in memory");

    // extract float ptrs from tensors 
    const float3* camera_position_ptr = reinterpret_cast<const float3*>(camera_position.data_ptr<float>()); 
    const float3* mu_world_ptr = reinterpret_cast<const float3*>(mu_world.data_ptr<float>()); 
    const float3* albedo_coeff_ptr = reinterpret_cast<const float3*>(albedo_coeff.data_ptr<float>()); 
    const float3* illumination_coeff_ptr = reinterpret_cast<const float3*>(illumination_coeff.data_ptr<float>()); 

    // output
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mu3d.device());
    torch::Tensor colors = torch::zeros({total_gaussians, 3}, options); // N x 3 
    float3* colors_ptr = reinterpret_cast<float3*>(colors.data_ptr<float>()); 

    int threads = 256; 
    int blocks = (total_gaussians + threads - 1)/threads; 

    spherical_harmonics_kernel<<<threads, blocks>>>(
        total_gaussians, camera_position_ptr, 
        mu_world_ptr, albedo_coeff_ptr, illumination_coeff_ptr, 
        colors_ptr
    ); 

    return colors; 
}