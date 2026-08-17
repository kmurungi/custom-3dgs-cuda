#include <torch/extension.h>
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>


//main projection kernel 
__global__ void projection_fused(
    int total_gaussians, 
    const float3* __restrict__ mu3d,               // N x 3
    const float4* __restrict__ q3d,                // N x 4 (w, x, y, z)
    const float3* __restrict__ s3d,                // N x 3
    const float3* __restrict__ camera_rotation,   // 3 x 3 matrix rows
    const float3* __restrict__ camera_translation,// 3 x 1 vector
    float fx, float fy, float cx, float cy, 
    float2* __restrict__ mu2d,                     // N x 2 screen pixels
    float3* __restrict__ cov2d,                    // N x 3 symmetric 2x2 covariance (a, b, c)
    float* __restrict__ depths                     // N camera-space z
){ 
    // 1 Thread per Gaussian
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_gaussians) return;

    // -------------------------------------------------------------------------
    // 1. MEAN PROJECTION
    // -------------------------------------------------------------------------
    
    // Load world position
    glm::vec3 mu_world(mu3d[idx].x, mu3d[idx].y, mu3d[idx].z);

    // Build Camera Rotation Matrix 
    glm::mat3 R_cam(
        camera_rotation[0].x, camera_rotation[1].x, camera_rotation[2].x, // Col 0
        camera_rotation[0].y, camera_rotation[1].y, camera_rotation[2].y, // Col 1
        camera_rotation[0].z, camera_rotation[1].z, camera_rotation[2].z  // Col 2
    );
    glm::vec3 t_cam(camera_translation[0].x, camera_translation[0].y, camera_translation[0].z);

    // World to Camera Space
    glm::vec3 mu_cam = R_cam * mu_world + t_cam;

    // Frustum Cull
    if (mu_cam.z <= 0.2f) return;

    depths[idx] = mu_cam.z;

    // Camera to Screen Pixels
    float2 screen_pos = make_float2(
        (fx * mu_cam.x) / mu_cam.z + cx, 
        (fy * mu_cam.y) / mu_cam.z + cy
    );
    mu2d[idx] = screen_pos;


    // -------------------------------------------------------------------------
    // 2. COVARIANCE PROJECTIONS
    // -------------------------------------------------------------------------

    //convert to glm matrices 
    glm::quat q(q3d[idx].x, q3d[idx].y, q3d[idx].z, q3d[idx].w); 
    glm::mat3 R = glm::toMat3(glm::normalize(q));

    
    // Exponential activation on scale to guarantee positive scale
    glm::mat3 S = glm::mat3(
        expf(s3d[idx].x), 0.0f,             0.0f, 
        0.0f,             expf(s3d[idx].y), 0.0f, 
        0.0f,             0.0f,             expf(s3d[idx].z)
    );

    // 3D Covariance in World Space
    glm::mat3 M = R * S; 
    glm::mat3 Sigma_world = M * glm::transpose(M);

    
    // Transform 3D Covariance to Camera Space
    glm::mat3 Sigma_cam = R_cam * Sigma_world * glm::transpose(R_cam);

    //  Projection Jacobian Matrix J
    float z2 = mu_cam.z * mu_cam.z;
    float J00 = fx / mu_cam.z;
    float J02 = -(fx * mu_cam.x) / z2;
    float J11 = fy / mu_cam.z;
    float J12 = -(fy * mu_cam.y) / z2;

    
    float a = J00 * (J00 * Sigma_cam[0][0] + J02 * Sigma_cam[2][0]) + 
              J02 * (J00 * Sigma_cam[0][2] + J02 * Sigma_cam[2][2]);
              
    float b = J00 * (J11 * Sigma_cam[0][1] + J12 * Sigma_cam[0][2]) + 
              J02 * (J11 * Sigma_cam[2][1] + J12 * Sigma_cam[2][2]);
              
    float c = J11 * (J11 * Sigma_cam[1][1] + J12 * Sigma_cam[1][2]) + 
              J12 * (J11 * Sigma_cam[2][1] + J12 * Sigma_cam[2][2]);

    // Apply 0.3px Anti-Aliasing to diagonals
    a += 0.3f;
    c += 0.3f;

    // Write upper-triangular 2D covariance components (a, b, c) since matrix is symetric
    cov2d[idx] = make_float3(a, b, c);

}

//pass in set 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> project_gaussians_to_2d(
    int total_gaussians, 
    const torch::Tensor  mu3d,               // N x 3
    const torch::Tensor  q3d,                // N x 4 (w, x, y, z)
    const torch::Tensor  s3d,                // N x 3
    const torch::Tensor  camera_rotation,   // 3 x 3 matrix rows
    const torch::Tensor  camera_translation,// 3 x 1 vector
    float fx, float fy, float cx, float cy
){
    // cuda tensor checks
    TORCH_CHECK(mu3d.is_cuda(), "3D means must be a CUDA tensor");
    TORCH_CHECK(q3d.is_cuda(), "quarternions must be a CUDA tensor");
    TORCH_CHECK(s3d.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(camera_rotation.is_cuda(), "camera rotation must be a CUDA tensor");
    TORCH_CHECK(camera_translation.is_cuda(), "camera translation must be a CUDA tensor");

    // contiguous checks
    TORCH_CHECK(mu3d.is_contiguous(), "3D means must be contiguous in memory");
    TORCH_CHECK(q3d.is_contiguous(), "quarternions must be contiguous in memory");
    TORCH_CHECK(s3d.is_contiguous(), "scales must be contiguous in memory");
    TORCH_CHECK(camera_rotation.is_contiguous(), "camera rotation must be contiguous in memory");
    TORCH_CHECK(camera_translation.is_contiguous(), "camera translation must be contiguous in memory");

    // extract float pointer from tensor
    const float3* mu3d_ptr = reinterpret_cast<const float3*>(mu3d.data_ptr<float>());
    const float4* q3d_ptr = reinterpret_cast<const float4*>(q3d.data_ptr<float>());
    const float3* s3d_ptr = reinterpret_cast<const float3*>(s3d.data_ptr<float>());
    const float3* camera_rotation_ptr = reinterpret_cast<const float3*>(camera_rotation.data_ptr<float>());
    const float3* camera_translation_ptr = reinterpret_cast<const float3*>(camera_translation.data_ptr<float>());

    // 1 gaussian per thread
    int numThreadsPerBlock = 256;
    int numBlocks = (total_gaussians + numThreadsPerBlock - 1)/numThreadsPerBlock;
    
    //output locations
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(mu3d.device());
    torch::Tensor mu2d_tensor  = torch::zeros({total_gaussians, 2}, options);
    torch::Tensor cov2d_tensor = torch::zeros({total_gaussians, 3}, options); // (a, b, c)
    torch::Tensor depths_tensor = torch::zeros({total_gaussians}, options);

    float2* mu2d_ptr  = reinterpret_cast<float2*>(mu2d_tensor.data_ptr<float>());
    float3* cov2d_ptr = reinterpret_cast<float3*>(cov2d_tensor.data_ptr<float>());
    float* depths_ptr = depths_tensor.data_ptr<float>();

    projection_fused<<<numBlocks, numThreadsPerBlock>>>(
        total_gaussians,
        mu3d_ptr, q3d_ptr, s3d_ptr, camera_rotation_ptr, camera_translation_ptr, 
        fx, fy, cx, cy, 
        mu2d_ptr, cov2d_ptr, depths_ptr
    );

    return std::make_tuple(mu2d_tensor, cov2d_tensor, depths_tensor);
}
