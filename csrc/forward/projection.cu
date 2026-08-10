#include <torch/extension.h>
#include "projection.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>





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
    float3* __restrict__ cov2d                     // N x 3 symmetric 2x2 covariance (a, b, c)
){ 
    // 1 Thread per Gaussian
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_gaussians) return;

    // -------------------------------------------------------------------------
    // 1. MEAN PROJECTION
    // -------------------------------------------------------------------------
    
    // Load world position
    glm::vec3 mu_world(mu3d[idx].x, mu3d[idx].y, mu3d[idx].z);

    // Build Camera Rotation Matrix (R_cam) from 3 float3 rows
    glm::mat3 R_cam(
        camera_rotation[0].x, camera_rotation[1].x, camera_rotation[2].x, // Col 0
        camera_rotation[0].y, camera_rotation[1].y, camera_rotation[2].y, // Col 1
        camera_rotation[0].z, camera_rotation[1].z, camera_rotation[2].z  // Col 2
    );
    glm::vec3 t_cam(camera_translation[0].x, camera_translation[0].y, camera_translation[0].z);

    // World to Camera Space: p_cam = R_cam * p_world + t_cam
    glm::vec3 mu_cam = R_cam * mu_world + t_cam;

    // Frustum Cull: Gaussians behind near plane
    if (mu_cam.z <= 0.2f) return;

    // Camera to Screen Pixels (Perspective Divide)
    float2 screen_pos = make_float2(
        (fx * mu_cam.x) / mu_cam.z + cx, 
        (fy * mu_cam.y) / mu_cam.z + cy
    );
    mu2d[idx] = screen_pos;


    // -------------------------------------------------------------------------
    // 2. COVARIANCE PROJECTIONS
    // -------------------------------------------------------------------------

    //convert to glm matrices 
    glm::quat q(q3d[idx].w, q3d[idx].x, q3d[idx].y, q3d[idx].z);
    glm::quat q_norm = glm::normalize(q);
    glm::mat3 R = glm::toMat3(q_norm);
    
    glm::mat3 S(s3d[idx].x, 0, 0, 
                0, s3d[idx].y, 0, 
                0, 0, s3d[idx].z);

    glm::mat3 M = R * S; 

    glm::mat3 sigma_world = M * glm::transpose(M); 

    



}



//pass in set 
void project_gaussians_to_2d(
    torch::tensor mu, 
    torch::tensor q, 
    torch::tensor s, 
    torch::tensor img, 
    int num_gaussians, 

){
    //tensor checks
    TORCH_CHECK(mu.is_contiguous(), "3D means must be contiguous in memory");
    TORCH_CHECK(q.is_contiguous(), "quarternions must be contiguous in memory");
    TORCH_CHECK(s.is_contiguous(), "scales must be contiguous in memory");
    TORCH_CHECK(img.is_contiguous(), "imgs must be contiguous in memory");


    // 1 gaussian per thread
    dim3 numThreadsPerBlock = 32;
    dim3 numBlocks = (N + numThreadsPerBlock - 1)/numThreadsPerBlock; 

    return projection_fused<<<numBlocks, numThreadsPerBlock>>>(mu, q, s, img);
}