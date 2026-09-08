#include "projection_backward.h"
#include <cuda_runtime.h>
#define GLM_FORCE_CUDA
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>


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

    // -------------------------------------------------------------------------
    // 1. MEAN BACKWARDS PROJECTION
    // -------------------------------------------------------------------------

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
    
    // -------------------------------------------------------------------------
    // 2. COVARIANCE BACKWARDS PROJECTIONS
    // -------------------------------------------------------------------------

    // construct gradient matrix     
    float a = cov_2d_grad[idx].x;
    float b = cov_2d_grad[idx].y;
    float c = cov_2d_grad[idx].z;  
    glm::mat2 dl_d2d(
        a, 0.5f * b, 
        0.5f * b, c
    );

    //construct Jacobian (matches forward 2x3 J)
    glm::mat3x2 J(
        fx / z,  0.0f,
        0.0f,    fy / z,
        -(fx * mu_cam.x) / z2,  -(fy * mu_cam.y) / z2
    );

    // construct S and R matrices (same as forward)
    float sx = expf(s[idx].x);
    float sy = expf(s[idx].y);
    float sz = expf(s[idx].z);
    glm::mat3 S = glm::mat3(
        sx, 0.0f, 0.0f,
        0.0f, sy, 0.0f,
        0.0f, 0.0f, sz
    );

    glm::quat quat = glm::normalize(glm::quat(q[idx].x, q[idx].y, q[idx].z, q[idx].w));
    glm::mat3 R = glm::toMat3(quat);
    glm::mat3 M = R * S;

    // chain rule
    glm::mat3 dl_dcam = glm::transpose(J) * dl_d2d * J; // to camera space
    glm::mat3 dl_dworld = glm::transpose(R_cam) * dl_dcam * R_cam; // to world space
    glm::mat3 dl_dM = 2.0f * dl_dworld * M;
    glm::mat3 dl_dR = dl_dM * S;
    glm::mat3 dl_dS = glm::transpose(R) * dl_dM;

    s_grad[idx] = make_float3(
        dl_dS[0][0] * sx,
        dl_dS[1][1] * sy,
        dl_dS[2][2] * sz
    );

    
    glm::mat3 dL_dRt = glm::transpose(dl_dR);
    float qw = quat.w;
    float qx = quat.x;
    float qy = quat.y;
    float qz = quat.z;
    float4 dq = make_float4(
        2.0f * qz * (dL_dRt[0][1] - dL_dRt[1][0]) + 2.0f * qy * (dL_dRt[2][0] - dL_dRt[0][2]) + 2.0f * qx * (dL_dRt[1][2] - dL_dRt[2][1]),
        2.0f * qy * (dL_dRt[1][0] + dL_dRt[0][1]) + 2.0f * qz * (dL_dRt[2][0] + dL_dRt[0][2]) + 2.0f * qw * (dL_dRt[1][2] - dL_dRt[2][1]) - 4.0f * qx * (dL_dRt[2][2] + dL_dRt[1][1]),
        2.0f * qx * (dL_dRt[1][0] + dL_dRt[0][1]) + 2.0f * qw * (dL_dRt[2][0] - dL_dRt[0][2]) + 2.0f * qz * (dL_dRt[1][2] + dL_dRt[2][1]) - 4.0f * qy * (dL_dRt[2][2] + dL_dRt[0][0]),
        2.0f * qw * (dL_dRt[0][1] - dL_dRt[1][0]) + 2.0f * qx * (dL_dRt[2][0] + dL_dRt[0][2]) + 2.0f * qy * (dL_dRt[1][2] + dL_dRt[2][1]) - 4.0f * qz * (dL_dRt[1][1] + dL_dRt[0][0])
    );

    // Backprop through normalize(q)
    float4 q_raw = q[idx];
    float inv_norm = 1.0f / sqrtf(q_raw.x * q_raw.x + q_raw.y * q_raw.y + q_raw.z * q_raw.z + q_raw.w * q_raw.w);
    float4 qhat = make_float4(qw, qx, qy, qz); // normalized, float4 (w,x,y,z)
    float dot = qhat.x * dq.x + qhat.y * dq.y + qhat.z * dq.z + qhat.w * dq.w;
    q_grad[idx] = make_float4(
        inv_norm * (dq.x - qhat.x * dot),
        inv_norm * (dq.y - qhat.y * dot),
        inv_norm * (dq.z - qhat.z * dot),
        inv_norm * (dq.w - qhat.w * dot)
    );

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
