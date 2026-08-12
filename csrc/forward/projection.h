#pragma once
#include <tuple>
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> project_gaussians_to_2d(
    int total_gaussians, 
    const torch::Tensor  mu3d,               // N x 3
    const torch::Tensor  q3d,                // N x 4 (w, x, y, z)
    const torch::Tensor  s3d,                // N x 3
    const torch::Tensor  camera_rotation,   // 3 x 3 matrix rows
    const torch::Tensor  camera_translation,// 3 x 1 vector
    float fx, float fy, float cx, float cy
);
