#pragma once
#include <tuple>
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launch_backward_projection(
    const torch::Tensor mean_2d_grad,        // N x 2
    const torch::Tensor cov_2d_grad,         // N x 3
    const torch::Tensor mu,                  // N x 3
    const torch::Tensor q,                   // N x 4
    const torch::Tensor s,                   // N x 3
    const torch::Tensor camera_rotation,     // 3 x 3
    const torch::Tensor camera_translation,  // 3
    float fx, float fy, float cx, float cy
);
