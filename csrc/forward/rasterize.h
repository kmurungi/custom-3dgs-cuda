#pragma once
#include <torch/extension.h>

torch::Tensor launch_rasterization(
    int num_gaussians,                 // N
    const torch::Tensor mean_2d,       // N x 2
    const torch::Tensor cov_2d,        // N x 2 x 2 (or N x 3)
    const torch::Tensor colors,        // N x 3
    const torch::Tensor alpha,         // N x 1
    int height,
    int width
);
