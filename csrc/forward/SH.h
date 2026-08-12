#pragma once
#include <torch/extension.h>

torch::Tensor launch_spherical_harmonics_kernel(
    int total_gaussians,
    const torch::Tensor camera_position,      // 3
    const torch::Tensor mu_world,             // N x 3
    const torch::Tensor albedo_coeff,         // N x 3
    const torch::Tensor illumination_coeff    // (N, 15, 3)
);
