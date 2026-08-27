#pragma once
#include <tuple>
#include <torch/extension.h>

// Returns:
//   rendered_img  [H, W, 3] float32
//   sorted_ids    [M]       int32   Gaussian IDs in tile-then-depth order
//   tile_ranges   [T, 2]    int32   per-tile [start, end) into sorted_ids
//   final_T       [H, W]    float32 leftover transmittance after blending
//   n_contrib     [H, W]    int32   1-based index of last contributing Gaussian
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_rasterization(
    int num_gaussians,                 // N
    const torch::Tensor depths,        // N camera-space z
    const torch::Tensor mean_2d,       // N x 2
    const torch::Tensor cov_2d,        // N x 3 upper-triangular [a, b, c]
    const torch::Tensor colors,        // N x 3
    const torch::Tensor alpha,         // N x 1
    int height,
    int width
);
