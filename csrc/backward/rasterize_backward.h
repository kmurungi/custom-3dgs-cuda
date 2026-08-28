#pragma once
#include <tuple>
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
launch_backward_rasterization(
    const torch::Tensor grad_output,   // H x W x 3
    const torch::Tensor mean_2d,       // N x 2
    const torch::Tensor cov_2d,        // N x 3
    const torch::Tensor colors,        // N x 3
    const torch::Tensor alpha,         // N x 1
    const torch::Tensor sorted_ids,    // M
    const torch::Tensor tile_ranges,   // T x 2
    const torch::Tensor final_T,       // H x W
    const torch::Tensor n_contrib,     // H x W
    int num_gaussians,
    int height,
    int width
);
