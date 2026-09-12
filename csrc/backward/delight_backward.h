#pragma once
#include <tuple>
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
launch_backward_SH(
    const torch::Tensor colors_grad,      // N x 3
    const torch::Tensor mu,               // N x 3
    const torch::Tensor albedo,           // N x 3
    const torch::Tensor k_j,              // N x 15 x 3
    const torch::Tensor camera_position   // 3
);
