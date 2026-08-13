#include <torch/extension.h>
#include "projection.h"
#include "rasterize.h"
#include "SH.h"


//projection function
std::tuple<torch::Tensor, torch::Tensor> project(
    int total_gaussians,
    const torch::Tensor mu3d,
    const torch::Tensor q3d,
    const torch::Tensor s3d,
    const torch::Tensor camera_rotation,
    const torch::Tensor camera_translation,
    float fx, float fy, float cx, float cy
){
    return project_gaussians_to_2d(
        total_gaussians,
        mu3d,
        q3d,
        s3d,
        camera_rotation,
        camera_translation,
        fx, fy, cx, cy);
}

//spherical harmonics function
torch::Tensor spherical_harmonics(
    int total_gaussians,
    const torch::Tensor camera_position,
    const torch::Tensor mu_world,
    const torch::Tensor albedo_coeff,
    const torch::Tensor illumination_coeff
){
    return launch_spherical_harmonics_kernel(
        total_gaussians,
        camera_position,
        mu_world,
        albedo_coeff,
        illumination_coeff
    );
}

//rasterize function
torch::Tensor rasterize(
    int num_gaussians,
    const torch::Tensor mean_2d,
    const torch::Tensor cov_2d,
    const torch::Tensor colors,
    const torch::Tensor alpha,
    int height,
    int width
){
    return launch_rasterization(
        num_gaussians,
        mean_2d,
        cov_2d,
        colors,
        alpha,
        height,
        width
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project", &project, "3DGS Projection (CUDA)");
    m.def("SH", &spherical_harmonics, "3DGS Spherical Harmonics (CUDA)");
    m.def("rasterize", &rasterize, "3DGS Tile Rasterization (CUDA)");
}
