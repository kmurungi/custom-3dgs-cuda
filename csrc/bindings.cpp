#include <torch/extension.h>
#include "forward/projection.h"
#include "forward/rasterize.h"
#include "forward/SH.h"

#include "backward/delight_backward.h"
#include "backward/projection_backward.h"
#include "rbackward/asterize_backward.h"


//projection function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> project(
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
    const torch::Tensor depths,
    const torch::Tensor mean_2d,
    const torch::Tensor cov_2d,
    const torch::Tensor colors,
    const torch::Tensor alpha,
    int height,
    int width
){
    return launch_rasterization(
        num_gaussians,
        depths,
        mean_2d,
        cov_2d,
        colors,
        alpha,
        height,
        width
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>backwards_rasterization(
    const float grad_output
){
    return launch_backwards_rasterization(
        grad_output
    );
}

std::tuple<torch::Tensor, torch::Tensor>backwards_SH(
    const torch::Tensor colors, 
    const torch::Tensor alpha
){
    return launch_backwards_SH(
        colors, 
        alpha
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>backwards_projection(
    const torch::Tensor mean_2d, 
    const torch::Tensor cov_2d
){
    return launch_backwards_projection(
        mean_2d, 
        cov_2d
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    //forward kernels
    m.def("project", &project, "3DGS Projection (CUDA)");
    m.def("SH", &spherical_harmonics, "3DGS Spherical Harmonics (CUDA)");
    m.def("rasterize", &rasterize, "3DGS Tile Rasterization (CUDA)");

    //backwards kernels
    m.def("backwards_rasterization", &backwards_rasterization, "3DGS Backwards Rasterization (CUDA)"); 
    m.def("backwards_SH", &backwards_SH, "3DGS backwards Delighting (CUDA)"); 
    m.def("backwards_projection", &backwards_projection, "3DGS Backwards Projection (CUDA)"); 


}
